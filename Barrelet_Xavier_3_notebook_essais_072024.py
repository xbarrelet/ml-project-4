import itertools
import os
import shutil
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import unique
from pandas.core.frame import DataFrame
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# CONFIG
plt.style.use("fivethirtyeight")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

MIN_CLUSTERS_NUMBER = 2
# MAX_CLUSTERS_NUMBER = 10
MAX_CLUSTERS_NUMBER = 6

KMEANS_DEFAULT_ARGS = {
    "init": "k-means++",
    "n_init": 50,
    "max_iter": 500
}
BEST_KMEANS_CLUSTERS_NUMBER = 3


def remove_last_run_plots():
    """Removes the content of the saved plots."""
    shutil.rmtree('plots', ignore_errors=True)
    os.mkdir('plots')


def display_plot(plot, filename: str, prefix: str) -> None:
    os.makedirs(f"plots/{prefix}", exist_ok=True)

    # fig = plot.get_figure()
    if hasattr(plot, 'get_figure'):
        fig = plot.get_figure()
    elif hasattr(plot, '_figure'):
        fig = plot._figure
    else:
        fig = plot

    fig.savefig(f"plots/{prefix}/{filename}.png")
    plt.close()


def load_data(nb_elements=99999999):
    """Load the data from the db, extract the RFM and average review attributes and returns them."""
    con = sqlite3.connect("resources/olist.db")
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    res = cur.execute("""SELECT customer_id, customer_unique_id FROM customers
    where customer_id in (select customer_id from orders)""")
    customers = res.fetchall()

    res = cur.execute("select order_id, review_score from order_reviews")
    reviews = res.fetchall()

    res = cur.execute(
        """SELECT o.order_id, o.customer_id, o.order_purchase_timestamp, oi.price
    FROM orders o
    inner join order_items oi on o.order_id = oi.order_id""")
    orders = res.fetchall()

    res = cur.execute(
        "select order_id, payment_type from order_pymts where payment_type != 'not_defined'")
    payments = res.fetchall()

    cur.close()
    con.close()

    sorted_reviews = {}
    for review in reviews:
        sorted_reviews.setdefault(
            review['order_id'],
            []).append(
            review['review_score'])

    sorted_payments = {}
    for payment in payments:
        sorted_payments.setdefault(
            payment['order_id'],
            set()).add(
            payment['payment_type'])

    sorted_orders = {}
    for order in [dict(order) for order in orders]:
        order_id = order['order_id']
        order['review_score'] = sorted_reviews[order_id][0] if order['order_id'] in sorted_reviews else None

        if order_id in sorted_payments and len(sorted_payments[order_id]) > 1:
            if 'voucher' in sorted_payments[order_id]:
                sorted_payments[order_id].remove('voucher')
                payment_type = sorted_payments[order_id].pop()
                sorted_payments[order_id].add(payment_type + "_with_voucher")
            # only 1 case with debit_card + credit_card, I'm skipping it
            else:
                continue
        elif order_id not in sorted_payments:
            continue

        order['payment_type'] = sorted_payments[order_id] if order['order_id'] in sorted_payments else None
        sorted_orders.setdefault(order['customer_id'], []).append(order)

    sorted_customers = {}
    for customer in [dict(customer) for customer in customers]:
        sorted_customers.setdefault(
            customer['customer_unique_id'], []).append(
            customer['customer_id'])

    clients = []
    for customer_unique_id in sorted_customers.keys():
        customer_ids = sorted_customers[customer_unique_id]

        customer_orders = []
        for customer_id in customer_ids:
            customer_orders += sorted_orders[customer_id] if customer_id in sorted_orders else []

        if len(customer_orders) == 0:
            continue

        total_amount = sum([order['price'] for order in customer_orders])
        nb_products = len(customer_orders)

        order_timestamps = [
            datetime.strptime(
                order['order_purchase_timestamp'],
                DATE_FORMAT) for order in customer_orders]
        latest_purchase_date: datetime = max(order_timestamps)
        days_since_last_purchase = (datetime.now() - latest_purchase_date).days

        review_scores = [order['review_score']
                         for order in customer_orders if order['review_score'] is not None]
        if len(review_scores) > 0:
            average_review = sum(review_scores) / len(review_scores)
        else:
            average_review = 0

        payment_types = set(list(itertools.chain.from_iterable(
            [list(order['payment_type']) for order in customer_orders])
        ))
        if len(payment_types) == 1:
            payment_type = list(payment_types)[0]
        else:
            payment_type = "multiple"

        # Excludes 71 clients for a better visibility of the clusters
        if nb_products < 8:
            clients.append({
                'average_review': average_review,
                'recency': days_since_last_purchase,
                'frequency': nb_products,
                'monetary_value': total_amount,
                "payment_type": payment_type
            })

    return DataFrame(clients[:nb_elements])


def fit_kmeans(prepared_df):
    """Performs multiple Kmeans modelings and returns the SSE and silhouette scores."""
    ssd = []
    silhouette_coefficients = []

    for clusters_number in range(MIN_CLUSTERS_NUMBER, MAX_CLUSTERS_NUMBER + 1):
        print(f"KMeans clustering with {clusters_number} cluster{'s' if clusters_number > 1 else ''}, "
              f"started at {datetime.now().strftime("%H:%M:%S")}")

        kmeans = KMeans(n_clusters=clusters_number, **KMEANS_DEFAULT_ARGS)
        kmeans.fit(prepared_df)

        ssd.append(kmeans.inertia_)

        score = silhouette_score(prepared_df, kmeans.labels_)
        silhouette_coefficients.append(score)

        # visualize_clusters(scaled_features, best_model.labels_, f"kmeans")

    return ssd, silhouette_coefficients


def create_ssd_plot(sse):
    """Display the SSD plot."""
    plt.figure(figsize=(10, 9))
    plot = sns.lineplot(
        DataFrame(sse),
        x=range(
            MIN_CLUSTERS_NUMBER,
            MAX_CLUSTERS_NUMBER + 1),
        y=sse)

    plot.set_title("SSD curve")
    plot.set_xlabel("Number of Clusters")
    plot.set_ylabel("SSD")

    display_plot(plot, "elbow", "kmeans")


def create_silhouette_score_plot(silhouette_coefficients):
    """Generate a plot showing the silhouette score per cluster numbers and display it."""
    plt.figure(figsize=(10, 9))
    plot = sns.lineplot(
        DataFrame(silhouette_coefficients),
        x=range(
            MIN_CLUSTERS_NUMBER,
            MAX_CLUSTERS_NUMBER + 1),
        y=silhouette_coefficients)

    plot.set_title("Silhouette Coefficient curve")
    plot.set_xlabel("Number of Clusters")
    plot.set_ylabel("Silhouette Coefficient")

    display_plot(plot, "silhouette_coefficient", "kmeans")


def perform_kmeans_modeling(prepared_df):
    """Performs multiple Kmeans modelings, produces SSE and silhouette score plots
    and returns the best number of clusters based on the elbow method and its labels.
    """
    print("Starting KMEANS modeling.\n")

    ssd, silhouette_coefficients = fit_kmeans(prepared_df)
    create_ssd_plot(ssd)
    create_silhouette_score_plot(silhouette_coefficients)

    # kl = KneeLocator(range(MIN_CLUSTERS_NUMBER, MAX_CLUSTERS_NUMBER + 1), ssd, curve="convex", direction="decreasing")
    # print(f"\nElbow found at iteration:{kl.elbow}.\n")

    kmeans = KMeans(n_clusters=BEST_KMEANS_CLUSTERS_NUMBER, **KMEANS_DEFAULT_ARGS)
    kmeans.fit(prepared_df)
    return kmeans.labels_


def perform_hierarchical_modeling(scaled_df):
    """Performs hierarchical modeling."""
    print("\nStarting hierarchical modeling.\n")
    os.makedirs("plots/hierarchical", exist_ok=True)

    # Correct but not efficient method
    # for clusters_number in range(MIN_CLUSTERS_NUMBER, MAX_CLUSTERS_NUMBER + 1):
    #     hierarchical_cluster = AgglomerativeClustering(
    #         n_clusters=clusters_number, metric='euclidean', linkage='ward')
    #     labels = hierarchical_cluster.fit_predict(scaled_df)
    #
    #     visualize_clusters(scaled_df, labels, f"hierarchical")

    Z = linkage(scaled_df, 'ward')

    generate_dendrogram(Z)
    print("Dendrogram generated.\n")

    silhouette_coefficients = []
    for clusters_number in range(MIN_CLUSTERS_NUMBER, MAX_CLUSTERS_NUMBER + 1):
        print(
            f"Generating hierarchical clusters visualization with {clusters_number} clusters.")

        labels = fcluster(Z, clusters_number, criterion='maxclust')
        visualize_clusters(scaled_df, labels, f"hierarchical")

        score = silhouette_score(scaled_df, labels)
        silhouette_coefficients.append(score)

    create_silhouette_score_plot(silhouette_coefficients)


def generate_dendrogram(Z):
    """Generate dendrogram and display it"""
    plt.figure(figsize=(15, 15))
    dendrogram(Z, truncate_mode="level", p=6)

    plt.savefig(f"plots/hierarchical/dendrogram.png")
    plt.close()


def perform_density_based_modeling(scaled_df):
    """Performs DBSCAN and OPTICS modeling."""
    perform_dbscan_clustering(scaled_df)
    perform_optics_clustering(scaled_df)


def perform_dbscan_clustering(scaled_df):
    """Performs DBSCAN modeling."""
    print("\nStarting DBSCAN modeling.\n")

    for min_sample in range(100, 401, 50):
        # > 20 = 1 cluster
        for eps in np.arange(1, 21, 1):
            dbscan = DBSCAN(eps=eps, min_samples=min_sample)
            dbscan.fit(scaled_df)
            labels = dbscan.labels_

            clusters_number = len(unique(labels))
            print(
                f"DBSCAN with min_sample:{min_sample}, eps:{
                round(
                    eps,
                    2)} generated {clusters_number} clusters.")

            if MIN_CLUSTERS_NUMBER < clusters_number <= MAX_CLUSTERS_NUMBER:
                visualize_clusters(
                    scaled_df,
                    labels,
                    f"dbscan_min_samples_{min_sample}_eps_{
                    round(
                        eps,
                        2)}")


def perform_optics_clustering(scaled_df):
    """Performs OPTICS modeling."""
    print("\nStarting OPTICS modeling.\n")

    # < 125 = > 40 clusters.
    # > 350 = 1 cluster.
    for min_samples in range(125, 351, 25):
        optics = OPTICS(min_samples=min_samples)
        optics.fit(scaled_df)
        labels = optics.labels_

        clusters_number = len(unique(labels))
        print(
            f"OPTICS with min_samples:{min_samples} generated {clusters_number} clusters.")

        if MIN_CLUSTERS_NUMBER < clusters_number <= MAX_CLUSTERS_NUMBER:
            visualize_clusters(
                scaled_df,
                labels,
                f"optics_min_samples_{min_samples}")


def visualize_clusters(scaled_df, labels, strategy_name):
    """Generate a 2d TSNE graph showing the generated clusters and display the plot."""
    labels = pd.Categorical(labels)
    scaled_df['labels'] = labels

    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(scaled_df)
    scaled_df['tsne-x'] = tsne_results[:, 0]
    scaled_df['tsne-y'] = tsne_results[:, 1]

    plt.figure(figsize=(12, 10))
    plot = sns.scatterplot(
        scaled_df,
        x="tsne-x",
        y="tsne-y",
        hue="labels",
        palette="bright")

    plot.set_title(
        f'Scatter plot of clusters from strategy {
        strategy_name.upper()}')
    plot.set_xlabel(f'F1')
    plot.set_ylabel(f'F2')
    plot.grid(True)

    display_plot(plot,
                 f"{strategy_name.replace("_",
                                          " ")}_{len(labels.unique())}_clusters",
                 f"{strategy_name.split("_")[0]}")


def generate_ari_scores_plot(ari_scores):
    """Generate and display a plot showing the ARI scores by week number."""
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 12)
    plot = sns.lineplot(
        DataFrame(ari_scores),
        x="iteration",
        y="ari_score",
        ax=ax)

    average_score = round(
        sum(
            [score['ari_score'] for score in ari_scores]) /
        len(ari_scores),
        2)
    plot.set_title(f"ARI score per iteration, average score:{average_score}")
    plot.set_xlabel("Iteration")
    plot.set_ylabel("ARI score")
    ax.hlines(
        y=0.7,
        xmin=1,
        xmax=len(ari_scores),
        color='black',
        linestyles='dashdot')

    display_plot(plot, "ari_scores", "final")
    plt.close()


def verify_form_and_stability_of_best_strategy(prepared_df, original_labels):
    """Performs multiple kmeans modeling with the best clusters number to verify its form and result."""
    print("Starting verification of form and stability of the best strategy.\n")

    iterations_number = 100
    ari_scores = []
    for iteration in range(1, iterations_number + 1):
        kmeans = KMeans(n_clusters=BEST_KMEANS_CLUSTERS_NUMBER, **KMEANS_DEFAULT_ARGS)
        kmeans.fit(prepared_df)
        labels = kmeans.labels_

        # visualize_clusters(
        #     scaled_df,
        #     labels,
        #     f"final_kmeans_iteration_{iteration}")

        ari_score = round(adjusted_rand_score(labels, original_labels), 4)
        ari_scores.append({"ari_score": ari_score, "iteration": iteration})

    generate_ari_scores_plot(ari_scores)


def prepare_data(df):
    one_hot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False)

    payment_type_df = one_hot_encoder.fit_transform(
        DataFrame(df["payment_type"]))
    payment_type_df = DataFrame(
        payment_type_df,
        columns=one_hot_encoder.get_feature_names_out())
    payment_type_df.index = df.index

    df.drop("payment_type", axis=1, inplace=True)
    scaled_df = DataFrame(
        StandardScaler().fit_transform(df),
        columns=df.columns)
    encoded_df = pd.concat([scaled_df, payment_type_df], axis=1)

    return encoded_df


if __name__ == '__main__':
    print("Starting modeling script.\n")

    remove_last_run_plots()

    # df: DataFrame = load_data(nb_elements=15000)
    df: DataFrame = load_data()
    print("Data loaded.\n")

    prepared_df: DataFrame = prepare_data(df)
    # smaller_prepared_df: DataFrame = prepared_df.sample(n=10000, random_state=42)
    smaller_prepared_df: DataFrame = prepared_df.sample(n=40000, random_state=42)

    kmeans_labels = perform_kmeans_modeling(prepared_df)

    # perform_density_based_modeling(smaller_prepared_df)

    # perform_hierarchical_modeling(smaller_prepared_df)

    verify_form_and_stability_of_best_strategy(prepared_df, kmeans_labels)

    print("All processing is now done.")
