import os
import shutil
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from kneed import KneeLocator
from matplotlib import pyplot as plt
from numpy import unique
from pandas.core.frame import DataFrame
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# CONFIG
plt.style.use("fivethirtyeight")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

SILHOUETTE_BEST_CLUSTERS_NUMBER = 7

MIN_CLUSTERS_NUMBER = 3
MAX_CLUSTERS_NUMBER = 10


def remove_last_run_plots():
    shutil.rmtree('plots', ignore_errors=True)
    os.mkdir('plots')


def save_plot(plot, filename: str, prefix: str) -> None:
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


def load_data(nb_elements=9999999):
    con = sqlite3.connect("resources/olist.db")
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    res = cur.execute("SELECT * FROM customers where customer_id in (select customer_id from orders)")
    customers = res.fetchall()

    res = cur.execute("select order_id, review_score from order_reviews")
    reviews = res.fetchall()

    res = cur.execute("""SELECT o.order_id, o.customer_id, o.order_purchase_timestamp, oi.price
    FROM orders o
    inner join order_items oi on o.order_id = oi.order_id""")
    orders = res.fetchall()

    cur.close()
    con.close()

    sorted_reviews = {}
    for review in reviews:
        sorted_reviews.setdefault(review['order_id'], []).append(review['review_score'])

    sorted_orders = {}
    for order in [dict(order) for order in orders]:
        order['review_score'] = sorted_reviews[order['order_id']][0] if order['order_id'] in sorted_reviews else None
        sorted_orders.setdefault(order['customer_id'], []).append(order)

    clients = []
    for customer in customers:
        customer_orders = sorted_orders[customer['customer_id']] if customer['customer_id'] in sorted_orders else []
        if len(customer_orders) == 0:
            continue

        total_amount = sum([order['price'] for order in customer_orders])
        nb_products = len(customer_orders)

        order_timestamps = [order['order_purchase_timestamp'] for order in customer_orders]
        latest_purchase_date: datetime = datetime.strptime(max(order_timestamps), DATE_FORMAT)
        days_since_last_purchase = (datetime.now() - latest_purchase_date).days

        review_scores = [order['review_score'] for order in customer_orders if order['review_score'] is not None]
        if len(review_scores) > 0:
            average_review = sum(review_scores) / len(review_scores)
        else:
            average_review = 0

        if nb_products < 8:
            clients.append({
                # 'customer_id': customer['customer_id'],
                'average_review': average_review,
                'recency': days_since_last_purchase,
                'frequency': nb_products,
                'monetary_value': total_amount
            })

    return DataFrame(clients[:nb_elements])


def create_clusters_plot(df, x, y, strategy_name):
    plt.figure(figsize=(10, 8))
    plot = sns.scatterplot(x=x, y=y, hue="cluster", data=df)
    save_plot(plot, f"{x}_vs_{y}_clusters", strategy_name)


def fit_kmeans(scaled_features, kmeans_kwargs):
    sse = []
    silhouette_coefficients = []

    for k in range(1, MAX_CLUSTERS_NUMBER + 1):
        print(
            f"KMeans clustering with {k} cluster{'s' if k > 1 else ''} started at {datetime.now().strftime("%H:%M:%S")}")
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)

        visualize_clusters(scaled_df, kmeans.labels_, f"kmeans")

        sse.append(kmeans.inertia_)

        if k != 1:
            score = silhouette_score(scaled_features, kmeans.labels_)
            silhouette_coefficients.append(score)

    return sse, silhouette_coefficients


def create_sse_plot(sse, max_range):
    plt.figure(figsize=(10, 9))
    plot = sns.lineplot(DataFrame(sse), x=range(1, max_range), y=sse)

    plot.set_title("SSE curve")
    plot.set_xlabel("Number of Clusters")
    plot.set_ylabel("SSE")

    save_plot(plot, "elbow", "kmeans")


def create_silhouette_score_plot(silhouette_coefficients, max_range):
    # The silhouette coefficient is a measure of cluster cohesion and separation.
    # It quantifies how well a data point fits into its assigned cluster.
    plt.figure(figsize=(10, 9))
    plot = sns.lineplot(
        DataFrame(silhouette_coefficients), x=range(
            2, max_range), y=silhouette_coefficients)

    plot.set_title("Silhouette Coefficient curve")
    plot.set_xlabel("Number of Clusters")
    plot.set_ylabel("Silhouette Coefficient")

    save_plot(plot, "silhouette_coefficient", "kmeans")


def perform_kmeans_clustering(scaled_df):
    print("Starting KMEANS clustering.\n")

    kmeans_kwargs = {
        "init": "k-means++",
        "n_init": 50,
        "max_iter": 500,
        "random_state": 42}

    sse, silhouette_coefficients = fit_kmeans(scaled_df, kmeans_kwargs)
    create_sse_plot(sse, MAX_CLUSTERS_NUMBER + 1)
    create_silhouette_score_plot(silhouette_coefficients, MAX_CLUSTERS_NUMBER + 1)

    # Using the best clusters number from the Elbow found using the Knee locator
    kl = KneeLocator(range(1, MAX_CLUSTERS_NUMBER + 1), sse, curve="convex", direction="decreasing")
    print(f"Elbow found at iteration:{kl.elbow}.\n")

    return kl.elbow

    # kmeans = KMeans(n_clusters=kl.elbow, **kmeans_kwargs)
    # kmeans.fit(scaled_df)
    #
    # visualize_clusters(scaled_df, kmeans.labels_, "kmeans")
    #
    # # Using the best clusters number from the Silhouette coefficients
    # print(f"Using number of clusters:{SILHOUETTE_BEST_CLUSTERS_NUMBER} determined from the Silhouette coefficients.\n")
    # silhouette_kmeans = KMeans(n_clusters=SILHOUETTE_BEST_CLUSTERS_NUMBER, **kmeans_kwargs)
    # silhouette_kmeans.fit(scaled_df)
    #
    # visualize_clusters(scaled_df, silhouette_kmeans.labels_, "kmeans")


def reachability_plot(_df, model):
    reachability = model.reachability_[model.ordering_]
    labels = model.labels_[model.ordering_]
    unique_labels = set(labels)
    space = np.arange(len(_df))

    for k, col in zip(
            unique_labels, [
                "#00ADB5", "#FF5376", "#724BE5", "#FDB62F"]):
        xk = space[labels == k]
        rk = reachability[labels == k]
        plt.plot(xk, rk, col)
        plt.fill_between(xk, rk, color=col, alpha=0.5)

    plt.xticks(space, _df.index[model.ordering_], fontsize=10)
    plt.plot(space[labels == -1], reachability[labels == -1], "k.", alpha=0.3)

    plt.ylabel("Reachability Distance")
    plt.title("Reachability Plot")
    plt.show()
    plt.close()


def save_dendrogram_plot(scaled_df, method):
    os.makedirs("plots/hierarchical", exist_ok=True)

    clustering = linkage(scaled_df, method=method, metric="euclidean")
    dendrogram(clustering)
    plt.savefig(f"plots/hierarchical/{method}.png")


def perform_hierarchical_clustering(df, scaled_df):
    # https://www.datacamp.com/tutorial/introduction-hierarchical-clustering-python
    print("Starting hierarchical clustering.\n")

    # TODO: Il manque la partie de gauche sur les dendogram.

    save_dendrogram_plot(scaled_df, "complete")
    save_dendrogram_plot(scaled_df, "average")
    save_dendrogram_plot(scaled_df, "single")  # This one crashes.

    # 6 is ok, Kmeans is the one you should interprate most this one is mostly for show.
    hierarchical_cluster = AgglomerativeClustering(n_clusters=6, metric='euclidean', linkage='ward')
    labels = hierarchical_cluster.fit_predict(scaled_df)

    df["cluster"] = labels


def perform_density_based_clustering(scaled_df):
    perform_dbscan_clustering(scaled_df)
    perform_optics_clustering(scaled_df)


def perform_optics_clustering(scaled_df):
    print("Starting OPTICS clustering.\n")

    for min_samples in range(25, 150, 5):
        optics = OPTICS(min_samples=min_samples)
        optics.fit(scaled_df)
        labels = optics.labels_

        clusters_number = len(unique(labels))
        print(f"min_samples:{min_samples}, number of clusters:{clusters_number}")

        if MIN_CLUSTERS_NUMBER <= clusters_number <= MAX_CLUSTERS_NUMBER:
            visualize_clusters(scaled_df, labels, f"optics_min_samples_{min_samples}")


def perform_dbscan_clustering(scaled_df):
    print("Starting DBSCAN clustering.\n")

    for eps in np.arange(0.01, 2, 0.01):
        dbscan = DBSCAN(eps=eps, min_samples=100)
        dbscan.fit(scaled_df)
        labels = dbscan.labels_

        clusters_number = len(unique(labels))
        print(f"eps:{round(eps, 2)}, number of clusters:{clusters_number}")

        if MIN_CLUSTERS_NUMBER <= clusters_number <= MAX_CLUSTERS_NUMBER:
            visualize_clusters(scaled_df, labels, f"dbscan_eps_{round(eps, 2)}")


def visualize_clusters(scaled_df, labels, strategy_name):
    labels = pd.Categorical(labels)

    pca = PCA()
    pca_results = pca.fit_transform(scaled_df)
    pca_df = DataFrame(pca_results[:, :2], columns=['x', 'y'])
    pca_df['labels'] = pd.Categorical(labels)

    plt.figure(figsize=(10, 10))
    plot = sns.scatterplot(pca_df, x='x', y='y', hue="labels", palette="bright")

    plot.set_title(f'Scatter plot of clusters from strategy {strategy_name.upper()}')
    plot.set_xlabel(f'F1 ({round(100 * pca.explained_variance_ratio_[0], 1)}%)')
    plot.set_ylabel(f'F2 ({round(100 * pca.explained_variance_ratio_[1], 1)}%)')
    plot.grid(True)

    save_plot(plot, f"cluster_{strategy_name}_with_{len(labels.unique())}_clusters", "clusters")


def verify_form_and_stability_of_best_strategy(scaled_df, best_kmeans_number_of_clusters):
    print("Starting verification of form and stability of the best strategy.\n")

    for iteration in range(1, 11):
        kmeans = KMeans(n_clusters=best_kmeans_number_of_clusters, random_state=4242)
        kmeans.fit(scaled_df)
        labels = kmeans.labels_

        visualize_clusters(scaled_df, labels, f"final_kmeans_iteration_{iteration}")


if __name__ == '__main__':
    print("Let's start project 4.\n")

    remove_last_run_plots()

    # df: DataFrame = load_data()
    df: DataFrame = load_data(nb_elements=10000)
    print("Data loaded.\n")

    scaled_df = DataFrame(StandardScaler().fit_transform(df), columns=df.columns)

    # best_kmeans_number_of_clusters = perform_kmeans_clustering(scaled_df)

    # perform_density_based_clustering(scaled_df)

    perform_hierarchical_clustering(df, scaled_df)

    # verify_form_and_stability_of_best_strategy(scaled_df, best_kmeans_number_of_clusters)
