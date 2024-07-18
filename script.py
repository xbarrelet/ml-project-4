import os
import shutil
import sqlite3
from datetime import datetime

import dataframe_image as dfi
import numpy as np
import pandas as pd
import seaborn as sns
import squarify
from kneed import KneeLocator
from matplotlib import pyplot as plt
from pandas.core.frame import DataFrame
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# CONFIG
plt.style.use("fivethirtyeight")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


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

    res = cur.execute(
        "SELECT * FROM customers where customer_id in (select customer_id from orders)")
    customers = res.fetchall()

    res = cur.execute("select order_id, review_score from order_reviews")
    reviews = res.fetchall()

    sorted_reviews = {}
    for review in reviews:
        sorted_reviews.setdefault(
            review['order_id'],
            []).append(
            review['review_score'])

    res = cur.execute(
        """SELECT o.order_id, o.customer_id, o.order_purchase_timestamp, oi.price
    FROM orders o
    inner join order_items oi on o.order_id = oi.order_id""")
    orders = res.fetchall()

    sorted_orders = {}
    for order in [dict(order) for order in orders]:
        order['review_score'] = sorted_reviews[order['order_id']
                                               ][0] if order['order_id'] in sorted_reviews else None
        sorted_orders.setdefault(order['customer_id'], []).append(order)

    cur.close()
    con.close()

    clients = []
    for customer in customers:
        # for customer in customers:
        customer_orders = sorted_orders[customer['customer_id']] if customer['customer_id'] in sorted_orders else []
        if len(customer_orders) == 0:
            continue

        total_amount = sum([order['price'] for order in customer_orders])
        nb_products = len(customer_orders)

        order_timestamps = [order['order_purchase_timestamp']
                            for order in customer_orders]
        latest_purchase_date: datetime = datetime.strptime(
            max(order_timestamps), DATE_FORMAT)
        days_since_last_purchase = (datetime.now() - latest_purchase_date).days

        review_scores = [order['review_score']
                         for order in customer_orders if order['review_score'] is not None]
        if len(review_scores) > 0:
            average_review = sum(review_scores) / len(review_scores)
        else:
            average_review = 0

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


def fit_kmeans(scaled_features, max_range, kmeans_kwargs):
    sse = []
    silhouette_coefficients = []
    counter = 1

    for k in range(1, max_range):
        print(f"Pass:{counter} at {datetime.now().strftime("%H:%M:%S")}")
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)

        sse.append(kmeans.inertia_)

        if k != 1:
            score = silhouette_score(scaled_features, kmeans.labels_)
            silhouette_coefficients.append(score)

        counter += 1

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


def perform_kmeans_clustering(df, scaled_features):
    # https://realpython.com/k-means-clustering-python/
    print("Starting KMEANS clustering.\n")

    kmeans_kwargs = {
        "init": "k-means++",
        "n_init": 50,
        "max_iter": 500,
        "random_state": 42}
    max_range = 11

    sse, silhouette_coefficients = fit_kmeans(
        scaled_features, max_range, kmeans_kwargs)

    # Good answer entre 3 et 5, donc on se base la-dessus
    kl = KneeLocator(range(1, max_range), sse, curve="convex", direction="decreasing")
    print(f"\nelbow found at iteration:{kl.elbow}")
    # TODO: The silhouette coefficient should also be taken into account but produces a 7. Here we're looking for 3-5.
    # So produce the clustering for both and evaluate which is the best manually.

    create_sse_plot(sse, max_range)
    create_silhouette_score_plot(silhouette_coefficients, max_range)

    kmeans = KMeans(n_clusters=kl.elbow, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    df["cluster"] = pd.Categorical(kmeans.labels_)

    create_clusters_plots(df, "kmeans")


def create_clusters_plots(df, strategy_name: str):
    # Use a PCA, this current solution is not clear. 1 color per label.
    # With the same PCA you can change the colors, for example once with the labels
    # and once with the average price to try to correlate input values and result.
    create_clusters_plot(df, "recency", "frequency", strategy_name)
    create_clusters_plot(df, "recency", "monetary_value", strategy_name)
    create_clusters_plot(df, "recency", "average_review", strategy_name)
    create_clusters_plot(df, "frequency", "monetary_value", strategy_name)
    create_clusters_plot(df, "frequency", "average_review", strategy_name)
    create_clusters_plot(df, "monetary_value", "average_review", strategy_name)


def perform_dbscan_clustering(df, scaled_df, eps, min_samples, metric):
    print("Starting DBSCAN clustering.\n")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    dbscan.fit(scaled_df)

    df["cluster"] = dbscan.labels_

    create_clusters_plots(df, "dbscan")


def perform_optics_clustering(df, scaled_df, min_samples, metric):
    print("Starting OPTICS clustering.\n")
    optics = OPTICS(min_samples=min_samples, metric=metric)
    optics.fit(scaled_df)

    # TODO: Use better colors?
    # reachability_plot(df, optics)

    df["cluster"] = optics.labels_

    create_clusters_plots(df, "optics")


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


def save_rfm_stats(df: DataFrame):
    RFM_stats = df.groupby("RFM_Level").agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary_value': ['mean', 'count']
    }).round(1)
    RFM_stats.columns = RFM_stats.columns.droplevel()
    RFM_stats.columns = [
        'Recency_Mean',
        'Frequency_Mean',
        'MonetaryValue_Mean',
        'MonetaryValue_Count']

    dfi.export(
        RFM_stats,
        'plots/RFM/RFM_stats.png',
        table_conversion='matplotlib')
    return RFM_stats


def save_rfm_segments(RFM_stats):
    fig = plt.gcf()
    fig.set_size_inches(16, 9)

    plot = squarify.plot(
        sizes=RFM_stats['MonetaryValue_Count'],
        label=[
            'Premiere',
            'Champions',
            'Loyal',
            'Potential',
            'Promising',
            'Needs attention'],
        color=[
            "green",
            "orange",
            "purple",
            "maroon",
            "pink",
            "teal"],
        alpha=0.6)
    plot.set_title("RFM Segments")

    plot.get_figure().savefig(f"plots/RFM/RFM_segments.png")
    plt.close()


def visualize_data(df, prefix):
    create_visualization_plot_for_attribute(df, "Recency", prefix)
    create_visualization_plot_for_attribute(df, "Frequency", prefix)
    create_visualization_plot_for_attribute(df, "MonetaryValue", prefix)


def create_visualization_plot_for_attribute(df, attribute: str, prefix):
    plot = sns.displot(df[attribute.replace(
        "MonetaryValue", "Monetary_Value").lower()])
    plot.set_xlabels(attribute)
    plot.set_ylabels("Probability")
    save_plot(plot, attribute, f"visualization_{prefix}")


def add_rfm_columns(df):
    Rlabel = range(4, 0, -1)
    Mlabel = range(1, 5)

    df['R'] = pd.qcut(df['recency'], q=4, labels=Rlabel).values
    df['M'] = pd.qcut(df['monetary_value'], q=4, labels=Mlabel).values
    df['F'] = np.where(df['frequency'] == 1, 1, 2)  # Good enough

    df['RFM_Concat'] = df['R'].astype(str) + df['F'].astype(str) + df['M'].astype(str)
    df['Score'] = df[['R', 'F', 'M']].sum(axis=1)
    df['RFM_Level'] = df.apply(rfm_level, axis=1)

    return df


def rfm_level(df):
    if df['Score'] >= 9:
        return "Premiere"
    elif (df['Score'] >= 7) and (df['Score'] < 9):
        return 'Champions'
    elif (df['Score'] >= 6) and (df['Score'] < 7):
        return 'Loyal'
    elif (df['Score'] >= 5) and (df['Score'] < 6):
        return 'Potential'
    elif (df['Score'] >= 4) and (df['Score'] < 5):
        return 'Promising'
    elif (df['Score'] >= 3) and (df['Score'] < 4):
        return 'Needs attention'
    else:
        return 'Requires activation'


def visualize_rfm_segments(df):
    os.makedirs("plots/RFM", exist_ok=True)

    RFM_stats = save_rfm_stats(df)
    save_rfm_segments(RFM_stats)


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
    create_clusters_plots(df, "hierarchical")


def perform_density_based_clustering(df, scaled_df, true_labels):
    # https://www.atlantbh.com/clustering-algorithms-dbscan-vs-optics/
    print("Starting DBSCAN and OPTICS clustering.\n")

    max_ari_results = 0
    best_parameters = {}
    for min_samples in range(5, 25, 5):  # TODO: on veut du 100
        for eps in np.arange(0.01, 1, 0.01):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(scaled_df)
            ari_dbscan = round(adjusted_rand_score(true_labels, dbscan.labels_), 4)
            print("Adjusted rand score is", ari_dbscan, "when min sample is", min_samples, "and epsilon is", round(eps, 3))

            if ari_dbscan > max_ari_results:
                max_ari_results = ari_dbscan
                best_parameters = {"min_samples": min_samples, "eps": eps}
    print("Best parameters for DBSCAN are", best_parameters, "with ARI", max_ari_results)
    # Best parameters for DBSCAN are {'min_samples': 20, 'eps': np.float64(0.1)} with ARI 0.0195 10k et diff params
    # Best parameters for DBSCAN are {'min_samples': 5, 'eps': np.float64(0.03)} with ARI 0.05 10k
    # Best parameters for DBSCAN are {'min_samples': 15, 'eps': np.float64(0.03)} with ARI 0.0263 50k

    max_ari_results = 0
    best_parameters = {}
    for min_samples in range(5, 25, 5):
        optics = OPTICS(min_samples=min_samples)
        optics.fit(scaled_df)
        ari_dbscan = round(adjusted_rand_score(true_labels, optics.labels_), 4)
        print("Adjusted rand score is", ari_dbscan, "when min sample is", min_samples)

        if ari_dbscan > max_ari_results:
            max_ari_results = ari_dbscan
            best_parameters = {"min_samples": min_samples}
    print("Best parameters for OPTICS are", best_parameters, "with ARI", max_ari_results)
    # Best parameters for OPTICS are {'min_samples': 5} with ARI 0.0205 10k?
    # Best parameters for OPTICS are {'min_samples': 5} with ARI 0.0215 10k
    # Best parameters for OPTICS are {'min_samples': 15} with ARI 0.0022 50k

    # perform_dbscan_clustering(df, scaled_df, eps, min_samples, metric)
    # perform_optics_clustering(df, scaled_df, min_samples, metric)

    # TODO: Regarde combien de clusters I get, if I get tons like 100 it's not actionable.
    #  Take the same PCA, visualize the clusters and demonstrate that it doesn't make sense.


if __name__ == '__main__':
    print("Let's start project 4.\n")

    remove_last_run_plots()

    # Ideally you should extract the clients with more than 1 purchase in their own cluster and
    # then add it to the ML results but here they don't want that.

    # df: DataFrame = load_data()
    df: DataFrame = load_data(nb_elements=50000)
    print("Data loaded.\n")

    rfm_df = add_rfm_columns(df.copy())
    visualize_rfm_segments(rfm_df)
    print("RFM segmentation finished.\n")

    # With ML we don't aim towards these business rules
    label_encoder = LabelEncoder()
    true_labels = label_encoder.fit_transform(rfm_df['RFM_Level'])

    customers_with_more_than_one_product = df[df['frequency'] > 1]
    print(f"Customers with more than one product:{len(customers_with_more_than_one_product)} on {len(df)}, " 
          f"or {round(len(customers_with_more_than_one_product) * 100 / len(df), 2)}%\n")

    visualize_data(df, "pre_scaling")
    scaled_df = DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
    visualize_data(scaled_df, "after_scaling")

    perform_kmeans_clustering(df, scaled_df)

    perform_density_based_clustering(df, scaled_df, true_labels)

    perform_hierarchical_clustering(df, scaled_df)

    # Ultimately, your decision on the number of clusters to use should be guided by a combination of domain knowledge
    # and clustering evaluation metrics.
