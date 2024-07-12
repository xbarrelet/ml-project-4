import math
import os
import shutil
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import squarify
from kneed import KneeLocator
from matplotlib import pyplot as plt
from pandas.core.frame import DataFrame
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import dataframe_image as dfi

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
    else:
        fig = plot._figure

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

    sorted_reviews = {}
    for review in reviews:
        sorted_reviews.setdefault(review['order_id'], []).append(review['review_score'])

    res = cur.execute("""SELECT o.order_id, o.customer_id, o.order_purchase_timestamp, oi.price
    FROM orders o 
    inner join order_items oi on o.order_id = oi.order_id""")
    orders = res.fetchall()

    sorted_orders = {}
    for order in [dict(order) for order in orders]:
        order['review_score'] = sorted_reviews[order['order_id']][0] if order['order_id'] in sorted_reviews else None
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

        order_timestamps = [order['order_purchase_timestamp'] for order in customer_orders]
        latest_purchase_date: datetime = datetime.strptime(max(order_timestamps), DATE_FORMAT)
        days_since_last_purchase = (datetime.now() - latest_purchase_date).days

        review_scores = [order['review_score'] for order in customer_orders if order['review_score'] is not None]
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
    plot = sns.lineplot(DataFrame(silhouette_coefficients), x=range(2, max_range), y=silhouette_coefficients)

    plot.set_title("Silhouette Coefficient curve")
    plot.set_xlabel("Number of Clusters")
    plot.set_ylabel("Silhouette Coefficient")

    save_plot(plot, "silhouette_coefficient", "kmeans")


def perform_kmeans_clustering(df, scaled_features):
    print("Starting KMEANS clustering.\n")
    kmeans_kwargs = {"init": "k-means++", "n_init": 50, "max_iter": 500, "random_state": 42}
    max_range = 11

    sse, silhouette_coefficients = fit_kmeans(scaled_features, max_range, kmeans_kwargs)

    kl = KneeLocator(range(1, max_range), sse, curve="convex", direction="decreasing")
    print(f"\nelbow found at iteration:{kl.elbow}")
    # TODO: The silhouette coefficient should also be taken into account. What other method? One business is needed?

    create_sse_plot(sse, max_range)
    create_silhouette_score_plot(silhouette_coefficients, max_range)

    kmeans = KMeans(n_clusters=kl.elbow, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    df["cluster"] = pd.Categorical(kmeans.labels_)

    create_clusters_plots(df, "kmeans")


def create_clusters_plots(df, strategy_name: str):
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
    reachability_plot(df, optics)

    df["cluster"] = optics.labels_

    create_clusters_plots(df, "optics")


def reachability_plot(_df, model):
   reachability = model.reachability_[model.ordering_]
   labels = model.labels_[model.ordering_]
   unique_labels = set(labels)
   space = np.arange(len(_df))

   for k, col in zip(unique_labels, ["#00ADB5", "#FF5376", "#724BE5", "#FDB62F"]):
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
    RFM_stats.columns = ['Recency_Mean', 'Frequency_Mean', 'MonetaryValue_Mean', 'MonetaryValue_Count']

    dfi.export(RFM_stats, 'plots/RFM/RFM_stats.png', table_conversion='matplotlib')
    return RFM_stats


def save_rfm_segments(RFM_stats):
    fig = plt.gcf()
    fig.set_size_inches(16, 9)

    plot = squarify.plot(sizes=RFM_stats['MonetaryValue_Count'],
                         label=['Premiere', 'Champions', 'Loyal', 'Potential', 'Promising', 'Needs attention'],
                         color=["green", "orange", "purple", "maroon", "pink", "teal"],
                         alpha=0.6)
    plot.set_title("RFM Segments")

    plot.get_figure().savefig(f"plots/RFM/RFM_segments.png")
    plt.close()


def visualize_data(df, prefix):
    create_visualization_plot_for_attribute(df, "Recency", prefix)
    create_visualization_plot_for_attribute(df, "Frequency", prefix)
    create_visualization_plot_for_attribute(df, "MonetaryValue", prefix)


def create_visualization_plot_for_attribute(df, attribute: str, prefix):
    plot = sns.displot(df[attribute.replace("MonetaryValue", "Monetary_Value").lower()])
    plot.set_xlabels(attribute)
    plot.set_ylabels("Probability")
    save_plot(plot, attribute, f"visualization_{prefix}")


def add_rfm_columns(df):
    Rlabel = range(4, 0, -1)
    Mlabel = range(1, 5)

    df['R'] = pd.qcut(df['recency'], q=4, labels=Rlabel).values
    df['F'] = np.where(df['frequency'] == 1, 1, 2)
    df['M'] = pd.qcut(df['monetary_value'], q=4, labels=Mlabel).values

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


if __name__ == '__main__':
    # TODO: Run autopep8 --in-place --aggressive --aggressive to format  the code when ready
    print("Let's start project 4\n")

    remove_last_run_plots()

    # Il te faut du quantitatif la. Dans ton analyse montre que seulement un petit % des clients a plus d'1 commande.
    # Ideally you should extract the clients with more than 1 purchase in their own cluster and
    # then add it to the ML results but here they don't want that.

    # https://realpython.com/k-means-clustering-python/

    # df: DataFrame = load_data()
    df: DataFrame = load_data(nb_elements=10000)
    print("Data loaded\n")

    rfm_df = add_rfm_columns(df.copy())
    visualize_rfm_segments(rfm_df)

    customers_with_more_than_one_product = df[df['frequency'] > 1]
    print(f"Customers with more than one product:{len(customers_with_more_than_one_product)} on {len(df)}, "
          f"or {round(len(customers_with_more_than_one_product) * 100 / len(df), 2)}%\n")

    visualize_data(df, "pre_scaling")
    scaled_df = DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
    visualize_data(scaled_df, "after_scaling")

    # perform_kmeans_clustering(df, scaled_df)

    # https://www.atlantbh.com/clustering-algorithms-dbscan-vs-optics/

    eps = 0.5
    min_samples = 5
    metric = "euclidean"

    perform_dbscan_clustering(df, scaled_df, eps, min_samples, metric)
    perform_optics_clustering(df, scaled_df, min_samples, metric)

    # Ultimately, your decision on the number of clusters to use should be guided by a combination of domain knowledge
    # and clustering evaluation metrics.

    # Essaie le hierarchical clustering et le DABthingy.
