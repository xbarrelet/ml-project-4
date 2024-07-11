import os
import shutil
import sqlite3
from datetime import datetime

import pandas as pd
import seaborn as sns
from kneed import KneeLocator
from matplotlib import pyplot as plt
from pandas.core.frame import DataFrame
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#CONFIG
plt.style.use("fivethirtyeight")


def remove_last_run_plots():
    shutil.rmtree('plots', ignore_errors=True)
    os.mkdir('plots')


def save_plot(plot, filename: str, prefix: str) -> None:
    os.makedirs(f"plots/{prefix}", exist_ok=True)

    fig = plot.get_figure()
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

        if len(order_timestamps) == 1:
            order_frequency = 1
        else:
            order_earlier_date: datetime = datetime.strptime(min(order_timestamps), DATE_FORMAT)
            order_latest_date: datetime = datetime.strptime(max(order_timestamps), DATE_FORMAT)
            diff_days = (order_latest_date - order_earlier_date).days
            order_frequency = nb_products / diff_days if diff_days != 0 else 1

        review_scores = [order['review_score'] for order in customer_orders if order['review_score'] is not None]
        if len(review_scores) > 0:
            average_review = sum(review_scores) / len(review_scores)
        else:
            average_review = 0

        clients.append({
            # 'customer_id': customer['customer_id'],
            'average_review': average_review,
            'total_amount': total_amount,
            'nb_products': nb_products,
            'order_frequency': order_frequency
        })

    return DataFrame(clients[:nb_elements])


def create_clusters_plot(df, x, y):
    plt.figure(figsize=(10, 8))
    plot = sns.scatterplot(x=x, y=y, hue="cluster", data=df)
    save_plot(plot, f"{x}_vs_{y}_clusters", "kmeans")


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


def perform_kmeans_clustering(scaled_features):
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

    create_clusters_plot(df, "total_amount", "nb_products")
    create_clusters_plot(df, "total_amount", "order_frequency")
    create_clusters_plot(df, "nb_products", "order_frequency")
    create_clusters_plot(df, "nb_products", "average_review")
    create_clusters_plot(df, "total_amount", "average_review")
    create_clusters_plot(df, "order_frequency", "average_review")


if __name__ == '__main__':
    print("Let's start project 4\n")

    remove_last_run_plots()

    # Il te faut du quantitatif la. Dans ton analyse montre que seulement un petit % des clients a plus d'1 commande.
    # Ideally you should extract the clients with more than 1 purchase in their own cluster and
    # then add it to the ML results but here they don't want that.

    # https://realpython.com/k-means-clustering-python/

    # df: DataFrame = load_data()
    df: DataFrame = load_data(nb_elements=1000)
    print("Data loaded\n")

    customers_with_more_than_one_product = df[df['nb_products'] > 1]
    print(f"Customers with more than one product:{len(customers_with_more_than_one_product)} on {len(df)}, or {len(customers_with_more_than_one_product) * 100 / len(df)}%\n")

    scaled_df = StandardScaler().fit_transform(df)

    # perform_kmeans_clustering(scaled_df)



    # https://www.atlantbh.com/clustering-algorithms-dbscan-vs-optics/

    eps = 0.5
    min_samples = 5
    metric = "euclidean"

    # dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    # dbscan.fit(scaled_df)
    # df["cluster"] = dbscan.labels_ / 2

    optics = OPTICS(min_samples=min_samples, metric=metric)
    optics.fit(df)
    df["cluster"] = optics.labels_

    create_clusters_plot(df, "total_amount", "nb_products")

    # Ultimately, your decision on the number of clusters to use should be guided by a combination of domain knowledge
    # and clustering evaluation metrics.

    # Essaie le hierarchical clustering et le DABthingy.
