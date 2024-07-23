import os
import shutil
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pandas.core.frame import DataFrame
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# CONFIG
plt.style.use("fivethirtyeight")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

BEST_KMEANS_CLUSTERS_NUMBER = 4

kmeans_kwargs = {
    "init": "k-means++",
    "n_init": 50,
    "max_iter": 500,
    "random_state": 42
}


def remove_last_run_plots():
    shutil.rmtree('simulation_plots', ignore_errors=True)
    os.mkdir('simulation_plots')


def load_data(nb_elements=9999999):
    con = sqlite3.connect("resources/olist.db")
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    res = cur.execute("SELECT customer_id FROM customers where customer_id in (select customer_id from orders)")
    customers = res.fetchall()

    res = cur.execute("""SELECT o.order_id, o.customer_id, o.order_purchase_timestamp, oi.price
    FROM orders o
    inner join order_items oi on o.order_id = oi.order_id""")
    orders = res.fetchall()

    cur.close()
    con.close()

    sorted_orders = {}
    for order in [dict(order) for order in orders]:
        sorted_orders.setdefault(order['customer_id'], []).append(order)

    clients = []
    overall_earliest_purchase_date = None
    overall_latest_purchase_date = None
    for customer in customers:
        customer_orders = sorted_orders[customer['customer_id']] if customer['customer_id'] in sorted_orders else []
        if len(customer_orders) == 0:
            continue

        total_amount = sum([order['price'] for order in customer_orders])
        nb_products = len(customer_orders)

        order_timestamps = [datetime.strptime(order['order_purchase_timestamp'], DATE_FORMAT)
                            for order in customer_orders]
        latest_purchase_date: datetime = max(order_timestamps)
        earliest_purchase_date: datetime = min(order_timestamps)
        days_since_last_purchase = (datetime.now() - latest_purchase_date).days

        if overall_earliest_purchase_date is None or overall_earliest_purchase_date > earliest_purchase_date:
            overall_earliest_purchase_date = earliest_purchase_date

        if overall_latest_purchase_date is None or overall_latest_purchase_date < latest_purchase_date:
            overall_latest_purchase_date = latest_purchase_date

        # 36 customers are excluded and that eliminates some outliers in the PCA graphs and makes them more readable
        if nb_products < 8:
            clients.append({
                # 'customer_id': customer['customer_id'],
                'recency': days_since_last_purchase,
                'frequency': nb_products,
                'monetary_value': total_amount
            })

    return clients[:nb_elements], overall_earliest_purchase_date, overall_latest_purchase_date


def get_first_timeperiod_clients(overall_earliest_purchase_date, weeks_number, clients):
    days_since_first_period_start = (datetime.now() - overall_earliest_purchase_date).days

    first_period_end_date = overall_earliest_purchase_date + timedelta(weeks=weeks_number)
    days_since_first_period_end = (datetime.now() - first_period_end_date).days

    return [client for client in clients if days_since_first_period_end < client['recency']
            <= days_since_first_period_start]


def get_second_timeperiod_clients(overall_earliest_purchase_date, weeks_number, clients):
    second_period_start_date = overall_earliest_purchase_date + timedelta(weeks=weeks_number)
    days_since_second_period_start = (datetime.now() - second_period_start_date).days

    second_period_end_date = overall_earliest_purchase_date + timedelta(weeks=2 * weeks_number)
    days_since_second_period_end = (datetime.now() - second_period_end_date).days

    return [client for client in clients if days_since_second_period_end < client['recency']
            <= days_since_second_period_start]


def get_kmeans_model_fit_on_clients(clients):
    scaled_clients = DataFrame(StandardScaler().fit_transform(DataFrame(clients)))
    model = KMeans(n_clusters=BEST_KMEANS_CLUSTERS_NUMBER, **kmeans_kwargs)
    model.fit(scaled_clients)

    return model


def create_ari_scores_plot(ari_results):
    plt.figure(figsize=(10, 9))
    plot = sns.lineplot(DataFrame(ari_results), x="weeks_number", y="ari_score")

    plot.set_title("ARI score per weeks")
    plot.set_xlabel("Weeks number")
    plot.set_ylabel("ARI score")

    fig = plot.get_figure()
    fig.savefig(f"simulation_plots/ari_scores.png")
    plt.close()


if __name__ == '__main__':
    print("Starting simulation script.\n")
    remove_last_run_plots()

    clients, overall_earliest_purchase_date, overall_latest_purchase_date = load_data()
    print("Data loaded.\n")

    overall_week_numbers = (overall_latest_purchase_date - overall_earliest_purchase_date).days // 7
    print(f"Earliest purchase date:{overall_earliest_purchase_date}, "
          f"latest purchase date:{overall_latest_purchase_date}, overall week numbers:{overall_week_numbers}.\n")

    ari_results = []
    for weeks_number in range(1, int(overall_week_numbers / 2 + 1)):
        print(f"Considering periods of {weeks_number} week{'s' if weeks_number > 1 else ''}.")

        first_timeperiod_clients = get_first_timeperiod_clients(overall_earliest_purchase_date, weeks_number, clients)
        second_timeperiod_clients = get_second_timeperiod_clients(overall_earliest_purchase_date, weeks_number, clients)
        all_periods_clients = first_timeperiod_clients + second_timeperiod_clients

        if (len(first_timeperiod_clients) < BEST_KMEANS_CLUSTERS_NUMBER or
                len(second_timeperiod_clients) < BEST_KMEANS_CLUSTERS_NUMBER):
            print("Not enough clients in one of the time periods, skipping this iteration.\n")
            continue
        else:
            print(f"Number of clients in the first time period: {len(first_timeperiod_clients)}, "
                  f"number of clients in the second time period: {len(second_timeperiod_clients)}.")

        model_a = get_kmeans_model_fit_on_clients(first_timeperiod_clients)
        model_b = get_kmeans_model_fit_on_clients(all_periods_clients)

        scaled_second_period_clients = DataFrame(StandardScaler().fit_transform(DataFrame(second_timeperiod_clients)))
        model_a_labels = model_a.predict(scaled_second_period_clients)
        model_b_labels = model_b.predict(scaled_second_period_clients)

        ari_score = round(adjusted_rand_score(model_a_labels, model_b_labels), 4)
        ari_results.append({'weeks_number': weeks_number, 'ari_score': ari_score})
        print(f"ARI score:{ari_score} for weeks number:{weeks_number}.\n")

    create_ari_scores_plot(ari_results)

    print("All processing is now done.")

"""
Considering periods of 33 weeks.
Number of clients in the first time period: 7122, number of clients in the second time period: 34331.
ARI score:0.3011 for weeks number:33.

Considering periods of 34 weeks.
Number of clients in the first time period: 7802, number of clients in the second time period: 36144.
ARI score:0.8682 for weeks number:34.

Considering periods of 35 weeks.
Number of clients in the first time period: 8536, number of clients in the second time period: 37530.
ARI score:0.8893 for weeks number:35.

Considering periods of 36 weeks.
Number of clients in the first time period: 9328, number of clients in the second time period: 40238.
ARI score:0.873 for weeks number:36.

Considering periods of 37 weeks.
Number of clients in the first time period: 10244, number of clients in the second time period: 42495.
ARI score:0.9221 for weeks number:37.

Considering periods of 38 weeks.
Number of clients in the first time period: 11046, number of clients in the second time period: 44895.
ARI score:0.9651 for weeks number:38.

Considering periods of 39 weeks.
Number of clients in the first time period: 11870, number of clients in the second time period: 47675.
ARI score:0.9434 for weeks number:39.

Considering periods of 40 weeks.
Number of clients in the first time period: 12697, number of clients in the second time period: 50095.
ARI score:0.9471 for weeks number:40.

Considering periods of 41 weeks.
Number of clients in the first time period: 13463, number of clients in the second time period: 52519.
ARI score:0.9349 for weeks number:41.

Considering periods of 42 weeks.
Number of clients in the first time period: 14108, number of clients in the second time period: 54992.
ARI score:0.9335 for weeks number:42.

Considering periods of 43 weeks.
Number of clients in the first time period: 14828, number of clients in the second time period: 57662.
ARI score:0.9076 for weeks number:43.

Considering periods of 44 weeks.
Number of clients in the first time period: 15657, number of clients in the second time period: 60550.
ARI score:0.9044 for weeks number:44.

Considering periods of 45 weeks.
Number of clients in the first time period: 16588, number of clients in the second time period: 62499.
ARI score:0.9304 for weeks number:45.

Considering periods of 46 weeks.
Number of clients in the first time period: 17543, number of clients in the second time period: 63976.
ARI score:0.9604 for weeks number:46.

Considering periods of 47 weeks.
Number of clients in the first time period: 18465, number of clients in the second time period: 66015.
ARI score:0.9461 for weeks number:47.

Considering periods of 48 weeks.
Number of clients in the first time period: 19440, number of clients in the second time period: 67708.
ARI score:0.9201 for weeks number:48.

Considering periods of 49 weeks.
Number of clients in the first time period: 20381, number of clients in the second time period: 69348.
ARI score:0.9331 for weeks number:49.

Considering periods of 50 weeks.
Number of clients in the first time period: 21406, number of clients in the second time period: 71965.
ARI score:0.9243 for weeks number:50.

Considering periods of 51 weeks.
Number of clients in the first time period: 22272, number of clients in the second time period: 74989.
ARI score:0.947 for weeks number:51.

Considering periods of 52 weeks.
Number of clients in the first time period: 23271, number of clients in the second time period: 75358.
ARI score:0.9585 for weeks number:52.
    """

# TODO: ADD documentation like docstrings for each function
