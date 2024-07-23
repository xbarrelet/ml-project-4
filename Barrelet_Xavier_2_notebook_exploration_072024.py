import os
import shutil
import sqlite3
from datetime import datetime

import dataframe_image as dfi
import numpy as np
import pandas as pd
import seaborn as sns
import squarify
from matplotlib import pyplot as plt
from pandas.core.frame import DataFrame
from sklearn.preprocessing import StandardScaler

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# CONFIG
plt.style.use("fivethirtyeight")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def remove_last_run_plots():
    shutil.rmtree('analysis_plots', ignore_errors=True)
    os.mkdir('analysis_plots')


def save_plot(plot, filename: str, prefix: str) -> None:
    os.makedirs(f"analysis_plots/{prefix}", exist_ok=True)

    # fig = plot.get_figure()
    if hasattr(plot, 'get_figure'):
        fig = plot.get_figure()
    elif hasattr(plot, '_figure'):
        fig = plot._figure
    else:
        fig = plot

    fig.savefig(f"analysis_plots/{prefix}/{filename}.png")
    plt.close()


def load_data():
    con = sqlite3.connect("resources/olist.db")
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    res = cur.execute("SELECT customer_id FROM customers where customer_id in (select customer_id from orders)")
    customers = res.fetchall()

    res = cur.execute("select order_id, review_score from order_reviews")
    reviews = res.fetchall()

    res = cur.execute(
        """SELECT o.order_id, o.customer_id, o.order_purchase_timestamp, oi.price
    FROM orders o
    inner join order_items oi on o.order_id = oi.order_id""")
    orders = res.fetchall()

    cur.close()
    con.close()

    sorted_reviews = {}
    for review in reviews:
        sorted_reviews.setdefault(
            review['order_id'],
            []).append(
            review['review_score'])

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
        # nb_products = len(set([order['order_id'] for order in customer_orders]))

        order_timestamps = [datetime.strptime(order['order_purchase_timestamp'], DATE_FORMAT)
                            for order in customer_orders]
        latest_purchase_date: datetime = max(order_timestamps)
        days_since_last_purchase = (datetime.now() - latest_purchase_date).days

        review_scores = [order['review_score']
                         for order in customer_orders if order['review_score'] is not None]
        if len(review_scores) > 0:
            average_review = sum(review_scores) / len(review_scores)
        else:
            average_review = 0

        clients.append({
            'average_review': average_review,
            'recency': days_since_last_purchase,
            'frequency': nb_products,
            'monetary_value': total_amount
        })

    return DataFrame(clients)


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
    RFM_stats.sort_values("MonetaryValue_Count", ascending=False, inplace=True)

    dfi.export(
        RFM_stats,
        'analysis_plots/RFM/RFM_stats.png',
        table_conversion='matplotlib',
        fontsize=9)
    return RFM_stats


def save_rfm_segments(RFM_stats):
    fig = plt.gcf()
    fig.set_size_inches(16, 9)

    plot = squarify.plot(
        sizes=RFM_stats['MonetaryValue_Count'],
        label=[
            'Champions',
            'Loyaux',
            'Loyalistes potentiels',
            'À réactiver',
            'À risque',
            'Perdus'
        ],
        color=[
            "green",
            "orange",
            "purple",
            "maroon",
            "pink",
            "teal"],
        alpha=0.6)
    plot.set_title("RFM Segments")
    plot.set_axis_off()

    plot.get_figure().savefig(f"analysis_plots/RFM/RFM_segments.png")
    plt.close()


def create_pieplot_for_RFM_segments(df, prefix):
    unique_values = df["RFM_Level"].unique()
    data = []
    labels = []

    for value in unique_values:
        values_count = df["RFM_Level"].value_counts()[value]
        data.append(values_count)
        labels.append(value)

    plt.figure(figsize=(10, 8))
    colors = sns.color_palette('pastel')[0:6]
    plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%')
    plt.title('RFM segments distribution')
    plt.savefig(f"analysis_plots/visualization_{prefix}/RFM_segments_pieplot.png")
    plt.close()


def visualize_data(df, prefix):
    create_visualization_plot_for_attribute(df, "Recency", prefix)
    create_visualization_plot_for_attribute(df, "Frequency", prefix)
    create_visualization_plot_for_attribute(df, "Monetary_Value", prefix)
    create_visualization_plot_for_attribute(df, "average_review", prefix)
    create_visualization_plot_for_attribute(df, "RFM_Score", prefix)

    if "RFM_Level" in df.columns:
        create_pieplot_for_RFM_segments(df, prefix)


def create_visualization_plot_for_attribute(df, attribute: str, prefix):
    column_name = attribute.lower().replace(" ", "_")

    if column_name != "rfm_score":
        plt.figure(figsize=(8, 8))
        plot = sns.displot(df[column_name])

        plot.set_xlabels(attribute.replace("average_review", "Average Review"))
        plot.set_ylabels("Count")

        save_plot(plot, f"{column_name}_distplot", f"visualization_{prefix}")


def add_rfm_columns(df):
    Rlabel = range(4, 0, -1)
    Mlabel = range(1, 5)

    df['R'] = pd.qcut(df['recency'], q=4, labels=Rlabel).values
    df['M'] = pd.qcut(df['monetary_value'], q=4, labels=Mlabel).values
    df['F'] = np.where(df['frequency'] == 1, 1, 2)  # Good enough

    df['RFM_Score'] = df[['R', 'F', 'M']].sum(axis=1)
    df['RFM_Level'] = df.apply(rfm_level, axis=1)

    return df


def rfm_level(df):
    if (df['RFM_Score'] >= 7) and (df['RFM_Score'] < 9):
        return 'Champions'
    elif (df['RFM_Score'] >= 6) and (df['RFM_Score'] < 7):
        return 'Loyaux'
    elif (df['RFM_Score'] >= 5) and (df['RFM_Score'] < 6):
        return 'Loyalistes potentiels'
    elif (df['RFM_Score'] >= 4) and (df['RFM_Score'] < 5):
        return 'À réactiver'
    elif (df['RFM_Score'] >= 3) and (df['RFM_Score'] < 4):
        return 'À risque'
    else:
        return 'Perdus'


def visualize_rfm_segments(df):
    os.makedirs("analysis_plots/RFM", exist_ok=True)

    RFM_stats = save_rfm_stats(df)
    save_rfm_segments(RFM_stats)


if __name__ == '__main__':
    print("Starting the exploration script.\n")

    remove_last_run_plots()

    df: DataFrame = load_data()
    print("Data loaded.\n")

    customers_with_more_than_one_product = df[df['frequency'] > 1]
    print(f"Customers with more than one product:{len(customers_with_more_than_one_product)} on {len(df)}, "
          f"or {round(len(customers_with_more_than_one_product) * 100 / len(df), 2)}%\n")

    df = add_rfm_columns(df)
    visualize_rfm_segments(df)
    print("RFM segmentation finished.\n")

    visualize_data(df, "pre_scaling")

    df.drop(columns=["RFM_Level"], axis=1, inplace=True)
    scaled_df = DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
    visualize_data(scaled_df, "after_scaling")

    print("Visualization done.")
