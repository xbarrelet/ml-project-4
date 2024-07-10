from kneed import KneeLocator
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    print("Let's start project 4\n")

    # 1 ligne par client avec tous les orders dans un array ou 1 ligne client/order?

    # I'll use a pipeline again. I should use a scaler/standardiser and maybe a PCA(...) to reduce the number of features.

    # https://realpython.com/k-means-clustering-python/

    # La segmentation RFM prend en compte la Récence (date de la dernière commande), la Fréquence des commandes et
    # le Montant (de la dernière commande ou sur une période donnée) pour établir des segments de clients homogènes.

    features, true_labels = make_blobs(n_samples=200, centers=3, cluster_std=2.75, random_state=42)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans_kwargs = {"init": "k-means++", "n_init": 50, "max_iter": 500, "random_state": 42}
    max_range = 11

    sse = []
    silhouette_coefficients = []

    for k in range(1, max_range):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)

        sse.append(kmeans.inertia_)

        if k != 1:
            score = silhouette_score(scaled_features, kmeans.labels_)
            silhouette_coefficients.append(score)

    # There’s a sweet spot where the SSE curve starts to bend known as the elbow point. The x-value of this point is
    # thought to be a reasonable trade-off between error and number of clusters.
    kl = KneeLocator(range(1, max_range), sse, curve="convex", direction="decreasing")
    print(f"elbow found at iteration:{kl.elbow}")

    plt.style.use("fivethirtyeight")
    plt.figure(figsize=(10, 9))

    plt.plot(range(1, max_range), sse)
    plt.xticks(range(1, max_range))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()

    # The silhouette coefficient is a measure of cluster cohesion and separation.
    # It quantifies how well a data point fits into its assigned cluster.
    plt.figure(figsize=(10, 9))
    plt.plot(range(2, 11), silhouette_coefficients)
    plt.xticks(range(2, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()

    # Ultimately, your decision on the number of clusters to use should be guided by a combination of domain knowledge
    # and clustering evaluation metrics.
