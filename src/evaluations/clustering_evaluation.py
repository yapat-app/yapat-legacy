from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import DBSCAN

from src.evaluations import BaseEvaluation

class ClusteringEvaluation(BaseEvaluation):
    def __init__(self, embedding_method, clustering_method):
        """
                Initialize the ClusteringEvaluation.
        """
        super().__init__(embedding_method, clustering_method)

    def sil_score(self, embeddings, cluster_labels):
        return silhouette_score(embeddings, cluster_labels) # Silhouette Score

    def db_score(self, embeddings, cluster_labels): # Davies Bouldin Score
        return davies_bouldin_score(embeddings, cluster_labels)

    def evaluate(self):
        embeddings, cluster_labels = self.load_data()
        if embeddings is not None and cluster_labels is not None:
            self.scaled_data = self.scale_data(embeddings)
            sil_score = self.sil_score(self.scaled_data, cluster_labels)
            db_score = self.db_score(self.scaled_data, cluster_labels)
            evaluation_results = {
                "Silhouette Score": sil_score,
                "Davies Bouldin Score": db_score
            }
        else:
            evaluation_results = {
                "Silhouette Score": 0.0,
                "Davies Bouldin Score": 0.0
            }
        self.save_results('clusters', evaluation_results)
        return







