import pandas as pd
from sklearn.cluster import DBSCAN

from src.clustering import BaseClustering


class DBSCANClustering(BaseClustering):
    """

    Methods:
    --------
    fit(data: pd.DataFrame):
        Fit the DBSCAN clustering algorithm to the data.

    """

    def __init__(self, dataset_name: str = None, embedding_method: str = None, dataset_id: int = None,
                 embedding_id: int = None, embeddings: pd.DataFrame = None, eps: float = 0.001):
        """
        Initialize the DBSCANClustering class with the minimum cluster size.

        :param min_cluster_size: The minimum size of clusters. Clusters smaller than this size will be treated as noise.
        """
        super().__init__(dataset_name=dataset_name, embedding_method=embedding_method, dataset_id=dataset_id,
                         embedding_id=embedding_id, embeddings=embeddings)
        self.clusterer = DBSCAN(eps=eps)


