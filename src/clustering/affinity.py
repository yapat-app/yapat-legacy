import pandas as pd
from sklearn.cluster import AffinityPropagation

from src.clustering import BaseClustering


class Affinity(BaseClustering):
    """
    Methods:
    --------
    fit(data: pd.DataFrame):
        Fit the Affinity Propagation clustering algorithm to the data.

    """

    def __init__(self, dataset_name: str = None, embedding_method: str = None, dataset_id: int = None,
                 embedding_id: int = None, embeddings: pd.DataFrame = None, random_seed: int = 42):
        """
        Initialize the AffinityClustering class with the minimum cluster size.

        :param min_cluster_size: The minimum size of clusters. Clusters smaller than this size will be treated as noise.
        """
        super().__init__(dataset_name=dataset_name, embedding_method=embedding_method, dataset_id=dataset_id,
                         embedding_id=embedding_id, embeddings=embeddings)
        self.clusterer = AffinityPropagation(random_state=random_seed)
