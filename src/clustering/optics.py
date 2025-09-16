import pandas as pd
from sklearn.cluster import OPTICS

from src.clustering import BaseClustering

class OpticsClustering(BaseClustering):
    """

    Methods:
    --------
    fit(data: pd.DataFrame):
        Fit the Optics clustering algorithm to the data.

    """

    def __init__(self, dataset_name: str = None, embedding_method: str = None, dataset_id: int = None,
                 embedding_id: int = None, embeddings: pd.DataFrame = None):
        """
        Initialize the OpticsClustering class with the minimum cluster size.

        """
        super().__init__(dataset_name=dataset_name, embedding_method=embedding_method, dataset_id=dataset_id,
                         embedding_id=embedding_id, embeddings=embeddings)
        self.clusterer = OPTICS() # min_samples default = 5

    def fit(self):
        """
        :param data: DataFrame containing the data to be clustered.
        :return: DataFrame containing the cluster labels assigned to the data.
        """
        data = self.load_data()
        self.scaled_data = self.scale_data(data)
        self.clusterer.fit(self.scaled_data)
        self.labels = pd.DataFrame(self.clusterer.labels_, columns=['Cluster Label'], index=data.index)
        self.save_labels(self.labels)
        return self.labels







