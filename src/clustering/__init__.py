import os
import uuid

import pandas as pd
from sklearn.preprocessing import RobustScaler
from sqlalchemy import and_
from src.app import server
from src.extensions import sqlalchemy_db
from sqlalchemy.exc import SQLAlchemyError
from src.schema_model import Dataset, EmbeddingResult, ClusteringResult, ClusteringMethod, EmbeddingMethod

from src.extensions import sqlalchemy_db
from src.schema_model import Dataset, EmbeddingMethod, EmbeddingResult


class BaseClustering:
    """
    Base class for clustering models used to group data based on embeddings or other features.

    This class provides core functionality for clustering models, including loading data and storing
    clustering results. It is meant to be subclassed by specific clustering algorithms, which should
    implement their own logic for fitting the model and predicting clusters.

    Attributes:
    -----------
    data : pd.DataFrame or None
        DataFrame containing the data to be clustered.
    labels : pd.Series or None
        Series containing the cluster labels assigned to the data.

    Methods:
    --------
    load_data(file_path: str) -> pd.DataFrame:
        Loads a dataset from a CSV or pickle file into a pandas DataFrame.

    save_labels(file_path: str):
        Saves the cluster labels to a CSV or pickle file.

    """

    def __init__(self, dataset_name: str = None, embedding_method: str = None, dataset_id: int = None,
                 embedding_id: int = None, embeddings: pd.DataFrame = None) -> None:
        """
        Initialize the BaseClustering class. Sets up the data and labels attributes.
        """
        self.dataset_name = dataset_name
        self.embedding_method = embedding_method
        self.dataset_id = dataset_id
        self.embedding_id = embedding_id
        self.embeddings = embeddings if embeddings is not None else pd.DataFrame()
        self.data = None  # Will hold the data to be clustered.
        self.labels = None  # Will store the cluster labels after fitting the model.

    def load_data(self) -> pd.DataFrame:
        """
        Load the data to be clustered from a CSV or pickle file.

        :param file_path: Path to the data file (CSV or pickle format).
        :return: DataFrame containing the loaded data.
        """
        self.dataset_id = sqlalchemy_db.session.execute(
            sqlalchemy_db.select(Dataset.id).where(Dataset.dataset_name == self.dataset_name)
        ).scalar_one_or_none()
        self.embedding_id = sqlalchemy_db.session.execute(
            sqlalchemy_db.select(EmbeddingMethod.id).where(EmbeddingMethod.method_name == self.embedding_method)
        ).scalar_one_or_none()
        if self.dataset_id is None or self.embedding_id is None:
            return None

        file_path = sqlalchemy_db.session.execute(
            sqlalchemy_db.select(EmbeddingResult.file_path).where(and_(EmbeddingResult.dataset_id == self.dataset_id,
                                                                       EmbeddingResult.embedding_id == self.embedding_id))
        )

        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path, index_col=0)
        elif file_path.endswith('.pkl'):
            self.data = pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}. Please use CSV or pickle.")
        return self.data

    def scale_data(self, data):
        data.dropna(axis=1, inplace=True)
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(data.values)
        return scaled_data

    def save_labels(self, labels):
        """
        Save the cluster labels to a CSV or pickle file.

        :param file_path: Path where the cluster labels will be saved.
        """
        self.labels = labels
        if self.labels is None:
            raise ValueError("No cluster labels found. Fit the model or predict labels first.")

        unique_filename = f"{uuid.uuid4().hex}.pkl"
        file_path = os.path.join('results/', unique_filename)
        self.labels.to_pickle(file_path)
        # SQL QUERY TO SAVE RESULT

    def fit_predict(self):
        """
        :param self.embeddings: DataFrame containing the data to be clustered.
        :return: DataFrame containing the cluster labels assigned to the data.
        """
        if self.embeddings.empty:
            raise NotImplementedError(
                "As of now, embeddings need to be pre-computed and passed when instantiating the clustering model.")
            # data = self.load_data()
        self.scaled_data = self.scale_data(self.embeddings)
        self.clusterer.fit(self.scaled_data)
        self.labels = pd.DataFrame(self.clusterer.labels_, columns=['Cluster Label'], index=self.embeddings.index)
        # self.save_labels(self.labels)
        return self.labels


def get_clustering_model(method_name: str, *args, **kwargs):
    if method_name == "hdbscan":
        from src.clustering.hdbscan import HDBSCANClustering
        return HDBSCANClustering(*args, **kwargs)
    elif method_name == "dbscan":
        from src.clustering.dbscan import DBSCANClustering
        return DBSCANClustering(*args, **kwargs)
    elif method_name == "affinity":
        from src.clustering.affinity import Affinity
        return Affinity(*args, **kwargs)
    elif method_name == "kmeans":
        from src.clustering.kmeans import KMeansClustering
        return KMeansClustering(*args, **kwargs)
    elif method_name == "spectral":
        from src.clustering.spectral import SPECTRALClustering
        return SPECTRALClustering(*args, **kwargs)
    elif method_name == "optics":
        from src.clustering.optics import OpticsClustering
        return OpticsClustering(*args, **kwargs)
    else:
        raise ValueError(f"Unknown Clustering method: {method_name}")
