import pandas as pd
from datetime import datetime
import os
from src.extensions import sqlalchemy_db
from src.utils.extensions import server
from src.schema_model import EmbeddingResult, ClusteringResult, Dataset, EmbeddingMethod, ClusteringMethod, DimReductionMethod, DimReductionResult
from sqlalchemy.exc import SQLAlchemyError


class BaseVisualization:


    def __init__(self, embedding_method_name, clustering_method_name, dim_red_method_name) -> None:
        """
        Initialize the BaseClustering class. Sets up the data and labels attributes.
        """
        # The call back args can be changed if needed
        self.embedding_method_name = embedding_method_name
        self.clustering_method_name = clustering_method_name
        self.dim_red_method_name = dim_red_method_name
        self.embeddings = None
        self.cluster_labels = None

    def load_data(self):
        embedding_file_path, clustering_file_path, dim_red_file_path = self.load_file_paths()
        if embedding_file_path:
            embeddings = pd.read_pickle(embedding_file_path)
        else:
            embeddings = None
        if clustering_file_path:
            cluster_labels = pd.read_pickle(clustering_file_path)
        else:
            cluster_labels = None
        if dim_red_file_path:
            dim_reductions = pd.read_pickle(dim_red_file_path)
        else:
            dim_reductions = None

        return embeddings, cluster_labels, dim_reductions

    def load_file_paths(self):
        session = sqlalchemy_db.session
        try:
            with server.app_context():
                selected_dataset = session.query(Dataset).filter_by(is_selected=True).first()
                dataset_id = selected_dataset.id
                if self.embedding_method_name:
                    embedding_method = session.query(EmbeddingMethod).filter_by(
                        method_name=self.embedding_method_name).first()
                    embedding_result = session.query(EmbeddingResult).filter_by(
                        dataset_id=dataset_id, embedding_id=embedding_method.id, task='completed'
                    ).first()
                    if not embedding_result:
                        embedding_file_path = None
                    else:
                        embedding_file_path = embedding_result.file_path
                else:
                    embedding_file_path = None

                if self.clustering_method_name:
                    clustering_method = session.query(ClusteringMethod).filter_by(
                        method_name=self.clustering_method_name).first()
                    clustering_result = session.query(ClusteringResult).filter_by(
                        embedding_id=embedding_result.id, method_id=clustering_method.method_id, task='completed'
                    ).first()
                    if not clustering_result:
                        clustering_file_path = None
                    else:
                        clustering_file_path = clustering_result.cluster_file_path
                else:
                    clustering_file_path = None

                if self.dim_red_method_name:
                    dim_red_method = session.query(DimReductionMethod).filter_by(
                        method_name=self.dim_red_method_name).first()
                    dim_red_result = session.query(DimReductionResult).filter_by(
                        embedding_id=embedding_result.id, method_id=dim_red_method.method_id, task='completed'
                    ).first()
                    if not dim_red_result:
                        dim_red_file_path = None
                    else:
                        dim_red_file_path = dim_red_result.reduction_file_path
                else:
                    dim_red_file_path = None

        except SQLAlchemyError as e:
            session.rollback()
            return f"Error querying the database: {e}"

        return embedding_file_path, clustering_file_path, dim_red_file_path

    def parse_datetime_from_filename(self, index):
        filename = os.path.basename(index)
        parts = filename.split('_')
        date_str = parts[1]  # Date part
        time_str = parts[2]  # Time part
        return datetime.strptime(date_str, '%Y%m%d'), datetime.strptime(time_str, '%H%M%S').time()









