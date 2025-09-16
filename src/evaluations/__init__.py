import json
import logging
import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sqlalchemy.exc import SQLAlchemyError

from src.extensions import sqlalchemy_db
from src.schema_model import Dataset, EmbeddingResult, EmbeddingMethod, ClusteringMethod, ClusteringResult
from src.utils.extensions import server

logger = logging.getLogger(__name__)


class BaseEvaluation:
    def __init__(self, embedding_method_name, clustering_method_name):
        self.embedding_method_name = embedding_method_name
        self.clustering_method_name = clustering_method_name
        self.data = None  # Will hold the data to be clustered.
        self.evaluation_result = None

    def load_data(self):

        embedding_file_path, clustering_file_path = self.check_pipeline_completion()
        if embedding_file_path:
            embeddings = pd.read_pickle(embedding_file_path)
        else:
            embeddings = None
        if clustering_file_path:
            cluster_labels = pd.read_pickle(clustering_file_path)
        else:
            cluster_labels = None

        return embeddings, cluster_labels

    def scale_data(self, data):
        data.dropna(axis=1, inplace=True)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data.values)
        return scaled_data

    def extract_hour(self, index):
        filename = os.path.basename(index)
        return filename.split('_')[2][:2].split('.')[0]

    def extract_location(self, index):
        filename = os.path.basename(index)
        return filename.split('_')[0]

    def save_results(self, indicator_evaluation, evaluation_results):
        session = sqlalchemy_db.session
        try:
            with server.app_context():
                selected_dataset = session.query(Dataset).filter_by(is_selected=True).first()
                dataset_id = selected_dataset.id
                embedding_method = session.query(EmbeddingMethod).filter_by(
                    method_name=self.embedding_method_name).first()
                embedding_result = sqlalchemy_db.session.query(EmbeddingResult).filter_by(
                    dataset_id=dataset_id,
                    embedding_id=embedding_method.id
                ).one_or_none()
                if indicator_evaluation == 'embeddings':
                    embedding_result.evaluation_results = json.dumps(evaluation_results)
                elif indicator_evaluation == 'clusters':
                    clustering_method = session.query(ClusteringMethod).filter_by(
                        method_name=self.clustering_method_name).first()
                    clustering_result = sqlalchemy_db.session.query(ClusteringResult).filter_by(
                        method_id=clustering_method.method_id,
                        embedding_id=embedding_result.id
                    ).one_or_none()
                    clustering_result.evaluation_results = json.dumps(evaluation_results)
                session.commit()

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save results: {e}")
        finally:
            session.close()

    def check_pipeline_completion(self):
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

                return embedding_file_path, clustering_file_path

        except SQLAlchemyError as e:
            session.rollback()
            return f"Error querying the database: {e}"
