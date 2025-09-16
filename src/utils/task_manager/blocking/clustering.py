import logging
import os
import uuid
from typing import Union

import pandas as pd
from sqlalchemy import create_engine, select, and_
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from src.clustering import get_clustering_model
from src.schema_model import EmbeddingResult, Dataset, ClusteringResult, ClusteringMethod

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def update_database_clustering_result(
        embedding_result_id: int,
        clustering_method: str,
        filepath: Union[str, os.PathLike],
        task_state: str,
        task_key: str = None,
):
    """Callback function to update the database upon task completion."""
    # Connect to the database
    url_db = 'sqlite:///instance/pipeline_data.db'  # Adjust as necessary
    engine = create_engine(url_db)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        try:
            method_id = session.execute(
                select(ClusteringMethod.id).where(ClusteringMethod.method_name == clustering_method)
            ).scalar_one_or_none()
            if method_id is None:
                raise ValueError(f"Clustering method '{clustering_method}' not found")

            model_instance = ClusteringResult(embedding_result_id=embedding_result_id, method_id=method_id,
                                              task_state=task_state, task_key=task_key, file_path=filepath)
            session.add(model_instance)

            session.commit()  # Commit the transaction
            print(f"Updated database with clustering method: '{clustering_method}' and embedding results id: {embedding_result_id}")
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(e)


def _get_embedding_results(dataset_name):
    url_db = 'sqlite:///instance/pipeline_data.db'  # Adjust as necessary
    engine = create_engine(url_db)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        dataset_id = session.execute(
            select(Dataset.id).where(Dataset.dataset_name == dataset_name)
        ).scalar_one_or_none()
        if dataset_id is None:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        # Query the database for embedding results
        df = pd.read_sql_table('embedding_results', con=url_db, index_col='id')

        # Convert the results to a Pandas DataFrame
        return df


def _get_pickled_embeddings(dataset_id, embedding_id):
    url_db = 'sqlite:///instance/pipeline_data.db'  # Adjust as necessary
    engine = create_engine(url_db)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        path_name = session.execute(
            select(EmbeddingResult.file_path).where(and_(EmbeddingResult.dataset_id == dataset_id,
                                                         EmbeddingResult.embedding_id == embedding_id))
        ).scalar_one_or_none()
        if path_name is None:
            raise ValueError(f"Pickled embeddings not found for embedding id: {embedding_id}, dataset id: {dataset_id}")
    pickled_embedding = pd.read_pickle(path_name)
    return pickled_embedding


def compute_clusters(dataset_name, list_clustering_methods):
    print(f"Computing clusters for dataset: '{dataset_name}'")
    df_embedding_results = _get_embedding_results(dataset_name=dataset_name)
    for id, embedding_result in df_embedding_results.iterrows():
        print(f"Computing clusters for embedding method id: {embedding_result.embedding_id}")
        embeddings = _get_pickled_embeddings(dataset_id=embedding_result.dataset_id,
                                             embedding_id=embedding_result.embedding_id)
        for clustering_method in list_clustering_methods:
            print(f"Computing clusters for clustering method: '{clustering_method}'")
            cluster_instance = get_clustering_model(clustering_method, embeddings=embeddings)
            cluster_instance.fit_predict()
        #     # raise NotImplementedError

            filepath = os.path.join('instance', 'clusters', f'{id}_{clustering_method}_{uuid.uuid4()}.pkl')
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            cluster_instance.labels.to_pickle(filepath)
            update_database_clustering_result(embedding_result_id=id, clustering_method=clustering_method,
                                              filepath=filepath,
                                              task_key=None, task_state='completed')
    print(f"Finished computing clusters for dataset {dataset_name}")
    # return
