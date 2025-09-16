import logging
import os
import uuid
from typing import Union

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from src.clustering import get_clustering_model
from src.schema_model import EmbeddingResult, Dataset, EmbeddingMethod
from src.utils import get_embedding_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def update_database_embedding_result(
        dataset_name: str,
        embedding_method: str,
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
            # Fetch task state and other identifiers from the database
            dataset_id = session.execute(
                select(Dataset.id).where(Dataset.dataset_name == dataset_name)
            ).scalar_one_or_none()
            if dataset_id is None:
                raise ValueError(f"Dataset '{dataset_name}' not found")

            embedding_id = session.execute(
                select(EmbeddingMethod.id).where(EmbeddingMethod.method_name == embedding_method)
            ).scalar_one_or_none()
            if embedding_id is None:
                raise ValueError(f"Embedding method '{embedding_method}' not found")

            # Assuming result is a dictionary and maps to your SQLAlchemy model
            model_instance = EmbeddingResult(dataset_id=dataset_id, embedding_id=embedding_id, task_state=task_state,
                                             task_key=task_key, file_path=filepath)
            session.add(model_instance)

            session.commit()  # Commit the transaction
            print(f"Updated database with dataset {dataset_name} and embedding method {embedding_method}")
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(e)


def compute_embeddings(dataset_name, list_embedding_methods):
    logger.info(f"Computing embeddings for dataset '{dataset_name}'")
    for embedding_method in list_embedding_methods:
        sampling_rate = 22000 if embedding_method == 'acoustic_indices' else None
        logger.info(f"Computing embeddings for method '{embedding_method}'")
        embedder_instance = get_embedding_model(method_name=embedding_method, dataset_name=dataset_name,
                                                sampling_rate=sampling_rate)
        embedder_instance.process()
        filepath = os.path.join('instance', 'embeddings', f'{dataset_name}_{embedding_method}_{uuid.uuid4()}.pkl')
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        embedder_instance.embeddings.to_pickle(filepath)
        update_database_embedding_result(dataset_name=dataset_name, embedding_method=embedding_method,
                                         filepath=filepath, task_key=None, task_state='completed')
    logger.info(f"Finished computing embeddings for dataset '{dataset_name}'")
    return