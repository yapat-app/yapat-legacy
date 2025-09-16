import pandas as pd
import logging
import os
from typing import Optional
from sklearn.preprocessing import StandardScaler
import dask
import uuid

from sqlalchemy.exc import SQLAlchemyError
from src.app import server
from src.extensions import sqlalchemy_db

from src.schema_model import Dataset, EmbeddingResult, EmbeddingMethod, DimReductionResult, DimReductionMethod

logger = logging.getLogger(__name__)

class BaseDimensionalityReduction:
    """
    Base class for dimensionality reduction models.

    Attributes:
    -----------
    n_components : int
        The number of dimensions to reduce the data to.
    transformed_data : pd.DataFrame or None
        DataFrame containing the reduced dimensionality data.

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

    def __init__(self, n_components: int = 3):
        """
        Initialize the BaseDimensionalityReduction class.

        :param n_components: The number of dimensions to reduce the data to.
        """
        self.n_components = n_components
        self.transformed_data = None  # Placeholder for the transformed data
        self.data = None

    def load_data(self, embedding_method_name):
        session = sqlalchemy_db.session
        try:
            with server.app_context():
                selected_dataset = session.query(Dataset).filter_by(is_selected=True).first()
                if not selected_dataset:
                    logger.error("No dataset is currently selected.")

                embedding_method = session.query(EmbeddingMethod).filter_by(method_name=embedding_method_name).first()
                if not embedding_method:
                    logger.error(f"No embedding method found for '{embedding_method_name}'.")

                embedding_result = session.query(EmbeddingResult).filter_by(dataset_id=selected_dataset.id,
                                                                            embedding_id=embedding_method.id).first()
                if not embedding_result:
                    logger.error(f"No embedding result found for dataset '{selected_dataset.dataset_name}'.")

                # Check if the file path exists
                embedding_file_path = embedding_result.file_path
                if not os.path.exists(embedding_file_path):
                    logger.error(f"Embedding result file does not exist at path: {embedding_file_path}")
                    raise FileNotFoundError(f"Embedding result file does not exist at path: {embedding_file_path}")

                embedding_data_df = pd.read_pickle(embedding_file_path)
                return embedding_data_df

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise Exception(f"Database error: {e}")

        except (ValueError, FileNotFoundError) as e:
            raise e  # Re-raise to handle it outside if needed

        except Exception as e:
            logger.error(f"Error loading embedding data: {e}")
            raise Exception(f"Error loading embedding data: {e}")

    def scale_data(self, data):
        data.dropna(axis=1, inplace=True)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data.values)
        return scaled_data

    def save_transformed_data(self, dimred_method_name: str, embedding_method_name: str, reduced_data: pd.DataFrame):
        session = sqlalchemy_db.session
        try:
            with server.app_context():
                selected_dataset = session.query(Dataset).filter_by(is_selected=True).first()
                if not selected_dataset:
                    logger.error("No dataset is currently selected.")
                    raise ValueError("No dataset is currently selected.")

                embedding_method = session.query(EmbeddingMethod).filter_by(method_name=embedding_method_name).first()
                if not embedding_method:
                    logger.error(f"No embedding method found for '{embedding_method_name}'.")
                    raise ValueError(f"No embedding method found for '{embedding_method_name}'.")

                embedding_result = session.query(EmbeddingResult).filter_by(
                    dataset_id=selected_dataset.id,
                    embedding_id=embedding_method.id
                ).first()
                if not embedding_result:
                    logger.error(
                        f"No embedding result found for dataset '{selected_dataset.dataset_name}' using the '{embedding_method_name}' method.")
                    raise ValueError(
                        f"No embedding result found for dataset '{selected_dataset.dataset_name}' using the '{embedding_method_name}' method."
                    )

                dimred_method = session.query(DimReductionMethod).filter_by(method_name=dimred_method_name).first()
                if not dimred_method:
                    logger.error(f"No dimensionality reduction method found for '{dimred_method_name}'.")
                    raise ValueError(f"No dimensionality reduction method found for '{dimred_method_name}'.")

                unique_filename = f"{uuid.uuid4().hex}_dimred.pkl"
                reduction_file_path = os.path.join('results/', unique_filename)
                reduced_data.to_pickle(reduction_file_path)

                dimred_result = session.query(DimReductionResult).filter_by(
                    embedding_id=embedding_result.id,
                    method_id=dimred_method.method_id
                ).first()

                if not dimred_result:
                    dimred_result = DimReductionResult(
                        embedding_id=embedding_result.id,
                        method_id=dimred_method.method_id,
                        reduction_file_path=reduction_file_path,
                        hyperparameters={},  # Add relevant hyperparameters if any
                        task='completed'  # Mark the task as completed after saving
                    )
                    session.add(dimred_result)
                else:
                    # Update the existing DimReductionResult with the new file path and status
                    dimred_result.reduction_file_path = reduction_file_path
                    dimred_result.task = 'completed'

                # Commit the changes to the database
                session.commit()
                logger.info(f"Dimensionality reduction results saved successfully at {reduction_file_path}")

        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise Exception(f"Database error: {e}")
        except Exception as e:
            logger.error(f"Error saving dimensionality reduction results: {e}")
            raise Exception(f"Error saving dimensionality reduction results: {e}")


def get_dr_model(method_name: str, dask_client: Optional[dask.distributed.client.Client] = None):
    if method_name == "pca":
        from dimensionality_reduction.pca import PCA
        return PCA()
    elif method_name == "tsne":
        from dimensionality_reduction.tsne import TSNE
        return TSNE()
    elif method_name == "umap_reducer":
        from dimensionality_reduction.umap_reducer import UmapReducer
        return UmapReducer()
    else:
        raise ValueError(f"Unknown DR method: {method_name}")
