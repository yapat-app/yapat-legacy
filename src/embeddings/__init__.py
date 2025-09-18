import logging
import os
import pathlib
from typing import Optional, Union

import librosa
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import select
from sqlalchemy.exc import NoResultFound, SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from src.schema_model import Dataset, EmbeddingMethod, EmbeddingResult
from src.utils import glob_audio_dataset
from src.extensions import sqlalchemy_db
from src.utils.extensions import server

logger = logging.getLogger(__name__)


def _split_audio_into_chunks(filename: str, chunk_duration: float, sampling_rate: Optional[int] = None) -> pd.DataFrame:
    """
    Splits an audio file into non-overlapping chunks of specified duration.

    Args:
        filename (str): Path to the audio file.
        chunk_duration (float): Duration of each chunk in seconds.
        sampling_rate (int, optional): Sampling rate to load the audio file. If None, the default sampling rate
            is used by librosa.

    Returns:
        pd.DataFrame: A DataFrame with two columns:
            - 'filename': The chunked filename with the format 'original_filename_starttime_endtime.ext'.
            - 'audio_data': The corresponding chunked audio data.
    """
    # Load the audio file with librosa
    audio, sampling_rate = librosa.load(filename, sr=sampling_rate)

    # Ensure the length is a multiple of the chunk size by truncating the extra samples
    chunk_size = sampling_rate * chunk_duration

    # Clip audio at a multiple of chunk_size
    audio = audio[:int(len(audio) // chunk_size * chunk_size)]

    # Split the audio into non-overlapping chunks
    chunked_audio = np.split(audio, len(audio) // chunk_size)

    # Generate filenames for each chunk
    prefix, _ = filename.rsplit('.', 1)
    prefix = os.path.basename(prefix)
    t_start = [i * chunk_duration for i in range(len(chunked_audio))]
    t_end = [(i + 1) * chunk_duration for i in range(len(chunked_audio))]

    # Create a DataFrame with chunked filenames and corresponding audio data
    df = pd.DataFrame(
        data={
            "audio_data": chunked_audio,
            "filename": filename,
            "t_start": t_start,
            "t_end": t_end
        },
        index=[f"{prefix}_{i}_{j}" for i, j in zip(t_start, t_end)]
    )
    return df


class BaseEmbedding:
    """
        Base class for embedding models used to generate embeddings from audio data.

        This class provides core functionality for embedding models, including loading the model,
        reading and processing audio datasets, and optionally using Dask for distributed processing.
        It is meant to be subclassed by specific embedding models, which should implement their own
        model loading and processing logic.

        Attributes:
        -----------
        dataset_name : pd.DataFrame or None
            DataFrame holding the processed audio data (e.g., file paths, audio features).
        embeddings : pd.DataFrame or None
            DataFrame containing the generated embeddings for the audio dataset.

        Methods:
        --------
        load_model():
            Abstract method for loading the model. Must be implemented by subclasses.
            :raises NotImplementedError: This method must be implemented by subclasses for model loading.


        process(audio_files):
            Abstract method for processing audio files to generate embeddings. Must be implemented by subclasses.
            :raises NotImplementedError: This method must be implemented by subclasses for model loading.


        read_audio_dataset() -> pd.DataFrame:
            Reads and processes an audio dataset, optionally using Dask for parallel processing.
            Returns a pandas DataFrame containing the audio file paths and other metadata.
            :return: A pandas DataFrame indexed by 'filename' and a data column 'audio_data' containing processed audio chunks.
        """

    def __init__(
            self,
            dataset_name: str,
            clip_duration: float = 3.0,
            model_path: Optional[Union[str, pathlib.Path]] = None,
            sampling_rate: Optional[int] = None,
    ) -> None:
        """
        Initialize the BaseEmbedding class with the model path and an optional Dask client.

        :param dataset_name: Name of dataset as stored in Dataset.dataset_name from the database.
        :param clip_duration: Audio files are chunked in clips of duration specified in seconds.
        :param model_path: Path of pre-saved model (if any).
        :param sampling_rate: Sample rate used by `librosa.load`. If None, native sampling rate will be used.
        """

        # Input args
        self.dataset_name = dataset_name
        self.model_path = model_path
        self.sampling_rate = sampling_rate
        self.clip_duration = clip_duration

        # Placeholders
        self.data = pd.DataFrame()  # Initialize as empty DataFrame
        self.embeddings = pd.DataFrame()  # Initialize as empty DataFrame
        self.list_of_audio_files = []
        self.path_dataset = None

    def load_model(self):
        """
        Placeholder method for loading the model. This should be implemented by subclasses if needed.
        """
        if self.model_path:
            raise NotImplementedError("Subclasses should implement this method if a model path is provided.")
        else:
            pass

    def process(self):
        """
        Placeholder method for processing audio files. This should be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    def get_path_dataset(self, url_db: Optional[str] = None):

        # Create a new DB session for the task
        url_db = url_db or 'sqlite:///instance/pipeline_data.db'
        engine = create_engine(url_db)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Fetch the path to the dataset from the database within this session
        try:
            path_dataset = session.execute(
                select(Dataset.path_audio).where(Dataset.dataset_name == self.dataset_name)
            ).scalar_one()
        except NoResultFound:
            raise ValueError(f"No dataset found for name: {self.dataset_name}")
        finally:
            # Close the session after the task is done
            session.close()

        return path_dataset

    def read_audio_dataset(self) -> pd.DataFrame:
        """
        Read the dataset of audio files and process it using Dask for parallelization if available.

                :return: A pandas DataFrame containing audio file paths and any other relevant metadata.
                """

        self.path_dataset = self.get_path_dataset()
        self.list_of_audio_files = glob_audio_dataset(path_dataset=self.path_dataset)

        if not self.list_of_audio_files:
            raise ValueError("No audio files found in the dataset.")

        dfs_audio = [_split_audio_into_chunks(path_audio_file, self.clip_duration, self.sampling_rate) for
                     path_audio_file in self.list_of_audio_files]
        self.data = pd.concat(dfs_audio)

        # Return the concatenated DataFrame of processed audio files.
        return self.data

    def save_embeddings(self, embedding_method_name: str, embeddings):
        if embeddings is None:
            logger.warning("No embeddings available to save")
            return
        try:
            with server.app_context():
                # It should be handled if the dataset is changed while the emebedding are calculated
                selected_dataset = sqlalchemy_db.session.query(Dataset).filter_by(is_selected=True).first()
                if not selected_dataset:
                    logger.warning("No dataset is currently selected. Cannot save embeddings.")
                    return
                embedding_method = sqlalchemy_db.session.query(EmbeddingMethod).filter_by(
                    method_name=embedding_method_name).first()
                if not embedding_method:
                    logger.error(f"Embedding method '{embedding_method_name}' not found in the database.")
                    return

                existing_entry = sqlalchemy_db.session.query(EmbeddingResult).filter_by(
                    dataset_id=selected_dataset.id,
                    embedding_id=embedding_method.id
                ).first()

                if existing_entry and os.path.exists(existing_entry.file_path):
                    logger.warning(
                        f"Embeddings for dataset ID {selected_dataset.id} and embedding method ID {embedding_method.id} already exist. Skipping save.")
                    return
                elif existing_entry and not os.path.exists(existing_entry.file_path):
                    logger.warning(
                        f"Database entry exists but file is missing. Recreating embeddings file.")
                    # Check if there are clustering results that reference this embedding result
                    from src.schema_model import ClusteringResult, DimReductionResult
                    clustering_results = sqlalchemy_db.session.query(ClusteringResult).filter_by(
                        embedding_result_id=existing_entry.id
                    ).all()
                    
                    if clustering_results:
                        logger.warning(f"Found {len(clustering_results)} clustering results that reference this embedding. Deleting them first.")
                        for clustering_result in clustering_results:
                            # Also check for dimensionality reduction results that reference this clustering result
                            dim_reduction_results = sqlalchemy_db.session.query(DimReductionResult).filter_by(
                                clustering_result_id=clustering_result.id
                            ).all()
                            
                            if dim_reduction_results:
                                logger.warning(f"Found {len(dim_reduction_results)} dimensionality reduction results that reference clustering result {clustering_result.id}. Deleting them first.")
                                for dim_reduction_result in dim_reduction_results:
                                    sqlalchemy_db.session.delete(dim_reduction_result)
                            
                            sqlalchemy_db.session.delete(clustering_result)
                    
                    # Delete the database entry so we can recreate it
                    sqlalchemy_db.session.delete(existing_entry)
                    sqlalchemy_db.session.commit()

                os.makedirs('results', exist_ok=True)
                embedding_file_path = os.path.join('results',
                                                   f"{selected_dataset.dataset_name}_{embedding_method_name}_embeddings.pkl")
                embeddings.to_pickle(embedding_file_path)
                self._save_embedding_metadata_to_db(
                    dataset_id=selected_dataset.id,
                    embedding_id=embedding_method.id,
                    file_path=embedding_file_path
                )
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            return

    def _save_embedding_metadata_to_db(self, dataset_id: int, embedding_id: int, file_path: str) -> None:

        try:
            # Add metadata to the EmbeddingResult table
            from src.schema_model import EmbeddingResult
            embedding_result = EmbeddingResult(
                dataset_id=dataset_id,
                embedding_id=embedding_id,
                file_path=file_path,
                hyperparameters={},  # Optionally add hyperparameters here
                evaluation_results={},  # Optionally add evaluation results here
                created_at=pd.Timestamp.now(),
                task_state='completed'
            )
            from src.extensions import sqlalchemy_db
            sqlalchemy_db.session.add(embedding_result)
            sqlalchemy_db.session.commit()
            logger.info(f"Embedding metadata saved to the database for dataset {dataset_id}")

        except SQLAlchemyError as e:
            sqlalchemy_db.session.rollback()
            logger.error(f"Error saving embedding metadata to the database: {e}")
