import pathlib
from typing import Optional, Union

import maad
import numpy as np
import pandas as pd
from dask.distributed import Client
from maad import sound, features

from src.embeddings import BaseEmbedding


def compute_spectral_features(audio_file: np.array, sampling_rate: int, **kwargs) -> pd.DataFrame:
    """
    Compute all spectral features for a single audio file using the `maad.features.all_spectral_features` method.

    :param audio_file: Path to the audio file to process.
    :param sampling_rate: Sampling rate of the audio file.
    :param kwargs: Additional parameters for the feature computation, such as 'nperseg', 'roi', and 'method'.
    :return: A pandas DataFrame containing all spectral features for the audio file.
    """

    # Compute all spectral features using the maad library's all_spectral_features function.
    spectral_features = features.all_spectral_features(audio_file, sampling_rate, **kwargs)

    # Convert the features into a pandas DataFrame.
    features_df = pd.DataFrame(spectral_features).T

    return features_df


class AcousticIndices(BaseEmbedding):
    """
    AcousticIndices class for computing spectral features from audio data.

    This class extends BaseEmbedding and provides functionality for computing 
    various spectral features using non-machine-learning methods. It computes features
    such as peak frequency, bandwidth, skewness, etc., using the method described.

    Attributes:
    -----------
    model_path : str
        Not used in this class, but inherited from BaseEmbedding for consistency.
    dask_client : dask.distributed.client.Client or None
        Optional Dask client for handling distributed task execution.
    data : pd.DataFrame or None
        DataFrame holding the processed audio data (e.g., file paths, audio features).
    embeddings : pd.DataFrame or None
        DataFrame containing the computed acoustic features for the audio dataset.

    Methods:
    --------
    load_model():
        This method is not applicable for this class, as no model is required.

    process(dataset_name: str, extension: str = '.wav', sampling_rate: int = 48000, **kwargs):
        Processes the audio dataset to compute spectral features.

    compute_spectral_features(audio_file: str, sampling_rate: int = 48000, **kwargs) -> pd.DataFrame:
        Computes all spectral features for a single audio file using the `maad.features.all_spectral_features` method.
    """

    def __init__(
            self,
            dataset_name: str,
            clip_duration: float = 3.0,
            model_path: Optional[Union[str, pathlib.Path]] = None,
            sampling_rate: Optional[int] = None,
    ):
        super().__init__(dataset_name, clip_duration, model_path, sampling_rate)

    def load_model(self):
        """
        AcousticIndices does not require a machine learning model, so this method is not used.
        """
        pass  # No model loading required for this class.

    def process(self):
        """
        Processes the dataset by reading the audio files and computing the acoustic indices.

        :param dataset_name: Name of the dataset to process.
        :param sampling_rate: Sampling rate for the audio files.
        :param kwargs: Additional keyword arguments for feature computation (e.g., 'nperseg', 'roi', 'method').
        :return: A pandas DataFrame containing computed acoustic features for each audio file.
        """
        if self.sampling_rate is None:
            raise ValueError("Sampling rate must be specified.")
        self.read_audio_dataset()
        all_features = []
        for row in self.data.iterrows():
            temporal_indices = maad.features.all_temporal_alpha_indices(s=row[1].audio_data, fs=self.sampling_rate)
            Sxx_power, tn, fn, _ = maad.sound.spectrogram(x=row[1].audio_data, fs=self.sampling_rate)
            spectral_indices, per_bin_indices = maad.features.all_spectral_alpha_indices(Sxx_power, tn, fn)
            all_indices = temporal_indices.join(spectral_indices)
            all_features.append(all_indices)

        self.embeddings = pd.concat(all_features, axis=0).set_index(self.data.index)
        return self.embeddings
