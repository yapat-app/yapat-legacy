import os
import pathlib
from typing import Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from src.embeddings import BaseEmbedding


# BirdNet embedding class that inherits from BaseEmbedding and provides specific implementation for BirdNet.
class BirdnetEmbedding(BaseEmbedding):
    """
    BirdNet embedding model class for generating embeddings from audio data using the BirdNET model.

    This class extends the BaseEmbedding class and provides the specific implementation for loading
    the BirdNet model and processing audio datasets. BirdNet is used for analyzing and extracting
    features from audio clips, often for tasks such as species identification from bird calls.

    Inherited Attributes:
    ---------------------
    model_path : str
        Path to the pre-trained model used for generating embeddings.
    dask_client : dask.distributed.client.Client or None
        Optional Dask client for handling distributed task execution.

    Methods:
    --------
    load_model():
        Loads the BirdNet-specific model using TensorFlow.

    process(dataset_name: str, extension: str = '.wav', sampling_rate: int = 48000):
        Processes the audio dataset to generate embeddings using the BirdNet model.
    """

    def __init__(
            self,
            dataset_name: str,
            clip_duration: float = 3.0,
            model_path: Optional[Union[str, pathlib.Path]] = os.path.join('assets', 'models', 'birdnet', 'V2.4',
                                                                          'BirdNET_GLOBAL_6K_V2.4_Model'),
            sampling_rate: Optional[int] = None,
    ):
        sampling_rate = sampling_rate or 48000
        super().__init__(dataset_name, clip_duration, model_path, sampling_rate)

    def load_model(self):
        """
        Load the BirdNet-specific model, using a TensorFlow SMLayer for generating embeddings.
        """

        input_layer = tf.keras.layers.Input(shape=(144000,), dtype='float32',
                                            name='input_layer')
        tfsm_layer = tf.keras.layers.TFSMLayer(self.model_path, call_endpoint='embeddings')(input_layer)
        self.model = tf.keras.Model(inputs=input_layer, outputs=tfsm_layer)

    def process(self):
        """
        Process the dataset using the BirdNet model to generate embeddings.

        :param dataset_name: Name of the dataset to process.
        :param sampling_rate: The sampling rate for the audio files (default is 48,000).
        :return: A pandas DataFrame containing the generated embeddings.
        """
        self.data = self.read_audio_dataset()
        self.load_model()
        results = []
        if self.sampling_rate != 48000:
            raise ValueError(f"Birdnet expects 48kHz sampling rate; got {self.sampling_rate}")
        for row in self.data.iterrows():
            audio_data = np.expand_dims(row[1].audio_data, axis=0)
            embedding = self.model(audio_data)
            embedding_array = embedding['embeddings'].numpy().flatten()
            results.append(embedding_array.tolist())

        self.embeddings = pd.DataFrame(results, index=self.data.index, columns=[f'embedding_{i}' for i in range(1024)])
        return self.embeddings
