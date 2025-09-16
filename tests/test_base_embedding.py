import os
import unittest
from unittest.mock import patch

import pandas as pd

from src.embeddings import BaseEmbedding
from src.extensions import dask_client


class TestBaseEmbedding(unittest.TestCase):

    def setUp(self):
        """Set up a test instance of BaseEmbedding with mock inputs."""
        self.dataset_name = "test_dataset"
        self.model_path = "mock_model_path"
        self.sample_rate = 22000
        self.clip_duration = 2.5
        self.dask_client = None
        self.base_embedding = BaseEmbedding(
            dataset_name=self.dataset_name,
            model_path=self.model_path,
            clip_duration=self.clip_duration,
            sampling_rate=self.sample_rate,
            dask_client=self.dask_client
        )

    def test_initialization(self):
        """Test that the BaseEmbedding class initializes with correct attributes."""
        self.assertEqual(self.base_embedding.dataset_name, self.dataset_name)
        self.assertEqual(self.base_embedding.clip_duration, self.clip_duration)
        self.assertEqual(self.base_embedding.model_path, self.model_path)
        self.assertEqual(self.base_embedding.sampling_rate, self.sample_rate)
        self.assertEqual(self.base_embedding.dask_client, self.dask_client)
        self.assertIs(self.base_embedding.data.empty, True)
        self.assertIs(self.base_embedding.embeddings.empty, True)

    def test_load_model_not_implemented(self):
        """Test that load_model raises NotImplementedError if not overridden."""
        with self.assertRaises(NotImplementedError):
            self.base_embedding.load_model()

    def test_process_not_implemented(self):
        """Test that process raises NotImplementedError if not overridden."""
        with self.assertRaises(NotImplementedError):
            self.base_embedding.process([])  # Passing dummy list of audio files

    @patch.object(BaseEmbedding, 'get_path_dataset')
    def test_get_path_dataset(self, mock_get_path_dataset):
        # Setup mock
        mock_get_path_dataset.return_value = os.path.join('tests', 'assets', 'test_data')

        # Call the method
        result = self.base_embedding.read_audio_dataset()

        # Assertions
        mock_get_path_dataset.assert_called_once()
        self.assertEqual(len(self.base_embedding.list_of_audio_files), 2, f"Should have 2 audio files, but got {len(self.base_embedding.list_of_audio_files)}")
        self.assertEqual(result.shape, (46, 4), f"Expected shape to be (46, 4), but got {result.shape}")

    @patch.object(BaseEmbedding, 'get_path_dataset')
    def test_read_audio_dataset_local(self, mock_get_path_dataset):
        """Test read_audio_dataset for local processing without Dask."""

        mock_get_path_dataset.return_value = os.path.join('tests', 'assets', 'test_data')
        result = self.base_embedding.read_audio_dataset()

        # Assertions
        mock_get_path_dataset.assert_called_once()
        self.assertEqual(result.shape, (46, 4))  # 46 rows for 4 variables

    @patch.object(BaseEmbedding, 'get_path_dataset')
    def test_read_audio_dataset_with_dask(self, mock_get_path_dataset):
        """Test read_audio_dataset for local processing without Dask."""

        instance = BaseEmbedding(
            dataset_name=self.dataset_name,
            model_path=self.model_path,
            clip_duration=self.clip_duration,
            sampling_rate=self.sample_rate,
            dask_client=dask_client
        )

        mock_get_path_dataset.return_value = os.path.join('tests', 'assets', 'test_data')

        result = instance.read_audio_dataset()

        # Assertions
        mock_get_path_dataset.assert_called_once()
        self.assertEqual(result.shape, (46, 4))  # 46 rows for 4 variables


if __name__ == '__main__':
    unittest.main()
