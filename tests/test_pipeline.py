import os
import tempfile
import unittest
from unittest.mock import patch

from src.clustering import get_clustering_model
from src.embeddings import get_embedding_model
from src.extensions import dask_client


class TestDaskAudioPipeline(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory before each test."""
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        """Clean up the temporary directory and Dask resources after each test."""
        self.test_dir.cleanup()

    def test_compute_embeddings(self):
        list_embeddings = ['acoustic_indices', 'birdnet', 'vae']
        list_clustering = ['affinity', 'dbscan', 'hdbscan', 'kmeans', 'optics', 'spectral']
        future_embeddings = {}
        future_embeddings_for_clustering = {}
        future_clustering = {}

        for embedding_name in list_embeddings:
            with self.subTest(embedding_name=embedding_name):
                embedding_object = get_embedding_model(embedding_name, dataset_name='test_dataset',
                                                       dask_client=dask_client)
                with patch.object(embedding_object, 'get_path_dataset') as mock_get_path_dataset:
                    # Mock return value
                    mock_get_path_dataset.return_value = os.path.join('tests', 'assets', 'test_data')

                    future_embeddings[embedding_name] = dask_client.submit(embedding_object.process)

            for clustering_name in list_clustering:
                with self.subTest(embedding_name=embedding_name, clustering_name=clustering_name):
                    embedding_object = get_embedding_model(embedding_name, dataset_name='test_dataset')
                    clustering_object = get_clustering_model(clustering_name, dataset_name='test_dataset',
                                                             embedding_method=embedding_name)

                    with patch.object(embedding_object, 'get_path_dataset') as mock_get_path_dataset:
                        with patch.object(clustering_object, 'load_data') as mock_load_data:
                            # Mock return value
                            mock_get_path_dataset.return_value = os.path.join('tests', 'assets', 'test_data')
                            mock_load_data.return_value = NotImplemented

                            future_embeddings_for_clustering[embedding_name] = dask_client.submit(
                                embedding_object.process)

                            # TODO Compute real embeddings and somehow save them to self.test_dir
                            # TODO Decide whether saving to disk happens inside embedding_object,
                            #  or inside a wrapper function such as
                            #  src.utils.embeddings.compute_embeddings(dataset_name, embeddings_name)
                            future_clustering[clustering_name] = dask_client.submit(NotImplemented)


if __name__ == '__main__':
    unittest.main()
