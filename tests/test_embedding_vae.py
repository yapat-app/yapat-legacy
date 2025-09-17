import os
import unittest
from unittest.mock import patch

import numpy as np
import tensorflow as tf

from src.assets.models.vae_xprize import VAE
from src.embeddings.vae import VAEEmbedding, _compute_spectrogram


class TestVAEEmbedding(unittest.TestCase):

    @patch.object(VAEEmbedding, 'get_path_dataset')
    def test_compute_spectrogram(self, mock_get_path_dataset):
        mock_get_path_dataset.return_value = os.path.join('tests', 'assets', 'test_data')

        instance = VAEEmbedding(
            dataset_name="test_dataset",
            model_path="mock_model_path",
            sampling_rate=22000,
            clip_duration=3.0,
            dask_client=None
        )

        # Load test audio files
        instance.read_audio_dataset()

        # Compute the spectrogram using the class method
        S_mel = _compute_spectrogram(audio=instance.data.iloc[0]['audio_data'], sample_rate=instance.sampling_rate,
                                     resolution=0.05, overlap=0.5, freq_min=1, freq_max=None, n_freqs=400)

        # Assertions
        mock_get_path_dataset.assert_called_once()

        # Check if the first dimension (n_mels) is correct
        expected_n_mels = instance.kw_spectrograms.get('n_freqs')
        expected_S_len = int(1 + instance.clip_duration / (
                instance.kw_spectrograms.get('resolution') * (1 - instance.kw_spectrograms.get('overlap'))))
        self.assertEqual(S_mel.shape[0], expected_n_mels)
        self.assertEqual(S_mel.shape[1], expected_S_len)

        # Check the shape of the resulting spectrogram
        self.assertTrue(np.issubdtype(S_mel.dtype, np.floating))

        # Check if the median of the scaled mel spectrogram is approximately 0
        median_value = np.median(S_mel)
        self.assertAlmostEqual(median_value, 0, places=5)

        # Compute all spectrograms
        instance.process()

    def test_init_with_invalid_clip_duration(self):
        with self.assertRaises(ValueError):
            VAEEmbedding(dataset_name='test_dataset', clip_duration=2.0)

    @patch.object(VAEEmbedding, 'get_path_dataset')
    def test_model(self, mock_get_path_dataset):
        mock_get_path_dataset.return_value = os.path.join('tests', 'assets', 'test_data')
        instance = VAEEmbedding(
            dataset_name="test_dataset",
            model_path="mock_model_path",
            sampling_rate=22000,
            clip_duration=3,
            dask_client=None,
        )
        instance.compute_spectrograms()

        input_shape = (instance.kw_spectrograms.get('n_freqs'), int(1 + instance.clip_duration / (
                instance.kw_spectrograms.get('resolution') * (1 - instance.kw_spectrograms.get('overlap')))), 1)
        model = VAE(input_shape=input_shape, latent_dim=instance.latent_dim, beta_kl=instance.beta_kl)
        model.compile(tf.keras.optimizers.Adam(learning_rate=instance.learning_rate))
        ds_train = tf.data.Dataset.from_tensor_slices(instance.spectrograms)
        ds_train = tf.data.Dataset.zip((ds_train, ds_train))
        ds_train = ds_train.batch(instance.batch_size).prefetch(buffer_size=instance.batch_size)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='reconstruction_loss', patience=3,
                                                          restore_best_weights=True, mode="min")
        history = model.fit(ds_train, epochs=10, verbose=2, callbacks=[early_stopping])

    def test_load_model(self):
        instance = VAEEmbedding(
            dataset_name="test_dataset",
            model_path="mock_model_path",
            sampling_rate=22000,
            clip_duration=3,
            dask_client=None
        )

        instance.load_model()
        self.assertIsNotNone(instance.model)
        self.assertTrue(hasattr(instance.model, 'fit'))
        self.assertTrue(hasattr(instance.model.encoder, 'predict'))


if __name__ == '__main__':
    unittest.main()
