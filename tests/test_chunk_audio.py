import os
import unittest

import librosa
import numpy as np
import pandas as pd

from src.embeddings import _split_audio_into_chunks


class TestSplitAudioChunks(unittest.TestCase):

    def setUp(self):
        """Set up a real example .wav file."""
        # Assuming test.wav is available in the 'tests/audio' directory.
        self.audio_file = os.path.join('tests', 'assets', 'test_data', 'INCT41_20210103_181500.wav')
        self.chunk_duration = 3.0  # Duration in seconds
        self.sampling_rate = 22050  # Example sampling rate

    def test_split_audio_into_chunks(self):
        """Test splitting a real audio file into chunks."""
        # Ensure the audio file exists
        self.assertTrue(os.path.exists(self.audio_file), "Test audio file not found")

        # Call the function to split the audio into chunks
        df_chunks = _split_audio_into_chunks(self.audio_file, self.chunk_duration, self.sampling_rate)

        # Ensure the returned object is a DataFrame
        self.assertIsInstance(df_chunks, pd.DataFrame, "Expected result to be a pandas DataFrame")

        # Check if the DataFrame has the correct structure (filename and audio_data)
        self.assertIn('audio_data', df_chunks.columns, "Missing 'audio_data' column in DataFrame")

        # Check the number of chunks
        audio, sr = librosa.load(self.audio_file, sr=self.sampling_rate)
        total_duration = len(audio) / sr
        expected_chunks = int(np.floor(total_duration / self.chunk_duration))
        self.assertEqual(len(df_chunks), expected_chunks, "Number of chunks does not match expected count")
        self.assertEqual(len(df_chunks), 19, "Number of chunks does not match expected count")

        # # Check the format of the chunk filenames
        # for chunk_filename in df_chunks.index:
        #     self.assertRegex(chunk_filename, r'tests/assets/test_data/INCT41_20210103_181500_\d+\_\d+\.\w{3}',
        #                      "Chunk filename format is incorrect")


if __name__ == '__main__':
    unittest.main()
