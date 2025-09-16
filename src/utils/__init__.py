import glob
import os

from src.utils.extensions import check_socket, server
from src.utils.utils import get_embedding_model


def glob_audio_dataset(path_dataset):
    extensions = ['wav', 'aac', 'm4a', 'flac', 'mp3']
    extensions += [ext.upper() for ext in extensions]
    list_of_audio_files = []
    for extension in extensions:
        audio_files = glob.glob(os.path.join(path_dataset, '**', '*.' + extension), recursive=True)
        list_of_audio_files.extend(audio_files)
    return list_of_audio_files
