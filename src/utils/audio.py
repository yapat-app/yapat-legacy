import os
import glob
import logging
import librosa
import soundfile as sf
import tensorflow as tf
import numpy as np

def get_list_files(audio_path):
    """Get list of audio files from directory and subdirectories."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio path {audio_path} does not exist")
    
    exts = ['wav', 'WAV', 'mp3', 'MP3', 'ogg', 'OGG', 'flac', 'FLAC']
    files = []
    for ext in exts:
        # Search in current directory
        files.extend(glob.glob(os.path.join(audio_path, f'*.{ext}')))
        # Search in subdirectories recursively
        files.extend(glob.glob(os.path.join(audio_path, f'**/*.{ext}'), recursive=True))
    return sorted(files)

def split_single_audio(input_file, output_dir, clip_duration):
    """Split a single audio file into clips."""
    if not os.path.exists(input_file):
        logging.error(f"Input file {input_file} does not exist")
        return

    try:
        # Load audio file
        y, sr = librosa.load(input_file, sr=None)
        
        # Calculate samples per clip
        samples_per_clip = int(clip_duration * sr)
        
        # Calculate number of clips
        num_clips = len(y) // samples_per_clip
        
        # Get filename without extension
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Split into clips and save
        for i in range(num_clips):
            start_sample = i * samples_per_clip
            end_sample = (i + 1) * samples_per_clip
            clip = y[start_sample:end_sample]
            
            # Calculate start and end times in seconds
            start_time = int(i * clip_duration)
            end_time = int((i + 1) * clip_duration)
            
            # Create output filename with start and end times
            output_file = os.path.join(
                output_dir,
                f"{base_name}_{start_time}_{end_time}.wav"
            )
            
            # Save clip using soundfile (replacement for deprecated librosa.output.write_wav)
            sf.write(output_file, clip, sr)
            
    except Exception as e:
        logging.error(f"Error processing {input_file}: {str(e)}")

def load_audio_files_with_tf_dataset(file_list, clip_duration=10.0, sample_rate=44100):
    """Load audio files as a TensorFlow dataset."""
    def load_and_process_audio(file_path):
        audio_binary = tf.io.read_file(file_path)
        waveform, _ = tf.audio.decode_wav(audio_binary, desired_samples=int(clip_duration * sample_rate))
        return waveform

    # Create dataset from file list
    dataset = tf.data.Dataset.from_tensor_slices(file_list)
    
    # Map loading function across dataset
    dataset = dataset.map(load_and_process_audio, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Prefetch for better performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset