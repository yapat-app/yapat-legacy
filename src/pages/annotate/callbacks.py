import base64
import io
import logging
import os

import dash
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from dash import callback, Output, Input, State
from maad.sound import spectrogram
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter

from .sampling import get_sample, process_annotation, replenish_queue

logger = logging.getLogger(__name__)

def get_audio_playback(audio, relayout_data, sample_rate):
    if relayout_data is not None:
        # crop audio in time
        if 'xaxis.range[0]' in relayout_data:
            t_0 = relayout_data['xaxis.range[0]']
            t_1 = relayout_data['xaxis.range[1]']
            t_0_sample = int(t_0 * sample_rate)
            t_1_sample = int(t_1 * sample_rate)
            audio = audio[t_0_sample:t_1_sample]

        # crop audio in frequency
        if 'yaxis.range[0]' in relayout_data:
            f_0 = max(5, relayout_data['yaxis.range[0]'])
            f_1 = max(10, relayout_data['yaxis.range[1]'])
            fs = sample_rate

            # butterworth filter
            [b, a] = butter(4, [f_0, f_1], fs=fs, btype='band')
            audio = lfilter(b, a, audio)

            # adjust datatype
            audio = audio.astype(np.float32)

            # adjust volume
            audio = audio / max(abs(audio))

    # convert audio into bytes
    buffer = io.BytesIO()
    write(buffer, sample_rate, audio)

    # encode audio as b64
    b64 = base64.b64encode(buffer.getvalue())

    # create data URI for the audio
    return "data:audio/x-wav;base64," + b64.decode("ascii")


def _update_spectrogram(current_sample, project_name):
    # Calculate the spectrogram
    path_clip = os.path.join('/app', 'projects', project_name, 'clips', current_sample)
    try:
        audio, sample_rate = librosa.load(path_clip, sr=None)

        sxx, tn, fn, extent = spectrogram(x=audio, fs=sample_rate, mode='amplitude')
        sxx = np.log(sxx + 1e-12)
        sxx_norm = (sxx - np.min(sxx)) / (np.max(sxx) - np.min(sxx))

        fig = px.imshow(sxx_norm, height=600, x=tn, y=fn, aspect='auto', labels=dict(x='Time [s]', y='Frequency [Hz]'),
                        origin='lower')
        fig.update_coloraxes(showscale=False)
        fig.update_layout({
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            'showlegend': False,
            'title': current_sample,
            # 'margin': dict(l=10, r=0, t=30, b=10),  # Adjust margins (left, right, top, bottom)
        })
        fig.update_traces({'showscale': False, 'coloraxis': None})
        relayout_data = {'autosize': True}

    except Exception as e:
        logger.error(f"Could not compute spectrogram for {path_clip}:\n{e}")
        return

    return fig, relayout_data


def register_callbacks():
    @callback(
        Output('project-content', 'data'),
        Output('counter', 'children'),
        Input('button_submit', 'n_clicks'),
        Input('button_skip', 'n_clicks'),
        State('retrieval-method', 'value'),
        State('checklist_annotation', 'options'),
        State('checklist_annotation', 'value'),
        State('project-content', 'data')
    )
    def update_sample(submit, skip, method, annot_options, annot_value, data):
        project_name = data.get('project_name')
        current_sample = data.get('current_sample')
        callback_trigger = dash.ctx.triggered_id

        if project_name:
            status = None

            if dash.ctx.triggered[0]['value']:
                if callback_trigger == 'button_submit':
                    status = 'processed'
                elif callback_trigger == 'button_skip':
                    status = 'skipped'

            if current_sample and status:
                process_annotation(project_name, current_sample, status, annot_value, dash.ctx.triggered)

            n_processed = replenish_queue(n_min=5, project_name=project_name, method=method)

            if status is not None or data.get('current_sample') == '':
                sample = get_sample(project_name)
                data['current_sample'] = sample

        return data, [f"{n_processed}"]

    @callback(
        Output('spectrogram', 'figure'),
        Output('spectrogram', 'relayoutData'),
        Input('project-content', 'data'),
        State('spectrogram', 'figure'),
        State('spectrogram', 'relayoutData'),
    )
    def update_spectrogram(data, fig, relayout_data):
        project_name = data.get('project_name')
        current_sample = data.get('current_sample')
        if project_name and current_sample:
            try:
                fig, relayout_data = _update_spectrogram(current_sample, project_name)
            except Exception as e:
                logger.error(f"Error updating spectrogram: {e}")
        return fig, relayout_data

    @callback(
        Output('audio_complete_file', 'src'),
        Input('project-content', 'data'),
        Input('spectrogram', 'relayoutData'),
        prevent_initial_call=True
    )
    def play_audio(data, relayout_data):
        project_name = data.get('project_name')
        current_sample = data.get('current_sample')
        if project_name and current_sample:
            path_clip = os.path.join('projects', project_name, 'clips', current_sample)
            try:
                assert (os.path.isfile(path_clip))
                audio, sample_rate = librosa.load(path_clip, sr=None)
                audio = get_audio_playback(audio, relayout_data, sample_rate)
            except Exception as e:
                logger.error(f"Could not find clip {path_clip}:\n{e}")
                audio = None
            return audio
        else:
            return None

    @callback(
        Output('checklist_annotation', 'options'),
        Output('checklist_annotation', 'value'),
        Input('project-content', 'data'),
        Input('new-class', 'value'),
        State('checklist_annotation', 'options'),
        State('checklist_annotation', 'value'),
    )
    def update_species_selection(data, new_class, checklist_options, checklist_value):
        callback_trigger = dash.ctx.triggered_id
        project_name = data.get('project_name')
    
        # Ensure project directory and base files exist to avoid FileNotFoundError in deployed environments
        project_dir = os.path.join('/app', 'projects', project_name) if project_name else None
        checklist_options = checklist_options or []
        checklist_value = checklist_value or []
    
        if project_dir:
            os.makedirs(project_dir, exist_ok=True)
            path_vocab = os.path.join(project_dir, 'vocabulary.txt')
            path_annotations = os.path.join(project_dir, 'annotations.csv')
    
            if not os.path.exists(path_vocab):
                # create an empty vocabulary file
                with open(path_vocab, 'w') as vf:
                    vf.write('')
                logger.warning(f"Created missing vocabulary file at {path_vocab}")
    
            if not os.path.exists(path_annotations):
                # create an empty annotations CSV with correct columns
                empty_ann = pd.DataFrame(columns=['sound_clip_url', 'label', 'timestamp'])
                empty_ann.index.name = 'id'
                empty_ann.to_csv(path_annotations)
                logger.warning(f"Created missing annotations file at {path_annotations}")
        else:
            path_vocab = os.path.join('/app', 'projects', '', 'vocabulary.txt')  # fallback, will not be used if project_name is None
    
        if callback_trigger == 'new-class':
            input_text = (new_class or '').strip().capitalize()
            if input_text:
                if input_text not in checklist_options:
                    checklist_options.append(input_text)
                if input_text not in checklist_value:
                    checklist_value.append(input_text)
                # Append new class to vocabulary (file exists because of creation above)
                try:
                    with open(path_vocab, 'a') as f:
                        f.write(input_text + '\n')
                except Exception as e:
                    logger.exception(f"Failed to write new class to {path_vocab}: {e}")
        elif callback_trigger == 'project-content':
            current_sample = data.get('current_sample')
            if current_sample and project_dir:
                try:
                    annotations = pd.read_csv(os.path.join(project_dir, 'annotations.csv'), index_col=0)
                except Exception:
                    annotations = pd.DataFrame(columns=['sound_clip_url', 'label', 'timestamp'])
                try:
                    with open(path_vocab, 'r') as f:
                        lines = f.readlines()
                    checklist_options = [line.strip() for line in lines if line.strip()]
                except Exception:
                    checklist_options = []
                checklist_value = []

        # # exclude options that are not searched for
        # if input_search:
        #     checklist_options = [item for item in checklist_options if input_search.lower() in item.lower()]

        checklist_options.sort()
        checklist_value.sort()

        return checklist_options, checklist_value
