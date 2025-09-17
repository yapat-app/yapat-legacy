import glob
import logging
import os
from datetime import datetime
from multiprocessing import Pool

import dash
import numpy as np
import pandas as pd
import tensorflow as tf
from dash import html, callback, Input, Output, State

from src.pages.explore.callbacks import list_existing_datasets
from src.pages.home import register_dataset
from src.schema_model import Dataset
from sqlalchemy.exc import SQLAlchemyError

from src.extensions import sqlalchemy_db
from src.utils import glob_audio_dataset, server
from src.utils.audio import get_list_files, split_single_audio, load_audio_files_with_tf_dataset

logger = logging.getLogger(__name__)


@callback(
    Input('dataset-list', 'value'),  # This is the RadioItems for selecting a dataset
    prevent_initial_call=True
)
def update_selected_dataset(selected_dataset_name):
    # If no dataset is selected, do nothing
    if not selected_dataset_name:
        return
    session = sqlalchemy_db.session
    try:
        session.query(Dataset).update({Dataset.is_selected: False})
        # Set is_selected to True for the selected dataset
        selected_dataset = session.query(Dataset).filter_by(dataset_name=selected_dataset_name).first()
        if selected_dataset:
            selected_dataset.is_selected = True
            session.commit()  # Commit changes to the database
            # Return dataset summary for UI display
            # return html.Div([
            #     html.P(f"Dataset: {selected_dataset.dataset_name}"),
            #     html.P(f"Created At: {selected_dataset.created_at}"),
            #     html.P(f"Audio Path: {selected_dataset.path_audio}")
            # ])
            return

    except SQLAlchemyError as e:
        # Roll back transaction in case of any errors
        session.rollback()
        logger.error(f"Error updating dataset selection: {e}")
        return



def scan_projects():
    """
    Scan the existing projects
    """
    options = ([{'label': x, 'value': x} for x in os.listdir('projects') if
                os.path.isdir(os.path.join('projects', x))])
    return options


def init_project(project_name, audio_path, clip_duration, embedding_model):
    """
    Initializes the project by generating clips, tables, queue, and embeddings.
    """
    try:
        logging.info(f'Initializing project {project_name}')
        os.makedirs(os.path.join('projects', project_name), exist_ok=True)
        gen_tables(project_name)
        gen_clips(project_name, audio_path, clip_duration)
        gen_queue(project_name, n=10)
        # gen_embeddings(project_name, embedding_model)
    except Exception as e:
        logger.error(f"Failed to initialize project '{project_name}': {str(e)}")
        raise


def gen_clips(project_name, audio_path, clip_duration):
    """
    Generates audio clips for the project.
    """
    path_clips = os.path.join('projects', project_name, 'clips')
    os.makedirs(path_clips, exist_ok=True)
    list_files = get_list_files(audio_path)
    logging.info(f'Generating audio clips for project {project_name} with {len(list_files)} audio files')
    with Pool() as pool:
        pool.starmap(split_single_audio, [(file, path_clips, clip_duration) for file in list_files])


def gen_tables(project_name):
    """
    Generates the initial vocabulary and annotations files for the project.
    """
    logging.info(f'Generating initial vocabulary and annotations files for project {project_name}')
    with open(os.path.join('projects', project_name, 'vocabulary.txt'), 'x') as f:
        os.utime(f.fileno(), None)

    annotations = pd.DataFrame({
        'sound_clip_url': [],
        'label': [],
        'timestamp': []
    }, index=pd.Index([], name='id'))
    annotations.to_csv(os.path.join('projects', project_name, 'annotations.csv'))


def gen_queue(project_name, n=5):
    """
    Generates a queue of audio clips to be annotated.
    """
    all_clips = glob_audio_dataset(os.path.join('projects', project_name, 'clips'))
    n = min(n, len(all_clips))
    logging.info(f'Generating queue of {n} audio clips for project {project_name}')
    try:
        sound_clip_url = np.random.choice(all_clips, n, replace=False)
        queue = pd.DataFrame({
            'sound_clip_url': [os.path.basename(p) for p in sound_clip_url],
            'status': 'pending',
            'method': 'random',
            # 'priority': 0,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, index=pd.Index(range(n), name='id'))
        queue.to_csv(os.path.join('projects', project_name, 'queue.csv'))
    except Exception as e:
        logger.error(f"Error generating queue for project '{project_name}': {str(e)}")
        raise


def gen_embeddings(project_name, embedding_model):
    """
    Generates embeddings for the audio clips in the project.
    """
    logging.info(f'Generating embeddings for project {project_name}')
    all_clips = glob_audio_dataset(os.path.join('projects', project_name, 'clips'))

    try:
        if embedding_model == 'birdnet':
            # Use the repository model path (src/assets/...) to avoid ambiguity when running from project root.
            model_dir = os.path.join('src', 'assets', 'models', 'birdnet', 'V2.4', 'BirdNET_GLOBAL_6K_V2.4_Model')
            if not os.path.exists(model_dir):
                logger.error(f"BirdNET model directory not found at {model_dir}")
                raise FileNotFoundError(f"BirdNET model directory not found: {model_dir}")
            model = tf.keras.layers.TFSMLayer(model_dir, call_endpoint='embeddings')
            sr = 48000
        else:
            raise ValueError(f"Unknown embedding model: {embedding_model}")

        audio_dataset = load_audio_files_with_tf_dataset(all_clips, sample_rate=sr)
        results = tf.keras.Sequential([model]).predict(audio_dataset.batch(128), verbose=2)

        embeddings_df = pd.DataFrame(
            index=[os.path.basename(file) for file in all_clips],
            data=results['embeddings']
        )

        embeddings_df.to_pickle(os.path.join('projects', project_name, 'embeddings.pkl'))
    except Exception as e:
        logger.error(f"Error generating embeddings for project '{project_name}': {str(e)}")
        raise


# %%
@callback(
    Output('dataset-list', 'options'),
    Output('dataset-list', 'value'),
    Output('project-content', 'data', allow_duplicate=True),
    Input('dataset-list', 'value'),
    Input('button-project-create', 'n_clicks'),
    State('navbar', 'brand'),
    State('dataset-name', 'value'),
    State('audio-path', 'value'),
    State('embedding-model', 'value'),
    State('project-content', 'data'),
    prevent_initial_call=True
)
def update_options_project(project_value, project_create, brand, project_name, path_audio, embedding_model, data):
    if dash.ctx.triggered_id == 'button-project-create':
        # dask_client.submit(register_dataset, dataset_name=project_name, path_audio=path_audio)
        register_dataset(dataset_name=project_name, path_audio=path_audio)
        project_value = project_name
    elif data.get('project_name') and not project_value:
        project_value = data['project_name']

    options = [{'label': x, 'value': x} for x in list_existing_datasets()]
    data['project_name'] = project_value
    data['current_sample'] = ''

    return options, project_value, data


@callback(
    Output('dataset-summary', 'children'),
    Input('dataset-list', 'value')
)
def update_project_summary(dataset_name):
    children = [html.H5('Dataset summary')]
    if not dataset_name:
        children.append(html.P("No dataset selected. Please select a dataset to view the summary."))
        return children
    try:
        with server.app_context():
            path_dataset = sqlalchemy_db.session.execute(
                sqlalchemy_db.select(Dataset.path_audio).where(
                    Dataset.dataset_name == dataset_name)).scalar_one_or_none()
        if path_dataset is None:
            children.append(html.P(f"Dataset '{dataset_name}' does not have a valid audio path."))
            return children
        if not os.path.exists(path_dataset):
            children.append(html.P(f"The audio path '{path_dataset}' does not exist."))
            return children

        all_clips = glob_audio_dataset(path_dataset=path_dataset)
        # annotations = pd.read_csv(os.path.join('projects', project_name, 'annotations.csv'))
        # n_labeled = len(annotations['sound_clip_url'].unique())
        # n_classes = len(annotations['label'].unique())
        n_labeled = '[UNK]'
        n_classes = '[UNK]'
        msg = f'Found {len(all_clips)} audio files, {n_labeled} of which are labelled for {n_classes} class categories'
        if n_classes == 1: msg = msg.replace('categories', 'category')
        children.append(html.P(msg))
    #     children.append(dcc.Link("Start annotating", href='/annotate'))
    #     children.append(html.P('Export project data'))

    except SQLAlchemyError as e:
        # Log the error and return a friendly error message
        logger.error(f"Error fetching dataset '{dataset_name}': {e}")
        children.append(
            html.P(f"An error occurred while fetching the dataset '{dataset_name}'. Please try again later."))
    except Exception as e:
        logger.error(f"Unexpected error while processing dataset '{dataset_name}': {e}")
        children.append(html.P(f"An unexpected error occurred. Please try again later."))

    return children


@callback(
    Output('button-project-create', 'disabled'),
    Input('dataset-name', 'valid'),
    Input('audio-path', 'valid'),
    Input('embedding-model', 'valid'),
)
def check_create(v1, v2, v3):
    return not (v1 and v2 and v3)


@callback(
    Output("collapse-new-dataset", "is_open"),
    Input("btn-new-dataset", "n_clicks"),
    Input("button-project-cancel", "n_clicks"),
    State("collapse-new-dataset", "is_open"),
)
def toggle_collapse(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@callback(
    Output('dataset-name', 'valid'),
    Output('dataset-name', 'invalid'),
    Input('dataset-name', 'value')
)
def check_project_name(value):
    if value:
        is_valid = value not in list_existing_datasets()
        return is_valid, not is_valid
    return False, False


@callback(
    Output('audio-path', 'valid'),
    Output('audio-path', 'invalid'),
    Input('audio-path', 'value')
)
def check_audio_path(value):
    if value:
        is_valid = os.path.isdir(os.path.abspath(value))
        return is_valid, not is_valid
    return False, False


@callback(
    Output('embedding-model', 'valid'),
    Output('embedding-model', 'invalid'),
    Output('embedding-model', 'options'),
    Input('embedding-model', 'value'),
    Input('btn-new-dataset', 'n_clicks'),
    prevent_initial_call=False  # Allow initial call to populate options
)
def check_embedding_model(value, _):
    from src.pages.explore.callbacks import list_existing_methods, update_db_methods
    try:
        # First, ensure methods are registered in DB
        add_methods = update_db_methods()
        if add_methods:
            logger.info(f"Added new embedding methods to DB: {add_methods}")
            sqlalchemy_db.session.add_all(add_methods)
            sqlalchemy_db.session.commit()
        
        # Get methods from DB
        methods = list_existing_methods('embeddings')
        logger.info(f"Available embedding methods: {methods}")
        
        # Update options whenever the callback is triggered
        options = [{'label': method, 'value': method} for method in methods]
        logger.info(f"Generated dropdown options: {options}")
        
        if value:
            return True, False, options
        return False, False, options
    except Exception as e:
        logger.exception("Error in embedding model callback")
        # Return empty options but don't break the UI
        return False, True, []

# layout = html.Div([
#     html.Label('Select project'),
#     dcc.Dropdown(
#         id='dataset-list-dropdown',
#         options=[{'label': 'True', 'value': 'True'}, {'label': 'False', 'value': 'False'}],
#     ),
#
#     html.H1('Environment Variable Control Panel'),
#
#
#     html.Label('Database URL'),
#     dcc.Input(
#         id='database-url-input',
#         type='text',
#         # value=database_url,
#         style={'width': '100%'}
#     ),
#
#     html.Label('API Key'),
#     dcc.Input(
#         id='api-key-input',
#         type='text',
#         # value=api_key,
#         style={'width': '100%'}
#     ),
#
#     html.Button('Save Changes', id='save-button', n_clicks=0),
#
#     html.Div(id='save-status')
# ])


#
# #
# #
# # @app.callback(
# #     Output('save-status', 'children'),
# #     [Input('save-button', 'n_clicks')],
# #      State('database-url-input', 'value'),
# #      State('api-key-input', 'value')]
# # )
# # def update_env_file(n_clicks, db_url_value, api_key_value):
# #     if n_clicks > 0:
# #         # Update the .env file
# #         set_key('.env', 'DATABASE_URL', db_url_value)
# #         set_key('.env', 'API_KEY', api_key_value)
# #
# #         return 'Environment variables updated!'
# #     return ''
# #
