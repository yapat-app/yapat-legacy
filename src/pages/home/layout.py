import logging

import dash
import dash_bootstrap_components as dbc
from dash import html

from src.pages.home import callbacks
# from .callbacks import scan_projects
from src.pages import login_required_layout
from src.pages.explore.callbacks import list_existing_methods


logger = logging.getLogger(__name__)

dash.register_page(
    __name__,
    path='/',
    redirect_from=['/home'],
    title='Home | YAPAT'
)


@login_required_layout
def layout():
    layout = dbc.Container([
        html.Div([
            # Header
            dbc.Row([
                html.H1('Welcome to YAPAT'),
                html.H6(["YAPAT is a smart annotation tool designed for passive acoustic monitoring data, ",
                         "using machine learning for efficient labeling and discovery of new sounds."]),
                html.P(["Select a dataset below, or start a new one. ",
                        "Check the ",
                        html.A("documentation", href="https://yapat.readthedocs.io/", target="_blank"),
                        " for guidance."])
                # html.Div(
                #     html.A('Select a project or start a new one here.', href='/project')
                # ),
                # html.Div(id='content')
            ]),
            # Main
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        html.H5('Select dataset'),
                        dbc.RadioItems(
                            id='dataset-list',
                            options=[],
                        ),
                    ], class_name='my-4'),
                    dbc.Row([
                        html.Div([
                            dbc.Button('New dataset...', id='btn-new-dataset', class_name='my-4'),
                            dbc.Collapse([
                                html.Div([
                                    dbc.FormFloating([
                                        dbc.Input(placeholder="Type name of new dataset...", type="text",
                                                  id="dataset-name", debounce=True),
                                        dbc.Label("Dataset name"),
                                        dbc.FormFeedback('This name is already in use. Please select another one.',
                                                         type='invalid')
                                    ], class_name='my-1'),
                                ]),
                                html.Div([
                                    dbc.FormFloating([
                                        dbc.Input(placeholder="Type path to directory containing audio files...",
                                                  type="text", debounce=True, id='audio-path'),
                                        dbc.Label("Path to audio directory"),
                                        dbc.FormFeedback('Not a valid directory path', type='invalid')
                                    ], class_name='py-1'),
                                ]),
                                dbc.Select(
                                    options=[{'label': method, 'value': method} for method in
                                     list_existing_methods('embeddings')],
                                    placeholder="Select embedding model", class_name='my-1', id='embedding-model'
                                ),
                                dbc.ButtonGroup([
                                    dbc.Button('Cancel', color='danger', id='button-project-cancel', class_name='mx-1'),
                                    dbc.Button('Create', id='button-project-create', class_name='mx-1')
                                ], class_name='my-2')],
                                id="collapse-new-dataset",
                                is_open=False
                            )
                        ]),
                    ], class_name='my-4')
                ], width=3),
                dbc.Col([
                    dbc.Row([
                        html.H5('Dataset summary'),
                    ], class_name='my-4', id='dataset-summary'),
                    dbc.Row([
                        # dbc.DropdownMenu()
                    ], class_name='my-4')
                ])
            ])
        ])
    ])
    return layout
