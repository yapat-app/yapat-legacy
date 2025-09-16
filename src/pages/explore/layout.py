import logging

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
from src.pages import login_required_layout
from src.pages.explore.callbacks import list_existing_methods

logger = logging.getLogger(__name__)

dash.register_page(
    __name__,
    path='/explore',
    redirect_from=['/visualise', '/visualize', '/viz'],
    title='Explore | YAPAT'
)


# layout = html.Header([
#     html.H1('Explore'),
#     dbc.Breadcrumb(
#         items=[
#             {"label": "Embed", "href": "/docs", "external_link": True},
#             {
#                 "label": "Cluster",
#                 "href": "/docs/components",
#                 "external_link": True,
#             },
#             {"label": "Visualize", "active": True},
#         ],
#     )
# ])


def create_modal(title, button_text, id_prefix):
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(title)),
            dbc.ModalBody([
                html.P("Representation space"),
                dcc.Dropdown(
                    options=[{'label': method, 'value': method} for method in list_existing_methods('embeddings')],
                    value='birdnet', multi=False,
                    placeholder="Select embedding",
                    id=f"{id_prefix}-methods-embedding",
                ),
                html.P("Clustering"),
                dcc.Dropdown(
                    options=[{'label': method, 'value': method} for method in list_existing_methods('clustering')],
                    value='kmeans', multi=False,
                    placeholder="Select clustering method",
                    id=f"{id_prefix}-methods-clustering"
                ),
                html.P("Dimensionality reduction (for visualization)"),
                dcc.Dropdown(
                    options=[{'label': method, 'value': method} for method in
                             list_existing_methods('dimensionality_reduction')],
                    value='umap_reducer', multi=False,
                    placeholder="Select post-clustering reduction",
                    id=f"{id_prefix}-methods-dimred-viz"
                )
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id=f"{id_prefix}-cancel", n_clicks=0, color="secondary"),
                dbc.Button(button_text, id=f"{id_prefix}-confirm", n_clicks=0, color="primary"),
            ])
        ],
        id=f"{id_prefix}-modal",
        centered=True,
        is_open=False
    )


@login_required_layout
def layout():
    columns = [
        {"name": "Metric", "id": "metric"},
        {"name": "BirdNET", "id": "birdnet"},
        {"name": "Acoustic Indices", "id": "acoustic_indices"},
        {"name": "VAE", "id": "vae"}
    ]

    # Define the data for each row
    data = [
        {"metric": "F1 Score (Time Prediction)", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
        {"metric": "Accuracy (Time Prediction)", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
        {"metric": "F1 Score (Location Prediction)", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
        {"metric": "Accuracy (Location Prediction)", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
        {"metric": "Explained Variance", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
        {"metric": "Entropy", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
        {"metric": "Silhouette Index", "birdnet": 0, "acoustic_indices": 0, "vae": 0},
        {"metric": "Davies Bouldin Index", "birdnet": 0, "acoustic_indices": 0, "vae": 0}
    ]

    layout = dbc.Container([
        html.Div([
            # Store for loaded figures
            dcc.Store(id='loaded-figures-store', data={}),
            # Header
            dbc.Row([
                html.H2("Visualization Pipelines"),
                html.P(["Mix and match methods for data representation/embedding, clustering, and "
                        "dimensionality reduction."
                        "Create new pipeline to calculate and Load Pipeline to evaluate and visualise."]),
            ]),

            # Main
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        # html.H5('Build pipeline'),
                        dbc.ButtonGroup([
                            # dbc.Button("Quickstart", id="quickstart-pipeline", n_clicks=0),
                            dbc.Button("New Pipeline (Compute)", id="new-pipeline", n_clicks=0),
                            dbc.Button("Load Pipeline (Visualize)", id="load-pipeline", n_clicks=0),
                            dbc.Button("Fetch Available Evaluations", id="fetch-eval", n_clicks=0),

                        ], vertical=True),
                    ], class_name='my-4'),
                    # Status Box
                    dbc.Row([
                        html.Div(id='status-box', children="", style={'marginTop': '20px'})
                    ])
                ], width=2),

                # Right Sidebar - Evaluation and Visualization
                dbc.Col([
                    dbc.Row([
                        html.H5("Evaluation Metrics"),
                        dash_table.DataTable(
                            id='evaluation-table',
                            columns=columns,
                            data=data,
                            merge_duplicate_headers=True,  # This will merge headers like 'birdnet' in one line
                            style_table={'overflowX': 'auto'},
                            style_cell={'textAlign': 'center'}
                        ),
                    ], class_name='my-4'),
                    dbc.Row([
                        # html.H5('Visualize results'),
                    ], class_name='my-4', id='pipeline-summary'),
                    dbc.Row([
                        # dbc.DropdownMenu()
                    ], class_name='my-4'),
                    html.P(id="create-pipeline-msg"),
                    dbc.Row([
                        # dbc.DropdownMenu()
                    ], class_name='my-4'),
                    html.P(id="create-pipeline-msg"),
                    dbc.Row([
                        # Visualization Section
                        html.H5("Visualization"),
                        dbc.Tabs([
                            dbc.Tab(label='Cluster Time Histogram', tab_id='cluster-time-histogram'),
                            dbc.Tab(label='Cluster Time Grid', tab_id='cluster-time-grid'),
                            dbc.Tab(label='State Space Visualisation', tab_id='cluster-state-space'),
                            dbc.Tab(label='Time Series Visualisation', tab_id='time-series'),
                            dbc.Tab(label='Temporal Rose Plot', tab_id='temp-rose-plot'),
                            dbc.Tab(label='Explained Variance Plot', tab_id='explained-variance'),

                        ], id='visualization-tabs', active_tab='cluster-time-histogram'),

                        # Placeholder for visualizations
                        html.Div(id='visualization-content', className='my-4')
                    ])
                ], width=9),
            ]),

            # Modals
            create_modal("Create New Pipeline", "Create", "new"),
            create_modal("Load Pipeline to Visualize", "Visualize Results", "load"),
            html.Div(id='loaded-figures', style={'display': 'none'})

            #         dbc.ModalFooter([
            #             html.P(id='new-pipeline-summary'),
            #             dbc.ButtonGroup([
            #                 dbc.Button("Cancel", id="cancel-new-pipeline", n_clicks=0, color="secondary"),
            #                 dbc.Button("Create", id="create-pipeline", n_clicks=0, color="primary"),
            #             ])
            #
            #         ]),
            #     ],
            #     id="modal-pipeline",
            #     centered=True,
            #     is_open=False,
            # ),
        ])
    ])
    return layout
