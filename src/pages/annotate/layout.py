import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from .callbacks import register_callbacks
from .. import login_required_layout

dash.register_page(
    __name__,
    path='/annotate',
    title='Annotation | YAPAT'
)

register_callbacks()


@login_required_layout
def layout():
    layout = html.Div([  # dbc.Container([
        dbc.Row([
            dbc.Col([], width=8),
            dbc.Col([
                html.H4('0 / 0', id='counter'),
            ], align='right', class_name='mx-4')
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Row(html.H6(id='text_complete_file_name', style={'display': 'block'}), justify='center'),
                dbc.Row(
                    dcc.Loading(dcc.Graph(id='spectrogram', config={'displaylogo': False}, className='my-4'),
                                type='circle'),
                    justify='center',
                ),
                dbc.Row(
                    html.Audio(
                        id='audio_complete_file',
                        controls=True,
                        autoPlay=False,
                        style={'width': '70%'}
                    ), justify='center',
                ),
                dbc.Row([
                    html.H5('Next sample:', className='mt-5 mb-3'),
                    html.Div([
                        dbc.RadioItems(
                            id="retrieval-method", className="btn-group", inputClassName="btn-check",
                            labelClassName="btn btn-outline-primary", labelCheckedClassName="active",
                            options=[
                                {"label": "Validate", "value": "validate", "disabled": True},
                                {"label": "Discover", "value": "discover", "disabled": True},
                                {"label": "Manual", "value": "manual", "disabled": True},
                                {"label": "Random", "value": "random"},
                                {"label": "Refine", "value": "refine", "disabled": False},
                                {"label": "Explore", "value": "explore", "disabled": False},
                            ],
                            value="explore",
                        )
                    ], className="radio-group my-3 mb-4")
                ], className='mb-4'),
            ], width='8', className='mb-5'),
            dbc.Col([
                dbc.Row([
                    html.Div([
                        # dbc.FormFloating([
                        #     dbc.Input(id='input_search', type='text', placeholder=''),
                        #     dbc.Label("Label search"),
                        # ], class_name='my-1'),
                    ])
                ],
                    align='center'
                ),
                dbc.Row(
                    dcc.Checklist(
                        id='checklist_annotation',
                        options=[],
                        style={'maxHeight': 300, 'height': 300, 'overflow-y': 'auto'}
                    ), class_name='my-3'
                ),
                dbc.Row(
                    html.Div([
                        dbc.FormFloating([
                            dbc.Input(type="text", id="new-class", placeholder='', debounce=True),
                            dbc.Label("Add label class"),
                            # dbc.FormFeedback('This name is already in use. Please select another one.',
                            #                  type='invalid')
                        ], class_name='my-1'),
                    ]), align='center'),
                dbc.Row([
                    dbc.ButtonGroup([
                        dbc.Button('submit',
                                   id='button_submit', color='success', size='lg',
                                   # n_clicks=0,
                                   # style={'width': '80%', 'height': '150%'}
                                   ),
                        dbc.Button('skip',
                                   id='button_skip', color='danger', size='lg',
                                   # n_clicks=0,
                                   # style={'width': '80%', 'height': '150%'}
                                   ),
                    ], class_name='mx-my-3')
                ]),
                # dbc.Row([
                #     html.H6('Dialogue'),
                #     dcc.Markdown(id='dialog-out', children='*Hello world*'),
                #     dcc.Textarea(id='dialog-in', placeholder='type in', disabled=True, className='mx-4')
                # ], class_name='my-4')
            ], class_name='mx-4')
        ])
        # ], fluid='xxl', style={'width': '100%', 'display': 'inline'})
    ])
    return layout

# col_annotation = html.Div(
#     [
#         # dcc.Interval(id='interval_1s', interval=1000),
#
#         html.H6(id='text_model_suggestions'),
#
#         html.Hr(),
#
#         dbc.Row(
#             [
#                 html.Hr(),
#                 dbc.Col(
#                     html.H6(id='text_species_to_process'),
#                     width='auto',
#                     style={'padding-right': '1'}
#                 ),
#                 dbc.Col(
#                     dcc.Dropdown(id='dropdown_selected_species', multi=True),
#                     width=12,
#                     style={'padding-right': '0', 'padding-left': '0'}
#                 )
#             ],
#             align='center',
#             id='row_species_to_process',
#             style={'display': 'block'}
#         ),
#
#         html.Hr(),
#
#     ]
# )
