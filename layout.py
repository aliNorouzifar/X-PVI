from dash import html, dcc
import dash_uploader as du
from pm4py.objects.log.importer.xes.variants.iterparse import import_from_string
from functions import my_functions
from dash import html, dcc
import dash_bootstrap_components as dbc
import base64
import dash_uploader as du
import pm4py
import uuid
from pathlib import Path
import dash_daq as daq



def create_layout():
    return dbc.Container([
        html.H1("Process Variant Identification", className='text-center text-primary mb-4',
                style={'textAlign': 'center'}),
        html.H4("Upload the event log", className='text-left bg-light mb-4', style={'textAlign': 'left'}),
        html.Div([
            html.Div(
                children=[
                    get_upload_component("upload-data")
                ],
                style={  # wrapper div style
                    "textAlign": "center",
                    "width": "1200px",
                    "padding": "10px",
                    "display": "inline-block",
                },
            ),
            html.Div(id='output-data-upload'),
            dbc.Row([
                dbc.Col([
                    html.Img(id='bar-graph-matplotlib')
                ], width={'size': 5, 'offset': 1}),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Img(id='bar-graph-matplotlib2')
                ], width={'size': 5, 'offset': 1})
            ]),
            dbc.Row([
                dbc.Col([
                    html.Img(id='bar-graph-matplotlib3')
                ], width={'size': 5, 'offset': 1})
            ]),
            dbc.Row([
                dbc.Col([
                    html.Img(id='bar-graph-matplotlib4')
                ], width={'size': 6, 'offset': 3})
            ]),
            html.Div(id='output-data-upload2'),
        ])
    ])

def get_upload_component(id):
    return du.Upload(
        id=id,
        max_file_size=500,  # 50 Mb
        chunk_size=400,  # 4 MB
        filetypes=["xes"],
        # upload_id=uuid.uuid1(),  # Unique session id
        upload_id="temp_log"
    )



# def upload_view(max_par,columns):
#     return html.Div([
#     html.Div([
#         # Left Section: Inputs
#         html.Div(
#             id="left-section",
#             children=[
#         html.Hr(),
#         html.H4("Which process indicator?", className='text-left bg-light mb-4', style={'textAlign': 'left'}),
#         dcc.Dropdown(id='xaxis-data',
#                      options=[{'label': x, 'value': x} for x in columns]),
#         # For debugging, display the raw contents provided by the web browser
#         html.H4("How many buckets?", className='text-left bg-light mb-4', style={'textAlign': 'left'}),
#         html.Div([
#             daq.NumericInput(
#                 id='my-numeric-input-1',
#                 min=2,
#                 max=max_par,
#                 value=min(100,max_par)
#             ),
#             html.Div(id='numeric-input-output-1')
#         ]),
#         html.H4("Window size?", className='text-left bg-light mb-4', style={'textAlign': 'left'}),
#         html.Div([
#             daq.NumericInput(
#                 id='my-numeric-input-2',
#                 min=0,
#                 max=max_par/2,
#                 value=2
#             ),
#                     html.Div(id='numeric-input-output-2')
#                 ]),
#                 html.H4("theta_cvg for pruning?", className='text-left bg-light mb-4', style={'textAlign': 'left'}),
#                 html.Div([
#                     daq.NumericInput(
#                         id='my-numeric-input-3',
#                         min=0,
#                         max=0.1,
#                         value=0.02
#                     ),
#                     html.Div(id='numeric-input-output-3')
#                 ]),
#
#
#         html.H4("Number of Clusters?", className='text-left bg-light mb-4', style={'textAlign': 'left'}),
#         html.Div([
#             daq.NumericInput(
#                 id='my-numeric-input-4',
#                 min=0,
#                 max=20,
#                 value=5
#             ),
#             html.Div(id='numeric-input-output-4'),
#             ]),
#         html.H4("What is a significant distance?", className='text-left bg-light mb-4', style={'textAlign': 'left'}),
#         html.Div(
#             [
#                 dcc.Slider(
#                     id='my-slider3',
#                     min=0,
#                     max=1,
#                     step=0.05,
#                     value=0.15,
#                     vertical=False,
#                 ),
#                 html.Div(id='slider-output-container3'),
#             ],
#         ),
#         html.H4("Faster (not accurate)?", className='text-left bg-light mb-4', style={'textAlign': 'left'}),
#         dcc.RadioItems(id='TF',
#                        options=[
#                            {'label': 'True', 'value': True},
#                            {'label': 'False', 'value': False}
#                        ],
#                        value=False
#                        ),
#         html.H4("Export the segments?", className='text-left bg-light mb-4', style={'textAlign': 'left'}),
#         dcc.RadioItems(id='TF2',
#                        options=[
#                            {'label': 'True', 'value': True},
#                            {'label': 'False', 'value': False}
#                        ],
#                        value=False
#                        ),
#         html.Button(id="submit-button", children="Run"),
#         html.Hr(),
#     ],  # Replace with actual max_par and columns
#             style={"width": "30%", "display": "inline-block", "verticalAlign": "top", "padding": "10px"}
#         ),
#         # Right Section: Outputs
#         html.Div(
#             id="right-section",
#             children=[
#                 html.Img(id="bar-graph-matplotlib", style={"width": "100%", "margin-bottom": "20px"}),  # First graph
#                 html.Img(id="bar-graph-matplotlib2", style={"width": "40%", "margin-bottom": "20px"}),   # Second graph
#                 html.Img(id="bar-graph-matplotlib3", style={"width": "100%", "margin-bottom": "20px"}),
#                 html.Img(id="bar-graph-matplotlib4", style={"width": "50%", "margin-bottom": "20px"})
#             ],
#             style={"width": "65%", "display": "inline-block", "verticalAlign": "top", "padding": "10px"}
#         )
#     ])
# ])

def upload_view(max_par, columns):
    return html.Div([
        # Wrapper for all content
        html.Div([
            # Top Section: Inputs
            html.Div(
                id="top-section",
                children=[
                    html.Hr(),
                    html.H4("Which process indicator?", className='text-left bg-light mb-4', style={'textAlign': 'left'}),
                    dcc.Dropdown(id='xaxis-data',
                                 options=[{'label': x, 'value': x} for x in columns]),
                    html.H4("How many buckets?", className='text-left bg-light mb-4', style={'textAlign': 'left'}),
                    html.Div([
                        daq.NumericInput(
                            id='my-numeric-input-1',
                            min=2,
                            max=max_par,
                            value=min(100, max_par)
                        ),
                        html.Div(id='numeric-input-output-1')
                    ]),
                    html.H4("Window size?", className='text-left bg-light mb-4', style={'textAlign': 'left'}),
                    html.Div([
                        daq.NumericInput(
                            id='my-numeric-input-2',
                            min=0,
                            max=max_par / 2,
                            value=2
                        ),
                        html.Div(id='numeric-input-output-2')
                    ]),
                    html.H4("theta_cvg for pruning?", className='text-left bg-light mb-4', style={'textAlign': 'left'}),
                    html.Div([
                        daq.NumericInput(
                            id='my-numeric-input-3',
                            min=0,
                            max=0.1,
                            value=0.02
                        ),
                        html.Div(id='numeric-input-output-3')
                    ]),
                    html.H4("Number of Clusters?", className='text-left bg-light mb-4', style={'textAlign': 'left'}),
                    html.Div([
                        daq.NumericInput(
                            id='my-numeric-input-4',
                            min=0,
                            max=20,
                            value=5
                        ),
                        html.Div(id='numeric-input-output-4'),
                    ]),
                    html.H4("What is a significant distance?", className='text-left bg-light mb-4', style={'textAlign': 'left'}),
                    html.Div(
                        [
                            dcc.Slider(
                                id='my-slider3',
                                min=0,
                                max=1,
                                step=0.05,
                                value=0.15,
                                vertical=False,
                            ),
                            html.Div(id='slider-output-container3'),
                        ],
                    ),
                    html.H4("Faster (not accurate)?", className='text-left bg-light mb-4', style={'textAlign': 'left'}),
                    dcc.RadioItems(id='TF',
                                   options=[
                                       {'label': 'True', 'value': True},
                                       {'label': 'False', 'value': False}
                                   ],
                                   value=False
                                   ),
                    html.H4("Export the segments?", className='text-left bg-light mb-4', style={'textAlign': 'left'}),
                    dcc.RadioItems(id='TF2',
                                   options=[
                                       {'label': 'True', 'value': True},
                                       {'label': 'False', 'value': False}
                                   ],
                                   value=False
                                   ),
                    html.Button(id="submit-button", children="Run"),
                    html.Hr(),
                ],
                style={"width": "100%", "padding": "10px"}
            ),
            # Bottom Section: Outputs
            html.Div(
                id="bottom-section",
                children=[
                    html.Img(id="bar-graph-matplotlib", style={"width": "100%", "margin-bottom": "20px"}),  # First graph
                    html.Img(id="bar-graph-matplotlib2", style={"width": "40%", "margin-bottom": "20px"}),  # Second graph
                    html.Img(id="bar-graph-matplotlib3", style={"width": "100%", "margin-bottom": "20px"}),
                    html.Img(id="bar-graph-matplotlib4", style={"width": "50%", "margin-bottom": "20px"})
                ],
                style={"width": "100%", "padding": "10px"}
            )
        ],
            style={"display": "flex", "flexDirection": "column", "alignItems": "center"}
        )
    ])
