from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_uploader as du
import dash_daq as daq
import json
import pandas as pd

def load_variables():
    try:
        with open("output_files\internal_variables.json", "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        return "No data file found."
    df = pd.read_json(data["df"], orient="split")
    data["df"] = df
    return data

def create_layout():
    return dbc.Container(className="page-container",
    children=[html.Div(className="nav-links",
            children=[
                dcc.Link("Introduction", href="/", className="nav-link"),
                dcc.Link("X-PVI", href="/main", className="nav-link"),
                dcc.Link("About Me", href="/about_me", className="nav-link")
        ]),
        html.H1("Process Variant Identification", className='header'),
        html.Div(
            className="flex-container",
            children=[
                # Left Side: Parameter Settings
                html.Div(
                    id="left-panel",
                    children=[
                        html.H4("Parameters Settings", className='text-left bg-light mb-4'),
                        html.H4("Upload the Event Log", className='text-left bg-light mb-4'),
                        html.Div(
                            children=[
                                get_upload_component("upload-data")
                            ],
                            className="upload-wrapper"
                        ),
                        html.Div(
                            className="parameters-wrapper",
                            children=[
                                html.Div(id='output-data-upload', className="parameter-block"),
                                html.Div(id='output-data-upload2', className="parameter-block"),
                                html.Div(id='output-data-upload4', className="parameter-block"),
                                html.Div(id='output-data-upload6', className="parameter-block"),
                            ]
                        )
                    ]
                ),

                # Right Side: Visualization Blocks
                html.Div(
                    id="right-panel",
                    children=[
                        html.Div(
                            className="visualization-wrapper",
                            children=[
                                html.Div(id='output-data-upload3', className="visualization-block"),
                                html.Div(id='output-data-upload5', className="visualization-block"),
                                html.Div(id='output-data-upload7', className="visualization-block"),
                            ]
                        )
                    ]
                )
            ]
        )
    ])

def get_upload_component(id):
    return du.Upload(
        id=id,
        max_file_size=500,
        chunk_size=100,
        max_files=1,
        filetypes=["xes"],
        upload_id="event_log",
    )

def parameters_view_PVI(max_par, columns):
    return html.Div([
        html.Div([
            html.Div(
                id="top-section",
                className="page-container",
                children=[
                    html.Hr(),
                    html.H4("Which process indicator?", className='text-left bg-light mb-4'),
                    dcc.Dropdown(id='xaxis-data', options=[{'label': x, 'value': x} for x in columns]),
                    html.H4("How many buckets?", className='text-left bg-light mb-4'),
                    html.Div([
                        daq.NumericInput(
                            id='my-numeric-input-1',
                            min=2,
                            max=max_par,
                            value=min(100, max_par)
                        ),
                        html.Div(id='numeric-input-output-1')
                    ]),
                    html.H4("Window size?", className='text-left bg-light mb-4'),
                    html.Div([
                        daq.NumericInput(
                            id='my-numeric-input-2',
                            min=0,
                            max=max_par / 2,
                            value=2
                        ),
                        html.Div(id='numeric-input-output-2')
                    ]),
                    html.H4("What is a significant distance?", className='text-left bg-light mb-4'),
                    html.Div([
                        dcc.Slider(
                            id='my-slider3',
                            min=0,
                            max=1,
                            step=0.05,
                            value=0.15,
                            vertical=False
                        ),
                        html.Div(id='slider-output-container3')
                    ]),
                    html.H4("Faster (not accurate)?", className='text-left bg-light mb-4'),
                    dcc.RadioItems(
                        id='TF',
                        options=[
                            {'label': 'True', 'value': True},
                            {'label': 'False', 'value': False}
                        ],
                        value=False
                    ),
                    html.H4("Export the segments?", className='text-left bg-light mb-4'),
                    dcc.RadioItems(
                        id='TF2',
                        options=[
                            {'label': 'True', 'value': True},
                            {'label': 'False', 'value': False}
                        ],
                        value=False
                    ),
                    html.Button(id="run_PVI", children="Run PVI", className="btn-primary", n_clicks=0),
                    html.Hr(),
                ]
            )
        ],
        className="flex-column align-center")
    ])

def PVI_figures(fig_src1, fig_src2):
    return html.Div(
        id="bottom-section",
        className="page-container",
        children=[
            html.Img(id="bar-graph-matplotlib", src=fig_src1, className="figure", style={"width": "100%", "height": "auto"}),
            html.Img(id="bar-graph-matplotlib2", src=fig_src2, className="figure small-figure", style={"width": "100%", "height": "auto"}),
            html.Button(id="X_parameters", children="Start The Explainability Extraction Framework!", className="btn-secondary", n_clicks=0)
        ]
    )

def parameters_view_explainability():
    return html.Div([
        html.Div([
            html.Div(
                id="top-section",
                className="page-container",
                children=[
                    html.Hr(),
                    html.H4("theta_cvg for pruning?", className='text-left bg-light mb-4'),
                    html.Div([
                        daq.NumericInput(
                            id='my-numeric-input-3',
                            min=0,
                            max=0.1,
                            value=0.02
                        ),
                        html.Div(id='numeric-input-output-3')
                    ]),
                    html.H4("Number of Clusters?", className='text-left bg-light mb-4'),
                    html.Div([
                        daq.NumericInput(
                            id='my-numeric-input-4',
                            min=0,
                            max=20,
                            value=5
                        ),
                        html.Div(id='numeric-input-output-4')
                    ]),
                    html.Button(id="XPVI_run", children="XPVI Run", className="btn-primary", n_clicks=0)
                ]
            )
        ],
        className="flex-column align-center")
    ])

def XPVI_figures(fig_src3, fig_src4):
    return html.Div(
        id="bottom-section",
        className="page-container",
        children=[
            html.Img(id="bar-graph-matplotlib", src=fig_src3, className="figure", style={"width": "100%", "height": "auto"}),
            html.Img(id="bar-graph-matplotlib2", src=fig_src4, className="figure small-figure", style={"width": "100%", "height": "auto"}),
            html.Button(id="decl2NL_framework", children="Convert Declare to Natural Language!", className="btn-secondary", n_clicks=0)
        ]
    )

def decl2NL_parameters():
    data = load_variables()
    segments_count = data["segments_count"]
    clusters_count = data["clusters_count"]
    return html.Div([
        html.Div(className="page-container",
            children=[
                html.H4("Which segment?", className='text-left bg-light mb-4'),
                dcc.Dropdown(id='segment_number', options=[{'label': x, 'value': x} for x in range(1, segments_count + 1)]),
                html.H4("Which cluster?", className='text-left bg-light mb-4'),
                dcc.Dropdown(id='cluster_number', options=[{'label': x, 'value': x} for x in range(1, clusters_count + 1)]),
                html.Button(id="decl2NL_pars", children="Show decl2NL parameters!", className="btn-primary", n_clicks=0)
            ]
        )
    ])


def statistics_print(list_sorted, list_sorted_reverse):
    return html.Div(
        className="page-container",
        children=[
            html.H4("Lowest Scores:", className='text-left bg-light mb-4'),
            html.Ul([html.Li(sentence, className="list-item") for sentence in list_sorted]),
            html.H4("Highest Scores:", className='text-left bg-light mb-4'),
            html.Ul([html.Li(sentence, className="list-item") for sentence in list_sorted_reverse])
        ]
    )
