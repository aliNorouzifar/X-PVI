from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_uploader as du
import dash_daq as daq
import json
import pandas as pd


def load_variables():
    try:
        with open("output_files/internal_variables.json", "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        return "No data file found."
    df = pd.read_json(data["df"], orient="split")
    data["df"] = df
    return data

def create_layout():
    return dbc.Container(
        className="page-container",
        children=[
            # Navigation Links
            html.Div(
                className="nav-links",
                children=[
                    dcc.Link("Introduction", href="/", className="nav-link"),
                    dcc.Link("X-PVI", href="/main", className="nav-link"),
                    dcc.Link("About Me", href="/about_me", className="nav-link"),
                ],
            ),
            # Tool Name and Description
            html.Div(
                className="tool-name-container",
                children=[
                    html.H1("Process Variant Identification", className="tool-name"),
                    html.P(
                        "A cutting-edge tool for detecting and understanding process variability across performance dimensions.",
                        className="tool-subtitle",
                    ),
                ],
            ),
            # Main Content Area
            html.Div(
                className="flex-container",
                children=[
                    # Left Panel: Parameter Settings
                    create_left_panel(),
                    # Right Panel: Visualization Blocks
                    create_right_panel(),
                ],
            ),
        ],
    )


def create_left_panel():
    return html.Div(
        id="left-panel",
        className="left-panel-container",
        children=[
            # Header for Parameters Settings
            html.Div(
                className="panel-header",
                children=html.H4("Parameters Settings", className="panel-title"),
            ),
            # Upload Section
            html.Div(
                className="upload-container",
                children=[
                    html.Div(
                        className="section-header",
                        children=html.H4("Upload the Event Log", className="section-title"),
                    ),
                    get_upload_component("upload-data"),
                ],
            ),
            # Parameters Section
            html.Div(
                className="parameters-wrapper",
                children=[
                    html.Hr(),

                    html.Div(id="output-data-upload2", className="parameter-block card"),
                    # html.Hr(),
                    html.Div(id="output-data-upload4", className="parameter-block card"),
                    # html.Hr(),
                    html.Div(id="output-data-upload6", className="parameter-block card"),
                ],
            ),
        ],
    )


def create_right_panel():
    return html.Div(
        id="right-panel",
        className="right-panel-container",
        children=[
            html.Div(
                className="panel-header",
                children=html.H4("Visualizations and Reports", className="panel-title"),
            ),
            html.Div(
                className="visualization-wrapper",
                children=[
                    html.Div(id="output-data-upload3", className="visualization-block"),
                    html.Hr(),
                    html.Div(id="output-data-upload5", className="visualization-block"),
                    html.Hr(),
                    html.Div(id="output-data-upload7", className="visualization-block"),
                ],
            ),
        ],
    )

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
                className="parameter-container",
                children=[
                    html.Div(
                        className="section-header",
                        children=html.H4("Process Variant Identification Parameters", className="section-title"),
                    ),
                    # html.Hr(),
                    html.H4("process indicator:", className="parameter-name"),
                    dcc.Dropdown(id='xaxis-data', options=[{'label': x, 'value': x} for x in columns]),
                    html.Hr(),
                    html.H4("N. buckets:", className="parameter-name"),
                    html.Div([
                        daq.NumericInput(
                            id='my-numeric-input-1',
                            min=2,
                            max=max_par,
                            value=min(100, max_par)
                        ),
                        html.Div(id='numeric-input-output-1')
                    ]),
                    html.Hr(),
                    html.H4("Window size:", className="parameter-name"),
                    html.Div([
                        daq.NumericInput(
                            id='my-numeric-input-2',
                            min=0,
                            max=max_par / 2,
                            value=2
                        ),
                        html.Div(id='numeric-input-output-2')
                    ]),
                    html.Hr(),
                    html.H4("significant distance:", className="parameter-name"),
                    html.Div([
                        daq.Knob(
                            id='my-slider3',
                            min=0,
                            max=1,
                            value=0.15,
                            size=100,  # Size of the circular slider
                            color="#007bff",  # Customize the color
                            # label="Significant Distance",
                            scale={'interval': 0.1}
                        ),
                        html.Div(id='slider-output-container3')
                    ]),
                    html.Hr(),
                    html.H4("Faster (not accurate)?", className="parameter-name"),
                    dcc.RadioItems(
                        id='TF',
                        options=[
                            {'label': 'True', 'value': True},
                            {'label': 'False', 'value': False}
                        ],
                        value=False,
                        inline=True
    ),
                    html.Hr(),
                    html.H4("Export the segments?", className="parameter-name"),
                    dcc.RadioItems(
                        id='TF2',
                        options=[
                            {'label': 'True', 'value': True},
                            {'label': 'False', 'value': False}
                        ],
                        value=False,
                        inline=True
                    ),
                    html.Hr(),
                    html.Button(id="run_PVI", children="Run PVI", className="btn-primary", n_clicks=0),
                    # html.Hr(),
                ]
            )
        ],
        className="flex-column align-center")
    ])

def parameters_view_explainability():
    return html.Div([
        html.Div([
            html.Div(
                className="parameter-container",
                children=[
                    html.Div(
                        className="section-header",
                        children=html.H4("Explainability Extraction parameters", className="section-title"),
                    ),
                    # html.Hr(),
                    html.H4("theta_cvg for pruning?", className="parameter-name"),
                    html.Div([
                        dcc.Input(
                            id='my-numeric-input-3',
                            type='number',
                            min=0,
                            max=1,
                            value=0.02,
                            step=0.01  # Specify the step size here
                        ),
                        html.Div(id='numeric-input-output-3')
                    ]),
                    html.Hr(),
                    html.H4("Number of Clusters?", className="parameter-name"),
                    html.Div([
                        daq.NumericInput(
                            id='my-numeric-input-4',
                            min=0,
                            max=20,
                            value=5
                        ),
                        html.Div(id='numeric-input-output-4')
                    ]),
                    html.Hr(),
                    html.Button(id="XPVI_run", children="XPVI Run", className="btn-primary", n_clicks=0)
                ]
            )
        ],
        className="flex-column align-center")
    ])

def decl2NL_parameters():
    data = load_variables()
    segments_count = data["segments_count"]
    clusters_count = data["clusters_count"]
    return html.Div([
        html.Div(className="parameter-container",
            children=[
                html.Div(
                    className="section-header",
                    children=html.H4("Report Generation Parameters", className="section-title"),
                ),
                html.H4("Which segment?", className="parameter-name"),
                dcc.Dropdown(id='segment_number', options=[{'label': x, 'value': x} for x in range(1, segments_count + 1)]),
                html.Hr(),
                html.H4("Which cluster?", className="parameter-name"),
                dcc.Dropdown(id='cluster_number', options=[{'label': x, 'value': x} for x in range(1, clusters_count + 1)]),
                html.Hr(),
                html.Button(id="decl2NL_pars", children="Show decl2NL parameters!", className="btn-primary", n_clicks=0)
            ]
        )
    ])


def PVI_figures(fig_src1, fig_src2):
    return html.Div(
        id="bottom-section",
        className="page-container",
        children=[
            html.Div(
                className="section-header",
                children=html.H4("Process Variant Identification Visualizations", className="section-title"),
            ),
            # html.Img(id="bar-graph-matplotlib", src=fig_src1, className="figure figure-large"),
            # html.Img(id="bar-graph-matplotlib2", src=fig_src2, className="figure figure-small"),
            dcc.Graph(id='heatmap1', figure=fig_src1, className="figure figure-large"),
            dcc.Graph(id='heatmap2', figure=fig_src2, className="figure figure-large"),
            html.Button(id="X_parameters", children="Start The Explainability Extraction Framework!", className="btn-secondary", n_clicks=0)
        ]
    )

def PVI_figures(fig_src1, fig_src2):
    return html.Div(
        id="bottom-section",
        className="page-container",
        children=[
            html.Div(
                className="section-header",
                children=html.H4("Process Variant Identification Visualizations", className="section-title"),
            ),
            # html.Img(id="bar-graph-matplotlib", src=fig_src1, className="figure figure-large"),
            # html.Img(id="bar-graph-matplotlib2", src=fig_src2, className="figure figure-small"),
            dcc.Graph(id='heatmap1', figure=fig_src1, className="figure figure-large"),
            dcc.Graph(id='heatmap2', figure=fig_src2, className="figure figure-large"),
            html.Button(id="X_parameters", children="Start The Explainability Extraction Framework!", className="btn-secondary", n_clicks=0)
        ]
    )


def XPVI_figures(fig_src3, fig_src4):
    return html.Div(
        id="bottom-section",
        className="page-container",
        children=[
            html.Div(
                className="section-header",
                children=html.H4("Explainability Extraction Visualizations", className="section-title"),
            ),
            html.Img(id="bar-graph-matplotlib3", src=fig_src3, className="figure figure-large"),
            html.Img(id="bar-graph-matplotlib4", src=fig_src4, className="figure figure-medium"),
            html.Button(id="decl2NL_framework", children="Convert Declare to Natural Language!", className="btn-secondary", n_clicks=0)
        ]
    )




def statistics_print(list_sorted, list_sorted_reverse):
    return html.Div(
        className="page-container",
        children=[
            html.Div(
                className="section-header",
                children=html.H4("Natual Language Report", className="section-title"),
            ),
            html.H4("Lowest Scores:", className='text-left bg-light mb-4'),
            html.Ul([html.Li(sentence, className="list-item") for sentence in list_sorted]),
            html.H4("Highest Scores:", className='text-left bg-light mb-4'),
            html.Ul([html.Li(sentence, className="list-item") for sentence in list_sorted_reverse])
        ]
    )
