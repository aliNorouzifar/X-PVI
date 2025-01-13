
from dash import html, dcc
import dash_bootstrap_components as dbc
import dash_uploader as du
import dash_daq as daq



def create_layout():
    return dbc.Container(className="page-container",
    children=[html.Div(className="nav-links",
            children=[
                dcc.Link("Introduction", href="/", style={"margin-right": "20px"}),
                dcc.Link("X-PVI", href="/main", style={"margin-right": "20px"}),
                dcc.Link("About Me", href="/about_me")
        ], style={"textAlign": "center", "margin-bottom": "20px"}),
        html.H1("Process Variant Identification", className='header',
                style={'textAlign': 'center'}),
        html.H4("Upload the event log", className='text-left bg-light mb-4', style={'textAlign': 'left'}),
        # html.Div([
            html.Div(
                children=[
                    get_upload_component("upload-data")
                ],
                style={  # wrapper div style
                    "textAlign": "center",
                    "width": "100%",
                    "padding": "10px",
                    "display": "inline-block",
                },
            ),
            html.Div(id='output-data-upload'),
            html.Div(id='output-data-upload2'),
            html.Div(id='output-data-upload3'),
            html.Div(id='output-data-upload4'),
            html.Div(id='output-data-upload5'),
            html.Div(id='output-data-upload6'),
            html.Div(id='output-data-upload7'),
    ])

def get_upload_component(id):
    return dbc.Container(className="page-container",
    children= du.Upload(
        id=id,
        max_file_size=500,
        chunk_size=100,
        max_files=1,
        filetypes=["xes"],
        upload_id="event_log",
    ),
    )


def parameters_view_PVI(max_par, columns):
    return html.Div([
        # Wrapper for all content
        html.Div([
            # Top Section: Inputs
            html.Div(className="page-container",
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
                    html.Button(id="run_PVI", children="Run PVI", n_clicks=0),
                    html.Hr(),
                ],
                style={"width": "100%", "padding": "10px"}
            ),
        ],
            style={"display": "flex", "flexDirection": "column", "alignItems": "center"}
        )
    ])

def PVI_figures(fig_src1, fig_src2):
    return html.Div(className="page-container",
             id="bottom-section",
             children=[
                 html.Img(id="bar-graph-matplotlib", src=fig_src1, style={"width": "100%", "margin-bottom": "20px"}),
                 html.Img(id="bar-graph-matplotlib2", src=fig_src2, style={"width": "40%", "margin-bottom": "20px"}),
                 html.Button(id="X_parameters", children="Start The Explainability Extraction Framework!", n_clicks=0)
             ]
             , style={"display": "flex", "flexDirection": "column", "alignItems": "center", "width": "100%", "padding": "10px"})

def parameters_view_explainability():
    return html.Div([
        # Wrapper for all content
        html.Div([
            # Top Section: Inputs
            html.Div(className="page-container",
                id="top-section",
                children=[
                    html.Hr(),
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
                html.Button(id="XPVI_run", children="XPVI Run", n_clicks=0)],
                style={"width": "100%", "padding": "10px"}
            ),
            # Bottom Section: Outputs
        ],
            style={"display": "flex", "flexDirection": "column", "alignItems": "center"}
        )
    ])

def XPVI_figures(fig_src3, fig_src4):
    return html.Div(className="page-container",
             id="bottom-section",
             children=[
                 html.Img(id="bar-graph-matplotlib", src=fig_src3, style={"width": "100%", "margin-bottom": "20px"}),
                 html.Img(id="bar-graph-matplotlib2", src=fig_src4, style={"width": "40%", "margin-bottom": "20px"}),
                 html.Button(id="decl2NL_framework", children="Convert Declare to Natural Language!", n_clicks=0)
             ]
             , style={"display": "flex", "flexDirection": "column", "alignItems": "center", "width": "100%", "padding": "10px"})


def decl2NL_parameters(segments_count, clusters_count):
    return html.Div([
            html.Div(className="page-container",
                children=[
    html.H4("Which segment?", className='text-left bg-light mb-4', style={'textAlign': 'left'}),
    dcc.Dropdown(id='segment_number', options=[{'label': x, 'value': x} for x in range(1,segments_count+1)]),
    html.H4("Which cluster?", className='text-left bg-light mb-4', style={'textAlign': 'left'}),
    dcc.Dropdown(id='cluster_number', options=[{'label': x, 'value': x} for x in range(1,clusters_count+1)]),
    html.Button(id="decl2NL_pars", children="Show decl2NL parameters!", n_clicks=0)
                ])
    ])

def statistics_print(list_sorted, list_sorted_reverse):
    return html.Div(className="page-container",
                children=[
        html.H4(f"{list_sorted}", className='text-left bg-light mb-4', style={'textAlign': 'left'}),
        html.H4(f"{list_sorted_reverse}", className='text-left bg-light mb-4', style={'textAlign': 'left'})])