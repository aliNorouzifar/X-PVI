from dash import html, dcc

layout = html.Div(
    className="page-container",
    children=[html.Div(className="nav-links",
            children=[
        dcc.Link("Introduction", href="/", style={"margin-right": "20px"}),
        dcc.Link("X-PVI", href="/main", style={"margin-right": "20px"}),
        dcc.Link("About Me", href="/about_me")
        ], style={"textAlign": "center", "margin-bottom": "20px"}),
    html.H1("Welcome to the Process Variant Identification Tool", className='header'),
    html.P("This is the start page of your application. Use the navigation below to get started.", className='content'),
    html.Div([
        dcc.Link("X-PVI", href="/main", style={"margin-right": "20px"})
    ], style={"textAlign": "center", "margin-top": "20px"}),
    html.Div(
    className="page-container",
    children=[
    html.P(
            "If you need an example to try the application follow the instructions below:",
            className="content"
        ),

    html.P(
            "You can download an example event log generated from the BPMN model below using the link provided. This event log serves as a sample file for exploring and testing the features of our tool.",
            className="content"
        ),
    html.Div(
            children=[
                html.A(
                    "Download Sample File",
                    href="/assets/test.xes",
                    download="test.xes",
                    style={
                        "display": "block",
                        "textAlign": "center",
                        "margin-top": "20px",
                        "color": "#ffffff",
                        "background-color": "#4CAF50",
                        "padding": "10px 20px",
                        "text-decoration": "none",
                        "border-radius": "5px"
                    }
                )
            ],
            style={"textAlign": "center"}
        ),
    html.Div(
            children=[
                html.Img(
                    src="/assets/bpmn_test.png",  # Replace with your BPMN model image file
                    alt="BPMN Model Used to Generate Event Log",
                    style={
                        "display": "block",
                        "margin": "20px auto",
                        "max-width": "100%",
                        "height": "auto",
                        "border": "2px solid #4CAF50",
                        "border-radius": "10px"
                    }
                )
            ]
        )]
        )
])