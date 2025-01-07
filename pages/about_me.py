from dash import html, dcc

layout = html.Div(className="page-container",
    children=[html.Div(className="nav-links",
            children=[
                dcc.Link("Introduction", href="/", style={"margin-right": "20px"}),
                dcc.Link("X-PVI", href="/main", style={"margin-right": "20px"}),
                dcc.Link("About Me", href="/about_me")
        ], style={"textAlign": "center", "margin-bottom": "20px"}),
    html.H1("About Me", className='header'),
    html.P("This application was created to analyze event logs and identify process variants.", className='content'),
    html.P("Feel free to reach out with any questions or feedback.", className='content')
])