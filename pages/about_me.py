from dash import html, dcc

layout = html.Div(className="page-container",
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
                                  "A cutting-edge tool for process analysis and variant detection.",
                                  className="tool-subtitle",
                              ),
                          ],
                      ),
    html.H1("About Me", className='header'),
    html.P("This application was created to analyze event logs and identify process variants.", className='content'),
    html.P("Feel free to reach out with any questions or feedback.", className='content')
])