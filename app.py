import dash_uploader as du
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
# from layout import create_layout
from callbacks import register_callbacks
import pages.start_page as start_page
import pages.about_me as about_me
import pages.main_page as main_page
import os

UPLOAD_FOLDER = "event_logs"
# List of directories to check/create
directories = ["event_logs", "output_files"]

for directory in directories:
    # Check if directory exists
    if not os.path.exists(directory):
        # Create the directory (and any necessary parent directories)
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP,"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"])
app.title = "Process Variant Identification"
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    # dcc.Store(id='storage', storage_type='memory'),
    dcc.Location(id='url', refresh=False),  # Tracks the URL
    html.Div(id='page-content'),         # Placeholder for page content
])

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/':
        return start_page.layout
    elif pathname == '/main':
        return main_page.create_layout()
        # return create_layout()
    elif pathname == '/about_me':
        return about_me.layout
    else:
        return html.H1("404: Page Not Found", style={"textAlign": "center"})


du.configure_upload(app, UPLOAD_FOLDER)

#
# app.layout = create_layout()

register_callbacks(app)

if __name__ == '__main__':
    app.run_server(host="0.0.0.0", debug=False, port=8002)
