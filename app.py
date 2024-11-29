from dash import Dash
import dash_bootstrap_components as dbc
import dash_uploader as du
from layout import create_layout
from callbacks import register_callbacks

UPLOAD_FOLDER = "event_logs"  # Define upload folder

# Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
du.configure_upload(app, UPLOAD_FOLDER)

# Set the layout
app.layout = create_layout()

# Register callbacks
register_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=False, port=8002)
