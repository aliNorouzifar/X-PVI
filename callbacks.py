from dash import Input, Output, State, html
from pathlib import Path
from functions.my_functions import import_log
from functions.EMD_based_framework import apply
from pages.main_page import upload_view
import os
import shutil

UPLOAD_FOLDER = "event_logs"


def clear_upload_folder(folder_path):
    shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def register_callbacks(app):
    clear_upload_folder("event_logs")
    @app.callback(
        Output("output-data-upload", "children"),
        [Input("upload-data", "isCompleted")],
        [State("upload-data", "fileNames"), State("upload-data", "upload_id")]
    )
    def display_files(isCompleted, filename, uid):
        if isCompleted:
            if filename is not None:
                print(uid)
                return html.Ul(html.Li(f"The file {filename[0]} is uploaded successfully!"))
        else:
            return html.Ul(html.Li("No Files Uploaded Yet!"))

    @app.callback(
        Output("output-data-upload2", "children"),
        [Input("output-data-upload", "children")],
        [State("upload-data", "upload_id")],
    )
    def parameters_PVI(filename,id):
        folder_path = os.path.join(UPLOAD_FOLDER, id)
        files = os.listdir(folder_path) if os.path.exists(folder_path) else []
        file = Path(UPLOAD_FOLDER) / f"{id}" / files[0]
        print(file)
        max_par, columns = import_log(file)
        return upload_view(max_par, columns)

    @app.callback(
        Output("bar-graph-matplotlib", "src"),
        Output("bar-graph-matplotlib2", "src"),
        Output("bar-graph-matplotlib3", "src"),
        Output("bar-graph-matplotlib4", "src"),
        Input("submit-button", "n_clicks"),
        State("my-numeric-input-1", "value"),
        State("my-numeric-input-2", "value"),
        State("my-numeric-input-3", "value"),
        State("my-numeric-input-4", "value"),
        State("my-slider3", "value"),
        State("TF", "value"),
        State("TF2", "value"),
        State("xaxis-data", "value")
    )
    def plot_data(n, n_bin, w, theta_cvg, n_clusters, sig, faster, export, x_data):
        if x_data is not None:
            fig_src1,fig_src2,fig_src3,fig_src4 = apply(n_bin, w, theta_cvg, n_clusters, sig, faster, export, x_data)
            return fig_src1, fig_src2, fig_src3, fig_src4  # Replace with actual figures
        return "", "", "", ""
