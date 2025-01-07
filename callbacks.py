from dash import Input, Output, State, html
from pathlib import Path
from functions.my_functions import import_log
from functions.EMD_based_framework import apply
from pages.main_page import upload_view


UPLOAD_FOLDER = "event_logs"


def register_callbacks(app):
    @app.callback(
        Output("output-data-upload", "children"),
        [Input("upload-data", "isCompleted")],
        [State("upload-data", "fileNames"), State("upload-data", "upload_id")]
    )
    def display_files(isCompleted, fileNames, uid):
        if not isCompleted:
            return
        if fileNames is not None:
            print(uid)
            out = []
            for filename in fileNames:
                file = Path(UPLOAD_FOLDER) / "temp_log" / filename
                out.append(file)
            max_par,columns = import_log(out[0])
            return upload_view(max_par,columns)
        return html.Ul(html.Li("No Files Uploaded Yet!"))

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
