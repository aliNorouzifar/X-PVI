from dash import Input, Output, State, html
from pathlib import Path
from functions.my_functions import import_log
from functions.EMD_based_framework import apply as PVI_apply
from functions.explainability_extraction import apply as XPVI_apply
from pages.main_page import parameters_view_PVI, parameters_view_explainability, PVI_figures, XPVI_figures
import os
import shutil

UPLOAD_FOLDER = "event_logs"
WINDOWS = [15,10,5,2]

def clear_upload_folder(folder_path):
    shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def register_callbacks(app):
    clear_upload_folder("event_logs")
    @app.callback(
        Output('output-data-upload', 'children'),
        [Input("upload-data", "isCompleted")],
        [State("upload-data", "fileNames"), State("upload-data", "upload_id")]
    )
    def display_files(isCompleted, filename, uid):
        if isCompleted:
            if filename is not None:
                print(uid)
                return html.Ul(html.Li("Event log is imported!"))
        # else:
        #     return html.Ul(html.Li("No Files Uploaded Yet!"))

    @app.callback(
        Output("output-data-upload2", "children"),
        [Input("output-data-upload", "children")],
        [State("upload-data", "upload_id")],
    )
    def parameters_PVI(ch,id):
        # if storage["status"]=="upload_done":
        folder_path = os.path.join(UPLOAD_FOLDER, id)
        files = os.listdir(folder_path) if os.path.exists(folder_path) else []
        file = Path(UPLOAD_FOLDER) / f"{id}" / files[0]
        print(file)
        max_par, columns = import_log(file)
        # return parameters_view_PVI(max_par, columns), {"status": "parameters_PVI_shown"}
        return parameters_view_PVI(max_par, columns)

    @app.callback(
        Output("output-data-upload3", "children"),
        Input("run_PVI", "n_clicks"),
        State("my-numeric-input-1", "value"),
        State("my-numeric-input-2", "value"),
        State("my-slider3", "value"),
        State("TF", "value"),
        State("TF2", "value"),
        State("xaxis-data", "value")
    )
    def plot_data(n, n_bin, w, sig, faster, export, kpi):
        if kpi is not None:
            fig_src1,fig_src2 = PVI_apply(n_bin, w, sig, faster, export, kpi, WINDOWS)
            return PVI_figures(fig_src1, fig_src2)


    @app.callback(
        Output("output-data-upload4", "children"),
        Input("X_parameters", "n_clicks"),
        )
    def parameters_explainability(n):
        return parameters_view_explainability()

    @app.callback(
        Output("output-data-upload5", "children"),
        Input("XPVI_run", "n_clicks"),
        State("my-numeric-input-1", "value"),
        State("my-numeric-input-2", "value"),
        State("my-numeric-input-3", "value"),
        State("my-numeric-input-4", "value"),
        State("xaxis-data", "value")
        )
    def plot_Xdata(n,n_bin, w, theta_cvg, n_clusters, kpi):
        fig_src3, fig_src4 = XPVI_apply(n_bin, w, theta_cvg, n_clusters, kpi, WINDOWS)
        return XPVI_figures(fig_src3, fig_src4)
