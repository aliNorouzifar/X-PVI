from dash import Input, Output, State, html
from pathlib import Path
from functions.my_functions import import_log
from functions.EMD_based_framework import apply_EMD,apply_segmentation, export_logs
from functions.explainability_extraction import decl2NL, apply_X, apply_feature_extraction, generate_features
from pages.main_page import parameters_view_PVI, parameters_view_explainability, PVI_figures_EMD,PVI_figures_Segments,parameters_feature_extraction, XPVI_figures, decl2NL_parameters, statistics_print, parameters_view_segmentation
import os
import shutil
import json
from functions.utils import save_variables, load_variables
from functions.redis_connection import redis_client

UPLOAD_FOLDER = "event_logs"
# WINDOWS = [15,10,5,2]
WINDOWS = []

def clear_upload_folder(folder_path):
    shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def register_callbacks(app):
    clear_upload_folder("event_logs")
    clear_upload_folder("output_files")

    ''' event log upload and PVI parameters'''
    @app.callback(
        Output("output-data-upload2", "children"),
        [Input("event_log_upload", "isCompleted")],
        [State("event_log_upload", "upload_id")],
    )
    def parameters_PVI(isCompleted,id):
        # redis_client.set('ali', "Hello World")
        if isCompleted==True:
            folder_path = os.path.join(UPLOAD_FOLDER, id)
            files = os.listdir(folder_path) if os.path.exists(folder_path) else []
            file = Path(UPLOAD_FOLDER) / f"{id}" / files[0]
            print(file)
            max_par, columns = import_log(file)
            return parameters_view_PVI(max_par, columns)

    '''significant distance parameter'''
    @app.callback(
        Output("output-data-upload4", "children"),
        [Input("Seg_parameters", "n_clicks")],
    )
    def parameters_segmentation(n):
        # print(redis_client.get('ali'))
        if n>0:
            # data = load_variables("internal_variables")
            max_dist = float(redis_client.get("max_dist"))
            return parameters_view_segmentation(max_dist)

    '''applying the EMD-based process variant identification'''
    @app.callback(
        Output("output-data-upload3", "children"),
        Input("n_bins", "value"),
        Input("w", "value"),
        Input("kpi", "value")
    )
    def plot_data_EMD(n_bin, w, kpi):
        if kpi is not None:
            fig1 = apply_EMD(n_bin, w, kpi, WINDOWS)
            return PVI_figures_EMD(fig1)


    '''pairwise comparison of the segments visualization'''
    @app.callback(
        Output("output-data-upload5", "children"),
        # Input("run_seg", "n_clicks"),
        State("n_bins", "value"),
        State("w", "value"),
        Input("sig_dist", "value"),
    )
    def plot_data_Segments(n_bin, w, sig):
        # if n>0:
            # fig_src1,fig_src2 = PVI_apply(n_bin, w, sig, faster, export, kpi, WINDOWS)
        fig2,peak_explanations = apply_segmentation(n_bin, w, sig, WINDOWS)
        return PVI_figures_Segments(fig2,peak_explanations)

    '''segments_ export'''
    @app.callback(Output("output-data-upload11", "children"),
        Input("export", "n_clicks")
    )
    def export_logs_func(n):
        if n > 0:
            # data = load_variables("segments_ids")
            segments_ids = json.loads(redis_client.get("segments_ids"))
            export_logs(segments_ids)
            return "Event logs are exported!"

    '''Feature space generation (calling Minerful)'''
    @app.callback(
        Output("output-data-upload6", "children"),
        Input("X_parameters", "n_clicks"),
        State("w", "value"),
        State("kpi", "value"),
        State("n_bins", "value"),
    )
    def parameters_explainability(n,w, kpi, n_bin):
        if n > 0:
            generate_features(w, kpi, n_bin)
            return parameters_feature_extraction()

    '''Explainability results visualizations'''
    @app.callback(
        Output("output-data-upload7", "children"),
        # Input("minerful_run", "n_clicks"),
        State("n_bins", "value"),
        State("w", "value"),
        Input("theta_cvg", "value"),
        State("kpi", "value"),
        Input("n_clusters", "value")
        )
    def parameters_explainability(n_bin, w,theta_cvg,kpi,n_clusters):
        # if n > 0:
        apply_feature_extraction(n_bin, w, theta_cvg, kpi)
        fig_src3, fig_src4 = apply_X(n_bin, w, n_clusters)
        return XPVI_figures(fig_src3, fig_src4)



    # @app.callback(
    #     Output("output-data-upload7", "children"),
    #     Input("XPVI_run", "n_clicks"),
    #     State("n_bins", "value"),
    #     State("w", "value"),
    #     State("n_clusters", "value")
    #     )
    # def plot_Xdata(n,n_bin, w, n_clusters):
    #     if n > 0:
    #         fig_src3, fig_src4 = apply_X(n_bin, w, n_clusters)
    #         return XPVI_figures(fig_src3, fig_src4)

    @app.callback(
        Output("output-data-upload10", "children"),
        Input("decl2NL_framework", "n_clicks"),
        )
    def X2NL(n):
        if n > 0:
            return decl2NL_parameters()

    @app.callback(
        Output("output-data-upload9", "children"),
        Input("decl2NL_pars", "n_clicks"),
        State("cluster_number", "value"),
        State("segment_number", "value")
    )
    def X2NL_calc(n,cluster, segment):
        if n > 0:
            list_sorted, list_sorted_reverse = decl2NL(cluster, segment)
            return statistics_print(list_sorted, list_sorted_reverse)
