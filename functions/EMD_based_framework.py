import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from pm4py.algo.evaluation.earth_mover_distance import algorithm as emd_evaluator
import scipy.signal as sci_sig
import os
import itertools
import pm4py

import json




# # Constants
# WINDOWS = [15,10,5,2]  # Window sizes for sliding window analysis
OUTPUT_DIR = "output_files"

os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output directory exists
color_theme_drift_map = 'Blues'




def save_variables(df, masks, map_range, peaks):
    df_json = df.to_json(orient="split")
    data = {
        "df": df_json,
        "masks": masks,
        "map_range": map_range,
        "peaks": peaks.tolist()
    }

    # Save to a JSON file
    with open("output_files\internal_variables.json", "w") as json_file:
        json.dump(data, json_file)






def bins_generation(kpi, n_bin):
    """Generate bins and map ranges for a given KPI."""
    case_table = pd.read_csv("output_files/out.csv").sort_values(by=[kpi])

    map_range = {i: case_table[kpi].iloc[round((i / n_bin) * len(case_table[kpi]))] for i in range(n_bin)}
    map_range[n_bin] = case_table[kpi].iloc[-1]

    bin_size = round(len(case_table) / n_bin)
    bins = [
        ((min(case_table[point:point + bin_size][kpi]), max(case_table[point:point + bin_size][kpi])), idx,
         case_table[point:point + bin_size]['trace'])
        if point + bin_size < len(case_table) else
        ((min(case_table[point:][kpi]), max(case_table[point:][kpi])), idx, case_table[point:]['trace'])
        for idx, point in enumerate(range(0, len(case_table), bin_size))
    ]

    return bins, map_range, case_table


def sliding_window(bins, n_bin, sensitivity,WINDOWS):
    """Perform sliding window analysis for change detection."""
    df = pd.DataFrame(0.0, index=WINDOWS, columns=[i for i in range(1, n_bin)])
    for window_size in WINDOWS:
        for mid in range(window_size, n_bin - window_size + 1):
            left = [item for b in bins[mid - window_size:mid] for item in b[2]]
            right = [item for b in bins[mid:mid + window_size] for item in b[2]]
            lang1 = pd.Series(left).value_counts(normalize=True).to_dict()
            lang2 = pd.Series(right).value_counts(normalize=True).to_dict()
            df.at[window_size, mid] = round(emd_evaluator.apply(lang1, lang2), 2)
            # df.loc[window_size][mid] = round(emd_evaluator.apply(lang1, lang2), 2)



    masks = [
        [True] * (window - 1) + [False] * (n_bin - 2 * window + 1) + [True] * (window - 1)
        for window in WINDOWS
    ]
    return df, masks


def segmentation(df,bins,n_bin,w,sen,sig):
    peaks, _ = sci_sig.find_peaks(df.loc[w], height=[sig])
    segments = []
    segments_ids = []
    last_p = -1
    x_state = 0
    for p in peaks:
        new = (x_state, x_state, pd.Series(list(itertools.chain.from_iterable([x[2] for x in bins[last_p + 1:p + 1]]))))
        new_ids = [item for b in bins[last_p + 1:p + 1] for item in b[2].index]
        segments.append(new)
        segments_ids.append(new_ids)
        last_p = p
        x_state += 1

    new = (x_state, x_state, pd.Series(list(itertools.chain.from_iterable([x[2] for x in bins[last_p + 1:]]))))
    new_ids = [item for b in bins[last_p + 1:] for item in b[2].index]
    segments.append(new)
    segments_ids.append(new_ids)

    def emd_dist(bin1, bin2):
        lang1 = bin1[2].value_counts(normalize=True)
        lang1_filt = lang1[lang1 > sen].to_dict()
        lang2 = bin2[2].value_counts(normalize=True)
        lang2_filt = lang2[lang2 > sen].to_dict()
        dist = round(emd_evaluator.apply(lang1_filt, lang2_filt), 2)
        return dist

    state = n_bin
    m_dend = []
    count_bin = []
    cal_list = {}

    itr = 1
    data_points = []

    matrices = []
    labels = []
    mins_vec = []

    state_dic = {}

    while len(segments) > 1:
        dist_matrix = np.ones((len(segments), len(segments)))
        for i in range(0, len(segments)):
            for j in range(i + 1, len(segments)):
                if (segments[i][0], segments[j][0]) not in cal_list:
                    cal_list[(segments[i][0], segments[j][0])] = emd_dist(segments[i], segments[j])
                dist_matrix[i, j] = cal_list[(segments[i][0], segments[j][0])]
                data_points.append((itr, dist_matrix[i, j]))
        matrices.append(dist_matrix)
        labels.append([b[1] for b in segments])
        min_dist_ind = np.unravel_index(np.argmin(dist_matrix, axis=None), dist_matrix.shape)
        mins_vec.append(dist_matrix[min_dist_ind])
        m_dend.append([segments[min_dist_ind[0]][1], segments[min_dist_ind[1]][1], dist_matrix[min_dist_ind],
                       len(segments[min_dist_ind[0]][2]) + len(segments[min_dist_ind[1]][2])])
        state_dic[state] = ((segments[min_dist_ind[0]][1], len(segments[min_dist_ind[0]][2])),
                            (segments[min_dist_ind[1]][1], len(segments[min_dist_ind[1]][2])))
        segments = [segments[k] for k in range(0, len(segments)) if (k != min_dist_ind[0] and k != min_dist_ind[1])] + [
            ((segments[min_dist_ind[0]][0], segments[min_dist_ind[1]][0]), state,
             pd.concat([segments[min_dist_ind[0]][2], segments[min_dist_ind[1]][2]]))]
        state = state + 1
        itr += 1

    new = (x_state, x_state, pd.Series(list(itertools.chain.from_iterable([x[2] for x in bins[last_p:]]))))
    new_ids = [item for b in bins[last_p:] for item in b[2].index]
    segments.append(new)
    segments_ids.append(new_ids)

    # fig = plt.figure(figsize=(9, 8))
    ittr = 0

    # order = [labels[ittr].index(i) for i in list_vec[ittr]]

    new_m = matrices[ittr]
    for i in range(0, len(matrices[ittr])):
        for j in range(i, len(matrices[ittr][i])):
            if i == j:
                new_m[j, i] = 0
            else:
                new_m[j, i] = new_m[i, j]
    return segments, segments_ids, new_m, peaks



def plot_figures(df, masks, n_bin, map_range, dist_matrix, peaks, w,WINDOWS):
    every = 2

    """Generate heatmaps and comparison plots."""
    # Sliding Window Heatmap
    fig1, ax1 = plt.subplots(figsize=(15, 3))
    sns.heatmap(df, cmap="Reds", mask=np.array(masks), ax=ax1)
    ax1.set_xticks(0.5 + np.arange(0, n_bin - 1, 3))
    ax1.set_xticklabels(
        [f"{round(x * (100 / n_bin))}% ({round(map_range[x], 1)})" for x in range(1, n_bin, 3)]
    )
    ax1.set_facecolor("gray")
    ax1.set_title("Sliding Window Analysis")
    ax1.set_xlabel("Traces")
    ax1.set_ylabel("Window Size")
    ax1.set_xticks(0.5 + np.arange(0, n_bin - 1, every))
    ax1.set_xticklabels(
        [str(round(x * (100 / n_bin))) + "% (" + str(round(map_range[x], 1)) + ")" for x in np.arange(1, n_bin, every)])
    ax1.set_yticks([x+0.5 for x in range(0,len(WINDOWS))], labels=WINDOWS)
    plt.xticks(rotation=90)
    plt.close(fig1)

    cmap = plt.cm.Reds
    fig2 = plt.figure(figsize=(7, 7))
    ax = sns.heatmap(dist_matrix, cmap=cmap, xticklabels=['segment' + str(i) for i in range(1, dist_matrix.shape[0] + 1)],
                     yticklabels=['segment' + str(i) for i in range(1, dist_matrix.shape[0] + 1)])

    fig2.suptitle('segments comparison', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    plt.xlabel(' ', fontsize=18)
    cbar.set_label('ldist', fontsize=18)
    plt.close(fig2)

    buf = BytesIO()
    fig1.savefig(buf, format="png", bbox_inches = 'tight')
    # Embed the result in the html output.
    fig_data1 = base64.b64encode(buf.getbuffer()).decode("ascii")

    buf = BytesIO()
    fig2.savefig(buf, format="png", bbox_inches='tight')
    # Embed the result in the html output.
    fig_data2 = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f'data:image/png;base64,{fig_data1}', f'data:image/png;base64,{fig_data2}'


def export_logs(segments_ids, case_table, export_enabled):
    """Export logs for each segment."""
    if not export_enabled:
        return

    event_file = "output_files/out_event.csv"
    event_table = pd.read_csv(event_file)

    os.makedirs("event_logs/exported_logs", exist_ok=True)
    for idx, segment_id_set in enumerate(segments_ids, start=1):
        segment_cases = case_table.loc[segment_id_set, 'case_id']
        segment_log = event_table[event_table['case_id'].isin(segment_cases)]
        segment_log = pm4py.format_dataframe(segment_log, case_id="case_id", activity_key="activity_name",
                                             timestamp_key="timestamp")
        event_log = pm4py.convert_to_event_log(segment_log)
        pm4py.write_xes(event_log, f"event_logs/exported_logs/segment_{idx}.xes")








def apply(n_bin, w, signal_threshold, faster, export, kpi,WINDOWS):
    """Main function to apply the analysis."""
    if w not in WINDOWS:
        WINDOWS.append(w)
        WINDOWS.sort(reverse=True)

    sensitivity = 0.01 if faster else 0.0
    bins, map_range, case_table = bins_generation(kpi, n_bin)
    df, masks = sliding_window(bins, n_bin, sensitivity,WINDOWS)
    segments, segments_ids, dist_matrix, peaks = segmentation(df, bins, n_bin, w, sensitivity, signal_threshold)
    save_variables(df, masks, map_range, peaks, )

    fig1_path, fig2_path = plot_figures(df, masks, n_bin, map_range, dist_matrix, peaks, w,WINDOWS)


    if export:
        export_logs(segments_ids, case_table, export)

    return fig1_path, fig2_path
