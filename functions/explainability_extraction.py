import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from matplotlib.colors import ListedColormap
import pandas as pd
import pm4py
import shutil
import os
from functions.minerful_calls import mine_minerful_for_declare_constraints
import ruptures as rpt
import csv
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial import KDTree
import json

linkage_method = 'ward'
linkage_metric = 'euclidean'

def load_variables():

    try:
        with open("output_files\internal_variables.json", "r") as json_file:
            data = json.load(json_file)
    except FileNotFoundError:
        return "No data file found."
    df = pd.read_json(data["df"], orient="split")
    return df, data["masks"], data["map_range"], data["peaks"]



def generate_features(w,kpi,n_bin):
    case_table = pd.read_csv("output_files/out.csv").sort_values(by=[kpi])
    ordered_case_ids = case_table['case_id']
    bin_size = round(len(case_table) / n_bin)

    event_table = pd.read_csv("output_files/out_event.csv")
    event_table['case_id'] = pd.Categorical(event_table['case_id'], categories=ordered_case_ids, ordered=True)
    event_table = event_table.sort_values('case_id')
    event_table['case:concept:name'] = event_table['case_id'].astype(str)
    event_table['concept:name'] = event_table['activity_name'].astype(str)
    event_table['time:timestamp'] = pd.to_datetime(event_table['timestamp'])
    log_xes = pm4py.convert_to_event_log(event_table)
    pm4py.write_xes(log_xes, f"output_files/log_ordered.xes")

    window_size = 2 * w * bin_size
    sliding_window_size = bin_size
    mine_minerful_for_declare_constraints(window_size,sliding_window_size)

def find_all_supersets(graph,node, visited):
    for neighbor in graph[node]:
        if neighbor not in visited:
            visited.add(neighbor)
            find_all_supersets(graph,neighbor, visited)
    return visited

def correlation_calc(peaks,w,constraints,clusters_dict):
    n = len(peaks)
    peakmodif = [p-(w-1) for p in peaks]
    segments_sig = {}

    for i in range(n + 1):
        # Start with a list of 100 zeros
        l = [0] * len(constraints[0])

        # For the first list, set elements from 0 to o[0]
        if i == 0:
            l[0:peakmodif[0]] = [1] * (peakmodif[0] - 0)

        # For the last list, set elements from o[n-1] to the end
        elif i == n:
            l[peakmodif[n - 1]:] = [1] * (len(constraints[0]) - peakmodif[n - 1])

        # For other lists, set elements from o[i-1] to o[i]
        else:
            l[peakmodif[i - 1]:peakmodif[i]] = [1] * (peakmodif[i] - peakmodif[i - 1])

        # Append the created list to the list of lists
        segments_sig[f'segment_{i}']=l



    # # Calculate the correlation with each time series in the set

    corr_mat = []
    for seg in segments_sig.keys():
        target_array = np.array(segments_sig[seg])
        average_correlations = []

        for cluster in sorted([i for i in clusters_dict.keys()]):
            correlations = []
            for series in clusters_dict[cluster]:
                # Convert the current series to a NumPy array
                series_array = np.array(series[3:])

                # Calculate the correlation coefficient between target_series and this series
                correlation = np.corrcoef(target_array, series_array)[0, 1]

                # Append the correlation to the list
                correlations.append(correlation)
            average_correlation = np.mean(correlations)
            average_correlations.append(average_correlation)
        corr_mat.append(average_correlations)
    return corr_mat

def sort_by_closest_neigbour_HEADER(data):

    print('There were: ' + str(len(data)) + " values")

    # Convert data to numpy array for efficient operations
    # print(data)
    data = np.array(data, dtype=object)

    # Initialize sorted data with the starting point (first point) and track remaining indices
    new_data = [data[0].tolist()]  # Start with the first point
    index_set = set(range(1, len(data)))  # Skip the first index, it's already in new_data

    # Track the original indices of points in the KDTree
    kd_tree_indices = list(index_set)
    kd_tree_data = data[kd_tree_indices]
    kd_tree = KDTree(kd_tree_data[:, 3:])

    while index_set:
        # Find nearest point in KDTree to the last point in `new_data`
        last_point = new_data[-1][3:]
        _, nearest_in_kd = kd_tree.query(last_point)

        # Map the KDTree index to the original data index
        min_ind = kd_tree_indices[nearest_in_kd]
        new_data.append(data[min_ind].tolist())


        # Update index_set and KDTree
        index_set.remove(min_ind)

        # Rebuild kd_tree for remaining points
        kd_tree_indices = list(index_set)
        kd_tree_data = data[kd_tree_indices]
        kd_tree = KDTree(kd_tree_data[:, 3:])

    return new_data

def import_minerful_constraints_timeseries_data(minerful_constraints_path,constraint_type_used):
    csvfile = open(minerful_constraints_path, 'r')
    csv_reader = csv.reader(csvfile, delimiter=';', quotechar='|')

    hea = next(csv_reader, None)
    hea2 = next(csv_reader, None)

    hea2 = hea2[2:]
    hea = hea[2:]

    header_output = list()

    for i in range(len(hea)):
        if i % 6 == 0:
            tem_h = [hea2[i][1:-1]]
            temp = hea[i]
            if temp[0] == '\'':
                temp = temp[1:]
            if temp[-1] == '\'':
                temp = temp[:-1]
            if temp[-1] == ')':
                temp = temp[:-1]
            # now we split the string
            name_of_constraint_end_index = temp.find('(')
            tem_h.append(temp[:name_of_constraint_end_index])
            temp = temp[name_of_constraint_end_index+1:]
            #find if we have two events or one
            separated_constraints_index = temp.find(', ')
            if not separated_constraints_index == -1:
                tem_h.append(temp[:separated_constraints_index])
                tem_h.append(temp[separated_constraints_index+1:])
            else:
                tem_h.append(temp)
                tem_h.append('')
        else:
            tem_h = [hea2[i][1:-1]] + tem_h[1:]

        header_output.append(tem_h)

    sequences = list()

    # -2 is for the first two columns
    for i in range(len(hea)):
        sequences.append(list())

    corresponding_number_of_traces = []
    n_lines =0
    for r in csv_reader:
        corresponding_number_of_traces.append(r[:2])
        n_lines += 1
        counter = 0
        # print(r[1])
        for i in range(len(r)):
            if counter > 1:
                # print(i)
                sequences[i-2].append(float(r[i]))
            else:
                counter += 1


    constraints = []
    for i, j in zip(sequences, header_output):
        if j[0] == constraint_type_used:
            constraints.append(j[1:] + i)

    return constraints
def clear_upload_folder(folder_path):
    shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def group_signals_by_type_activity(signals):
    # Create a nested defaultdict structure to handle the required dictionary format
    grouped_signals = defaultdict(lambda: defaultdict(set))

    for signal in signals:
        # Extract components
        signal_type = signal[0]  # The type of the signal
        activities = (signal[1], signal[2])  # Tuple of (activity1, activity2)
        signal_samples = tuple(signal[3:])  # Convert samples from position 3 onward to a tuple

        # Populate the dictionary
        if signal_type in {'RespondedExistence'}:
            grouped_signals[signal_samples][(activities[1].lstrip(), f' {activities[0]}')].add(
                'RespondedExistence_r')
            grouped_signals[signal_samples][activities].add(signal_type)
        else:
            grouped_signals[signal_samples][activities].add(signal_type)

    return grouped_signals

def prune_signals(theta_cvg):
    constraints_conf = import_minerful_constraints_timeseries_data(r"output_files\behavioral_signals.csv", 'Confidence')
    constraints_cov = import_minerful_constraints_timeseries_data(r"output_files\behavioral_signals.csv", 'Coverage')
    s_thr = theta_cvg * 100
    main_dim = constraints_conf
    filt_dim = constraints_cov

    not_include_templates = {'NotRespondedExistence',
                             'NotCoExistence',
                             'NotPrecedence',
                             'NotChainPrecedence',
                             'NotResponse',
                             'NotChainResponse',
                             'NotSuccession',
                             'NotChainSuccession'}

    filtered = []
    filtered_colored = []
    for i, c in enumerate(main_dim):
        filtered_sig = []
        filtered_colored_sig = []
        for j, v in enumerate(c):
            if j >= 3:
                if filt_dim[i][j] >= s_thr:
                    filtered_sig.append(v)
                    filtered_colored_sig.append(False)
                else:
                    filtered_sig.append(0)
                    if filt_dim[i][j] > 0:
                        filtered_colored_sig.append(True)
                    else:
                        filtered_colored_sig.append(False)
            else:
                filtered_sig.append(v)
                filtered_colored_sig.append(v)
        filtered.append(filtered_sig)
        filtered_colored.append(filtered_colored_sig)


    data = []
    data_color = []
    not_include = 0
    zeros_removed = 0
    hundreds_removed = 0
    nonchaning_removed = 0


    def all_list_in_interval(L):
        for i in L[1:]:
            if i < 1.05 * L[0] and i > L[0] * 0.95:
                continue
            else:
                return False
        return True

    for ind, j in enumerate(filtered):
        i = j[3:]
        if j[0] in not_include_templates:
            not_include += 1
        elif (mean_squared_error(i, [0] * len(i)) < 10):
            zeros_removed += 1
        elif mean_squared_error(i, [100] * len(i)) < 1:
            print(j[0:3])
            hundreds_removed += 1
        elif all_list_in_interval(i):
            nonchaning_removed += 1
        else:
            data.append(j)
            data_color.append(filtered_colored[ind])

    print('there are : ' + str(len(data)) + " values left after deleting 100, and 0s")
    print("there were: " + str(not_include) + " vectors not included")
    print("there were: " + str(zeros_removed) + " vectors with zeros removed")
    print("there were: " + str(hundreds_removed) + " vectors with hundreds removed")
    print("there were: " + str(nonchaning_removed) + " vectors with non changing values removed")

    data_uncut = data



    # Suppose `signals` is a list of lists or arrays, each with 100 samples
    # Example: signals = [[...100 samples...], [...100 samples...], ...]


    ddd = group_signals_by_type_activity(data_uncut)
    subset_relations = [('RespondedExistence', 'Response'),
                        ('Response', 'AlternateResponse'),
                        ('AlternateResponse', 'ChainResponse'),
                        ('RespondedExistence', 'CoExistence'),
                        ('Response', 'Succession'),
                        ('AlternateResponse', 'AlternateSuccession'),
                        ('ChainResponse', 'ChainSuccession'),
                        ('CoExistence', 'Succession'),
                        ('Succession', 'AlternateSuccession'),
                        ('AlternateSuccession', 'ChainSuccession'),
                        ('RespondedExistence_r', 'Precedence'),
                        ('Precedence', 'AlternatePrecedence'),
                        ('AlternatePrecedence', 'ChainPrecedence'),
                        ('RespondedExistence_r', 'CoExistence'),
                        ('Precedence', 'Succession'),
                        ('AlternatePrecedence', 'AlternateSuccession'),
                        ('ChainPrecedence', 'ChainSuccession'),
                        ('AtLeast1', 'AtLeast2'),
                        ('AtLeast2', 'AtLeast3'),
                        ('AtMost3', 'AtMost2'),
                        ('AtMost2', 'AtMost1'),
                        ('AtMost1', 'Absence'),
                        ('AtLeast1', 'Init'),
                        ('AtLeast1', 'End')]
    subset_relations = [(a[1], a[0]) for a in subset_relations]

    templates = {'RespondedExistence',
                 'RespondedExistence_r',
                 'CoExistence',
                 'Precedence',
                 'AlternatePrecedence',
                 'ChainPrecedence',
                 'Response',
                 'AlternateResponse',
                 'ChainResponse',
                 'Succession',
                 'AlternateSuccession',
                 'ChainSuccession',
                 'NotRespondedExistence',
                 'NotCoExistence',
                 'NotPrecedence',
                 'NotChainPrecedence',
                 'NotResponse',
                 'NotChainResponse',
                 'NotSuccession',
                 'NotChainSuccession',
                 'Absence',
                 'AtLeast1',
                 'AtLeast2',
                 'AtLeast3',
                 'AtMost1',
                 'AtMost2',
                 'AtMost3',
                 'End',
                 'Init'}

    # Create a dictionary to store direct subset relationships
    graph = {letter: set() for letter in templates}

    # Populate the graph based on subset_relations

    print(graph)
    for subset, superset in subset_relations:
        graph[subset].add(superset)

    # print(graph)
    pruned_list = []
    co_exist_list = set()
    for x in ddd.keys():
        for y in ddd[x].keys():
            # print(ddd[x][y])
            # print(ddd[x][y])
            maximal_letters = remove_subsets(ddd[x][y], subset_relations, graph)
            for dcl in maximal_letters - {'RespondedExistence_r'}:
                # print(list(x))
                if dcl == 'CoExistence':
                    if (y[1].lstrip(), f' {y[0]}') not in co_exist_list:
                        pruned_list.append([dcl, y[0], y[1]] + list(x))
                        co_exist_list.add((y[0], y[1]))
                else:
                    pruned_list.append([dcl, y[0], y[1]] + list(x))
            # print(maximal_letters)
    return pruned_list, data_color

def remove_subsets(letters, subset_relations, graph):
    # Find all letters that are subsets of other letters
    subsets_to_remove = set()
    for letter in letters:
        # Find all supersets of the current letter
        supersets = find_all_supersets(graph, letter, set())
        print(supersets)
        # print(supersets)
        subsets_to_remove.update(supersets)

    # The final set of letters excluding any that are subsets of another
    # print(subsets_to_remove)
    maximal_letters = letters - subsets_to_remove

    return maximal_letters

def clustering(pruned_list, linkage_method, linkage_metric, best_n_clusters):


    # delete headers
    data_cut = []
    for data_point in pruned_list:
        data_cut.append(data_point[3:])

    '''build the clustering method'''
    Z = linkage(data_cut, method=linkage_method, metric=linkage_metric)  # metric='correlation'

    # plt.figure(figsize=(10, 7))
    # dendrogram(Z)
    # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('Sample index')
    # plt.ylabel('Distance (linkage)')
    # plt.show()


    a = fcluster(Z, best_n_clusters, 'maxclust')
    clusters_dict = dict()
    for cluster_n, data in zip(a, pruned_list):
        if not cluster_n in clusters_dict:
            clusters_dict[cluster_n] = [data]
        else:
            clusters_dict[cluster_n].append(data)

    constraints = []
    cluster_bounds = []
    count = 0
    clusters_with_declare_names = {}

    order_cluster = [(i, len(clusters_dict[i])) for i in clusters_dict.keys()]
    # order_cluster = sorted(order_cluster, key=lambda x : -x[1])
    order_cluster = sorted(order_cluster, key=lambda x: -x[0])
    order_cluster = [key for key, _ in order_cluster]

    for key in order_cluster:
        # preprocess the clusters for plotting better (sorting them by similarity)
        clusters_dict[key] = sort_by_closest_neigbour_HEADER(clusters_dict[key])

        clusters_with_declare_names[key] = [clusters_dict[key][0][:3]]
        constraints.append(clusters_dict[key][0][3:])
        for i in clusters_dict[key][1:]:
            clusters_with_declare_names[key].append(i[:3])
            constraints.append(i[3:])
        count += len(clusters_dict[key])
        cluster_bounds.append(count)
        print("Cluster size: " + str(len(clusters_dict[key])))

    print('number of clusters: ' + str(len(clusters_dict)))
    print("clusters were ordered in the : ")
    print(order_cluster)

    return order_cluster, clusters_with_declare_names, cluster_bounds, clusters_dict, constraints
def constraints_export(clusters_with_declare_names, peaks, w,clusters_dict):
    file_path = r"output_files/data1.json"

    import statistics
    def generate_natural_language(constraint):
        c, a, b = constraint
        if c == "RespondedExistence":
            description = f"If {a} occurs, {b} occurs as well."
        elif c == "CoExistence":
            description = f"{a} and {b} always occur together."
        elif c == "Response":
            description = f" If {a} occurs, then {b} occurs after it."
        elif c == "AlternateResponse":
            description = f"If {a} occurs, then {b} occurs afterwards before {a} recurs."
        elif c == "ChainResponse":
            description = f"If {a} occurs, then {b} occurs immediately after it."
        elif c == "Precedence":
            description = f"{b} occurs only if preceded by {a}."
        elif c == "AlternatePrecedence":
            description = f"{b} occurs only if preceded by {a} with no other {b} in between."
        elif c == "ChainPrecedence":
            description = f"{b} occurs only if {a} occurs immediately before it. "
        elif c == "Succession":
            description = f"{a} occurs if and only if it is followed by {b}."
        elif c == "AlternateSuccession":
            description = f"{a} and {b} occur if and only if they follow one another, alternating."
        elif c == "ChainSuccession":
            description = f"{a} and {b} occurs if and only if {b} immediately follows {a}."
        elif c == "Init":
            description = f"{a} is the first to occur."
        elif c == "End":
            description = f"{a} is the last to occur."
        elif c == "Absence":
            description = f"{a} must never occur."
        elif c == "AtMost1":
            description = f"{a} occurs at most once."
        elif c == "AtMost2":
            description = f"{a} occurs at most two times."
        elif c == "AtMost3":
            description = f"{a} occurs at most three times."
        elif c == "AtLeast1":
            description = f"{a} occurs at least once."
        elif c == "AtLeast2":
            description = f"{a} occurs at least two times."
        elif c == "AtLeast3":
            description = f"{a} occurs at least three times."
        else:
            description = f"The constraint is un known!"

        return description

    def stat_extract(value, peaks, w):
        # print(value)
        new_list = []
        for const in value:
            # print(const)
            stat_dic = {'constraint type': const[0], 'first parameter': const[1], 'second parameter': const[2],
                        'description': generate_natural_language(const[0:3])}
            c = const[3:]
            seg_num = 1
            # stat_dic = {}
            for p in range(len(peaks) + 1):
                if seg_num == 1:
                    stat_dic[f'segment_{seg_num}'] = round(statistics.mean(c[0:(peaks[p] - (w - 1) + 1)]), 2)
                    stat_dic[f'~segment_{seg_num}'] = round(statistics.mean(c[(peaks[p] - (w - 1)):]), 2)
                elif seg_num <= len(peaks):
                    stat_dic[f'segment_{seg_num}'] = round(
                        statistics.mean(c[(peaks[p - 1] - (w - 1)):(peaks[p] - (w - 1) + 1)]), 2)
                    stat_dic[f'~segment_{seg_num}'] = round(
                        statistics.mean(c[0:(peaks[p - 1] - (w - 1) + 1)] + c[(peaks[p] - (w - 1)):]), 2)
                else:
                    stat_dic[f'segment_{seg_num}'] = round(statistics.mean(c[(peaks[p - 1] - (w - 1)):]), 2)
                    stat_dic[f'~segment_{seg_num}'] = round(statistics.mean(c[0:(peaks[p - 1] - (w - 1) + 1)]), 2)
                stat_dic[f'delta_segment_{seg_num}'] = round(
                    stat_dic[f'segment_{seg_num}'] - stat_dic[f'~segment_{seg_num}'], 2)
                seg_num += 1
            new_list.append(stat_dic)
        return new_list

    # stat_dic = stat_extract(d[3][0][3:], peaks,w)
    # print(stat_dic)

    const = clusters_with_declare_names[1]

    data_str_keys = {str(key): stat_extract(value, peaks, w) for key, value in clusters_dict.items()}
    # Write the dictionary to a JSON file
    with open(file_path, 'w') as file:
        json.dump(data_str_keys, file, indent=4)  # `indent=4` for pretty printing

    print(f"Dictionary successfully saved to {file_path}")

def PELT_change_points(order_cluster,clusters_dict):
    def dingOptimalNumberOfPoints(algo):
        point_detection_penalty = 50
        x_lines = algo.predict(pen=point_detection_penalty)

        while point_detection_penalty >= len(x_lines):
            point_detection_penalty -= 1
            x_lines = algo.predict(pen=point_detection_penalty)

        if len(x_lines) > 15:
            x_lines = x_lines[-1:]
        return x_lines

    horisontal_separation_bounds_by_cluster = {}
    # in ths case we want to detect the drifts in the whole range of constrains at the same time

    dd = []
    for dk in order_cluster:
        for i in clusters_dict[dk]:
            dd.append(i[3:])

    sig = np.array(dd)
    signal = np.transpose(sig)
    # algo = rpt.Pelt(model="rbf", custom_cost=c).fit(signal)
    algo = rpt.Pelt(model="rbf").fit(signal)
    x_lines = dingOptimalNumberOfPoints(algo)
    horisontal_separation_bounds_by_cluster[0] = x_lines
    # pen - penalizing the number of change points

    # print some info
    print('x lines: ')
    print(x_lines)

    return x_lines


def plot_figures(df, masks, n_bin, map_range, peaks, constraints, w, cluster_bounds,clusters_with_declare_names, data_color, corr_mat,WINDOWS):
    every = 2
    color_theme_drift_map = 'Blues'

    def insert_and_clean_np(array, new_number):
        """
        Insert a new number into a sorted NumPy array, maintain the order,
        and remove any numbers within a distance of 1 from the new number.

        Parameters:
        - array (np.ndarray): A sorted NumPy array of numbers.
        - new_number (int or float): The number to insert.

        Returns:
        - np.ndarray: The updated NumPy array.
        """
        if new_number not in array:
            # Find the correct insertion point
            insertion_index = np.searchsorted(array, new_number)

            # Insert the new number into the array
            array = np.insert(array, insertion_index, new_number)

            # Identify elements to keep (distance > 1 from the new number)
            mask = np.abs(array - new_number) != 1

            # Return the updated array
            return array[mask]
            # return mask
        else:
            return array

    ################## Figure 3############################################
    L1 = []
    for k in clusters_with_declare_names.keys():
        for s in clusters_with_declare_names[k]:
            L1.append(s)
    L1_index = {tuple(sublist[0:3]): idx for idx, sublist in enumerate(L1)}

    # Sort and filter L2 based on L1
    L2_ordered = sorted([sublist for sublist in data_color if tuple(sublist[0:3]) in L1_index],
                        key=lambda x: L1_index[tuple(x[0:3])])
    L2_ordered = [x[3:] for x in L2_ordered]
    data_c = []
    data_c_color_1 = []
    data_c_color_2 = []
    for i in range(len(constraints)):
        new_list = [0] * (w - 1) + constraints[i] + [0] * (w - 1)
        new_list_color_1 = [True] * (w - 1) + L2_ordered[i] + [True] * (w - 1)
        new_list_color_2 = [True] * (w - 1) + [False] * len(L2_ordered[i]) + [True] * (w - 1)
        data_c.append(new_list)
        data_c_color_1.append(new_list_color_1)
        data_c_color_2.append(new_list_color_2)
    print('ss')

    # Create a new figure with two subplots with different heights
    fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9),
                                   gridspec_kw={'height_ratios': [10, 1]})  # First plot 3 times bigger

    fig3.suptitle('Control Flow Features and Change Points Overview', fontsize=20)

    # sns.heatmap(data_c, mask=list(mask_ax1[windows.index(w)]), linewidth=0, cmap=color_theme_drift_map,  vmin=0, vmax=100, ax=ax1)
    light_gray_cmap = ListedColormap(['#d3d3d3'])
    sns.heatmap(data_c_color_1, mask=np.array(data_c_color_2), cmap=light_gray_cmap, cbar=False, ax=ax1)
    ax1.set_facecolor("gray")
    sns.heatmap(np.array(data_c), mask=np.array(data_c_color_1), linewidth=0, cmap=color_theme_drift_map, cbar=True,
                vmin=0, vmax=100, ax=ax1)

    cbar1 = ax1.collections[1].colorbar
    cbar1.ax.tick_params(labelsize=16)
    cbar1.set_label('Confidence', fontsize=18)

    original_ticks = cbar1.get_ticks()  # Get the current ticks (e.g., [0, 20, 40, ..., 100])
    normalized_ticks = np.linspace(0, 1, len(original_ticks))  # Map the ticks to 0–1

    # Update the color bar with normalized ticks
    cbar1.set_ticks(original_ticks)  # Retain original tick positions
    cbar1.set_ticklabels([f"{tick:.1f}" for tick in normalized_ticks])  # Set new labels (0–1)

    for bound in cluster_bounds:
        ax1.axhline(y=bound, color='black', linestyle='--', linewidth=2)

    for pp in peaks:
        ax1.axvline(x=0.5 + pp - 0.05, color='red', linestyle='--', linewidth=4)


    ax1.set_xticks([])
    y_tick = []
    y_labels = []
    for i, cb in enumerate(cluster_bounds):
        if i == 0:
            y_tick.append(round(cb / 2, 0))
        else:
            y_tick.append(round((cluster_bounds[i] + cluster_bounds[i - 1]) / 2, 0))
        y_labels.append(f'cluster {len(cluster_bounds) - i}')
    ax1.set_yticks(y_tick)
    ax1.set_yticklabels(y_labels)

    x_tick = []
    x_labels = []
    for i in range(len(peaks) + 1):
        if i == 0:
            x_tick.append(round(peaks[i] / 2, 0))
        elif i == len(peaks):
            x_tick.append(round((n_bin + peaks[i - 1]) / 2, 0))
        else:
            x_tick.append(round((peaks[i] + peaks[i - 1]) / 2, 0))
        x_labels.append(f'segment {i + 1}')
    ax1.set_xticks(x_tick)
    ax1.set_xticklabels(x_labels)
    ax1.tick_params(axis='x', labelsize=16, rotation=0)
    ax1.tick_params(axis='y', labelsize=16, rotation=0)
    ax1.grid(False)


    sns.heatmap(df.loc[w].to_frame().T, mask=masks[WINDOWS.index(w)], cmap="Reds", linewidth=0,
                ax=ax2)

    ticks = np.arange(0, n_bin - 1, 2)
    pk_id = []
    for new_number in peaks:
        ticks = insert_and_clean_np(ticks, new_number)
        pk_id.append(np.where(ticks == new_number)[0][0])
    ticks_labels = [str(round(x * (100 / n_bin))) + "% (" + str(round(map_range[str(x)], 1)) + ")" for x in (ticks + 1)]
    # ticks_labels = [str(round(x * (100 / n_bin))) + "% (" + str(round(map_range[x], 1)) + ")" for x in (ticks + 1)]

    print(pk_id)
    ax2.set_xticks(0.5 + ticks)
    ax2.set_xticklabels(ticks_labels)

    tick_labels = ax2.get_xticklabels()  # Get all x-axis tick labels
    print(tick_labels)
    for pk in pk_id:
        # Change the color of specific ticks
        tick_labels[pk].set_color('red')  # Change tick at position 2 to red

    ax2.set_facecolor("gray")
    ax2.grid(False)
    # ax2.set_title('sliding window analysis', fontsize=20)
    ax2.set_xlabel('traces', fontsize=18)
    ax2.set_ylabel('window size', fontsize=18)
    ax2.tick_params(axis='x', labelsize=16, rotation=90)
    ax2.tick_params(axis='y', labelsize=16)

    # Adjust colorbar font size
    cbar2 = ax2.collections[0].colorbar
    cbar2.ax.tick_params(labelsize=16)
    cbar2.set_label('ldist', fontsize=18)

    ##################### Figure 4####################
    from matplotlib.colors import LinearSegmentedColormap

    corr_mat_transposed = np.array(corr_mat).T  # Transpose the matrix

    # Output the list of correlations
    fig4 = plt.figure(figsize=(8, 7))
    fig4.suptitle('Correlation Between Features and Segments', fontsize=20)

    colors = ['#8B0000', 'white', '#00008B']  # Dark blue, white, dark red
    # Create a colormap
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmp", colors)

    # Create a heatmap with black grid lines
    ax3 = sns.heatmap(
        corr_mat_transposed,
        annot=False,
        cmap=custom_cmap,
        cbar=True,
        vmin=-1,
        vmax=1,
        linewidths=0.5,  # Width of the lines separating the cells
        linecolor='gray'  # Color of the grid lines
    )

    # Swap x and y axis ticks and labels
    ax3.set_xticks(0.5 + np.arange(0, len(peaks) + 1))
    ax3.set_xticklabels(x_labels)  # x_labels now on x-axis (was y-axis)

    ax3.set_yticks(0.5 + np.arange(0, len(cluster_bounds)))
    ax3.set_yticklabels(y_labels[::-1])  # y_labels now on y-axis (was x-axis)

    # Rotate the y-axis tick labels instead of x-axis
    ax3.tick_params(axis='y', rotation=0, labelsize=16)
    ax3.tick_params(axis='x', labelsize=16)

    ax3.invert_yaxis()

    cbar3 = ax3.collections[0].colorbar
    cbar3.ax.tick_params(labelsize=16)
    cbar3.set_label('correlation', fontsize=18)


    buf = BytesIO()
    fig3.savefig(buf, format="png", bbox_inches='tight')
    # Embed the result in the html output.
    fig_data3 = base64.b64encode(buf.getbuffer()).decode("ascii")

    buf = BytesIO()
    fig4.savefig(buf, format="png", bbox_inches='tight')
    # Embed the result in the html output.
    fig_data4 = base64.b64encode(buf.getbuffer()).decode("ascii")



    return f'data:image/png;base64,{fig_data3}', f'data:image/png;base64,{fig_data4}'


def apply(n_bin, w, theta_cvg, n_clusters, kpi, WINDOWS):
    """Main function to apply the analysis."""
    if w not in WINDOWS:
        WINDOWS.append(w)
        WINDOWS.sort(reverse=True)

    ################### Explainability ######################################
    generate_features(w,kpi,n_bin)
    # clear_upload_folder(r"\event_logs")
    pruned_list, data_color = prune_signals(theta_cvg)
    order_cluster, clusters_with_declare_names, cluster_bounds, clusters_dict, constraints = clustering(pruned_list,
                                                                                                        linkage_method,
                                                                                                        linkage_metric,
                                                                                                        n_clusters)
    df, masks, map_range, peaks = load_variables()
    # PELT_change_points(order_cluster, clusters_dict)
    constraints_export(clusters_with_declare_names, peaks, w, clusters_dict)
    corr_mat = correlation_calc(peaks, w, constraints, clusters_dict)
    fig3_path, fig4_path = plot_figures(df, masks, n_bin, map_range, peaks,
                                                              constraints, w, cluster_bounds,
                                                              clusters_with_declare_names, data_color, corr_mat,
                                                              WINDOWS)

    return fig3_path, fig4_path
