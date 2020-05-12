##############################################################################################
#   Craetes plots for visualizing differences between precision and recall 
#   across terms between two learning algorithms
##############################################################################################

import matplotlib as mpl
mpl.use('Agg')

import os
from os.path import join
import sys
import pandas as pd
import random
import json
from optparse import OptionParser
import collections
from collections import defaultdict
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas
import math
import subprocess
import colour
from colour import Color
import numpy as np
import scipy
from scipy.stats import wilcoxon

from graph_lib.graph import DirectedAcyclicGraph
from common import the_ontology
from vis_lib_py3 import vis_lib as vl
import metrics as cm
import generate_figures as gf


BLACKLIST_TERMS = set([
    "CL:0000010",   # cultured cell
    "CL:0000578",   # experimentally modified cell in vitro
    "CL:0001034",   # cell in vitro
    "CL:0000255",   # eukaryotic cell
    "CL:0000548"    # animal cell
])

def main():
    usage = "usage: %prog <options> <| delimited results files>, <| delimited method names>"
    parser = OptionParser()
    parser.add_option(
        "-o", 
        "--out_dir", 
        help="Directory in which to write the output. If it doesn't exist, create the directory."
    )
    parser.add_option(
        "-f",
        "--config_file",
        help="JSON file with all inputs required to run this analysis"
    )
    parser.add_option(
        "-t",
        "--thresholds",
        help="Either a JSON file mapping each label to a decision threshold or number denoting the threshold to use for all cell types" 
    )
    parser.add_option(
        "-v",
        "--threshold_val",
        help="A number denoting the threshold to use for all cell types"
    )
    parser.add_option(
        "-c", 
        "--conservative_mode", 
        action="store_true", 
        help="Compute conservative metrics"
    )
    (options, args) = parser.parse_args()

    conservative_mode = options.conservative_mode
    if options.threshold_val:
        label_to_thresh = defaultdict(lambda: float(options.threshold_val))
    elif options.thresholds:
        label_to_thresh_df = pd.read_csv(options.thresholds, sep='\t', index_col=0)
        label_to_thresh = {
            label: label_to_thresh_df.loc[label]['threshold']
            for label in label_to_thresh_df.index
        }
    out_dir = options.out_dir
    

    # Parse the input
    if options.config_file:
        config_f = args[0]
        with open(config_f, 'r') as f:
            config = json.load(f)
            label_graph_f = config['label_graph_file']
            labeling_f = config['labeling_file']
            results_fs = config['results_files']
            method_name = config['method_name'] 
    else:
        method_name = args[0]
        results_f = args[1]
        label_graph_f = args[2]     
        labeling_f = args[3]

    # Load the ontology
    og = the_ontology.the_ontology()

    # Load the labels' data
    with open(label_graph_f, 'r') as f:
        label_data = json.load(f)
    label_graph = DirectedAcyclicGraph(label_data['label_graph'])
    label_to_name = {
        x: og.id_to_term[x].name
        for x in label_graph.get_all_nodes()
    }

    # Load the labellings
    with open(labeling_f, 'r') as f:
        labelling = json.load(f)
    exp_to_labels = labelling['labels']

    # Load the results
    bin_results_df = pd.read_csv(results_f, sep='\t', index_col=0)

     # Create the output directory
    _run_cmd("mkdir -p %s" % out_dir)

    # Compute labels on which we will compute metrics
    include_labels = set(bin_results_df.columns) - BLACKLIST_TERMS

    # Create the assignment matrix where rows are samples, columns
    # are labels, and element (i,j) = True if sample i is annotated
    # with label j
    assignment_df = cm._compute_assignment_matrix(
        bin_results_df,
        exp_to_labels
    )
    #bin_results_da = {}
    #for label in results_df.columns:
    #    if options.thresholds and label not in label_to_thresh:
    #        continue
    #    confs = results_df[label] 
    #    bins = [
    #        (x > label_to_thresh[label])
    #        for x in confs
    #    ]
    #    bin_results_da[label] = bins
    #bin_results_df = pd.DataFrame(
    #    data=bin_results_da,
    #    index=results_df.index
    #)
    assignment_df = assignment_df.loc[bin_results_df.index][bin_results_df.columns]

    metrics_df = cm.compute_label_centric_metrics_binary(
        bin_results_df,
        assignment_df,
        include_labels,
        label_graph=label_graph,
        label_to_name=label_to_name,
        og=og,
        conservative=conservative_mode
    )

    metrics_df.to_csv(join(out_dir, 'binary_cell_type_metrics.tsv'), sep='\t')

    label_to_f1 = {
        label: metrics_df.loc[label]['F1-Score']
        for label in metrics_df.index
    }
    print(label_to_f1)

    # F1-score drawn atop ontology
    draw_collapsed_ontology(
        label_graph,
        label_to_name,
        label_to_f1,
        'F1-Score',
        out_dir
    )

    # Average precision box-plots
    #gf.draw_boxplot(
    #    method_name,
    #    metrics_df,
    #    'F1-Score',
    #    join(out_dir, "f1_scores_boxplot")
    #)

    


def draw_boxplot(
        method_names, 
        metrics_dfs, 
        value_name, 
        out_f_prefix, 
        color_progression=False
    ):
    da = []
    print("Plotting %s" % value_name)
    vals = []
    labels = []
    methods = []
    for method_name, metrics_df in zip(method_names, metrics_dfs):
        vals += list(metrics_df[value_name])
        labels += list(metrics_df.index)
        methods += [method_name for i in range(len(metrics_df))]
    df = pandas.DataFrame(
        data={
            'Method': methods,
            'Label': labels,
            value_name: vals
        }
    )
    fig, ax = plt.subplots(
        1,
        1,
        sharey=True,
        figsize=(0.55*len(method_names), 3)
    )
    if color_progression:
        sns.boxplot(
            data=df, 
            x='Method', 
            y=value_name, 
            ax=ax, 
            fliersize=0.0, 
            palette='Blues_r'
        )
    else: 
        sns.boxplot(
            data=df, 
            x='Method', 
            y=value_name, 
            ax=ax, 
            fliersize=0.0
        )
    sns.stripplot(
        data=df, 
        x='Method', 
        y=value_name, 
        ax=ax, 
        color=".3", 
        alpha=0.75, 
        size=2.
    )
    for method_name in method_names:
        if len(method_name) > 5:
            plt.xticks(rotation=90)
            break
    fig.savefig(
        "%s.pdf" % out_f_prefix, 
        format='pdf', 
        dpi=1000, 
        bbox_inches='tight'
    )
    fig.savefig(
        "%s.eps" % out_f_prefix, 
        format='eps', 
        dpi=1000, 
        bbox_inches='tight'
    )
     


def draw_collapsed_ontology(
        label_graph,
        label_to_name,
        label_to_metric,
        metric_name,
        out_dir
    ):

    tmp_dir = join(out_dir, "tmp_figs")
    _run_cmd("mkdir -p %s" % tmp_dir)

    source_to_targets = label_graph.source_to_targets
    target_to_sources = label_graph.target_to_sources

    result_dot_str = _diff_dot(
        source_to_targets,
        label_to_name,
        metric_name,
        label_to_metric
    )
    dot_f = join(tmp_dir, "ontology_graph_%s.dot" % metric_name)
    graph_out_f = join(out_dir, "%s_on_graph.pdf" % metric_name)
    with open(dot_f, 'w') as f:
        f.write(result_dot_str)
    _run_cmd("dot -Tpdf %s -o %s" % (dot_f, graph_out_f))


def _diff_dot(
        source_to_targets,
        node_to_label,
        metric_name,
        node_to_color_intensity
    ):
    max_color_intensity = float(math.log(
        max(node_to_color_intensity.values())+1.0
    ))

    g = "digraph G {\n"
    all_nodes = set(source_to_targets.keys())
    for targets in source_to_targets.values():
        all_nodes.update(targets)

    print(node_to_color_intensity)
    for node in all_nodes:
        if node not in node_to_color_intensity:
            continue
        if max_color_intensity == 0.0:
            intensity = 0.0
        else:
            intensity = math.log(node_to_color_intensity[node]+1.0) / max_color_intensity

        # The algorithm used to set the color's luminosity: 
        #
        #  black        color  target    white
        #   |--------------|-----|---------|     luminosity     
        #  0.0                   |        1.0
        #                        |
        #                        |                    
        #                  |-----|---------|     intensity
        #                 1.0  intensity  0.0   

        ## luminance corresponding to maximum intensity
        #the_color = vl.NICE_BLUE
        #min_luminance = Color(the_color).luminance
        ## difference of luminance we allow between pure white and the target color
        #luminance_range = 1.0 - min_luminance
        #luminance = min_luminance + ((1.0 - intensity) * luminance_range)
        #color_value = Color(the_color, luminance=luminance).hex_l

        cmap = mpl.cm.get_cmap('viridis')
        rgba = cmap(node_to_color_intensity[node])
        color_value = mpl.colors.rgb2hex(rgba)

        if node_to_color_intensity[node] > 0.5:
            font_color = 'black'
        else:
            font_color = 'white'

        g += '"%s\n%s = %f" [style=filled,  fillcolor="%s", fontcolor=%s, fontname = "arial", label="%s\n%s = %f"]\n' % (
            node_to_label[node],
            metric_name,
            node_to_color_intensity[node],
            color_value,
            font_color,
            node_to_label[node],
            metric_name,
            node_to_color_intensity[node]
        )
    for source, targets in source_to_targets.items():
        for target in targets:
            if source in node_to_color_intensity and target in node_to_color_intensity:
                g += '"%s\n%s = %f" -> "%s\n%s = %f"\n' % (
                    node_to_label[source],
                    metric_name,
                    node_to_color_intensity[source],
                    node_to_label[target],
                    metric_name,
                    node_to_color_intensity[target]
                )
    g += "}"
    return g




def _run_cmd(cmd):
    print("Running: %s" % cmd)
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    main()


