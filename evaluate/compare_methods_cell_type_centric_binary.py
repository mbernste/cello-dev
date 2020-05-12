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
import the_ontology
from map_sra_to_ontology import ontology_graph
import vis_lib
from vis_lib import vis_lib as vl
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
        action="store_true",
        help="Load plotting config from file rather than command line arguments"
    )
    parser.add_option(
        "-c", 
        "--conservative_mode", 
        action="store_true", 
        help="Compute conservative metrics"
    )
    (options, args) = parser.parse_args()

    conservative_mode = options.conservative_mode
    out_dir = options.out_dir

    # Parse the input
    if options.config_file:
        config_f = args[0]
        with open(config_f, 'r') as f:
            config = json.load(f)
            label_graph_f = config['label_graph_file']
            labeling_f = config['labeling_file']
            results_fs = config['results_files']
            method_names = config['method_names'] 
    else:
        method_names = args[0].split(',')
        result_fs = args[1].split(',')
        label_graph_f = args[0]     

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
    all_results = []
    for results_f in results_fs:
        all_results.append(
            pd.read_csv(results_f, sep='\t', index_col=0)
        )
    assert _comparable_results(all_results)

     # Create the output directory
    _run_cmd("mkdir -p %s" % out_dir)

    # Compute labels on which we will compute metrics
    include_labels = set(all_results[0].columns) - BLACKLIST_TERMS

    # Create the assignment matrix where rows are samples, columns
    # are labels, and element (i,j) = True if sample i is annotated
    # with label j
    assignment_df = cm._compute_assignment_matrix(
        all_results[0],
        exp_to_labels
    )

    metrics_dfs = []
    label_to_pr_curves = []
    for results_df in all_results:
        results_df = results_df.loc[assignment_df.index][assignment_df.columns]
        metrics_df = cm.compute_label_centric_metrics_binary(
            results_df,
            assignment_df,
            include_labels
        )
        metrics_dfs.append(metrics_df)

    # F1-score barplots overlaid on label-graph
    #draw_collapsed_ontology_w_figures(
    #    exp_to_labels,
    #    label_graph,
    #    label_to_name,
    #    label_to_pr_curves,
    #    [
    #        {
    #            label: metric_df.loc[label]['Avg. Precision']
    #            for label in metric_df.index
    #        }
    #        for metric_df in metrics_dfs
    #    ],
    #    method_names,
    #    out_dir
    #)

    # Average precision box-plots
    gf.draw_boxplot(
        method_names,
        metrics_dfs,
        'F1-Score',
        join(out_dir, "f1_scores_boxplot")
    )

    

def _comparable_results(all_results):
    """
    Make sure all result-matrices share the same test-samples
    and labels.
    """
    ref_exps = frozenset(all_results[0].index)
    ref_labels = frozenset(all_results[0].columns)
    for df in all_results:
        if frozenset(df.index) != ref_exps:
            return False
        if frozenset(df.columns) != ref_labels:
            print(set(df.columns) - ref_labels)
            return False
    return True


def _adjust_pr_curve_for_plot(precisions, recalls):
    new_precisions = [x for x in precisions]
    new_recalls = [x for x in recalls]
    prec_recs = zip(precisions, recalls)
    n_inserted = 0
    for i in range(1,len(prec_recs)):
        prec = prec_recs[i][0]
        rec = prec_recs[i][1]
        last_prec = prec_recs[i-1][0]
        last_rec = prec_recs[i-1][1]
        if rec > last_rec and prec < last_prec:
            #print "Found: (%f,%f) --> (%f,%f). Inserting: (%f, %f) at %d" % (last_rec, last_prec, rec, prec, last_rec, prec) 
            new_precisions.insert(i+n_inserted, prec)
            new_recalls.insert(i+n_inserted, last_rec)
            n_inserted += 1
    return new_precisions, new_recalls



def draw_boxplot(
        method_names, 
        metrics_dfs, 
        value_name, 
        out_f_prefix, 
        color_progression=False
    ):
    da = []
    print "Plotting %s" % value_name
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
     


def draw_comparison_heatmap(method_names, method_to_metrics, metric_name, compute_labels, out_f_prefix):
    print "Generating comparison matrix..."
    fig, ax = plt.subplots(
        1,
        1,
        sharey=True,
        figsize=(2.75, 2.75)
    )
    matrix = []
    annot = []
    all_vals = []
    stat_sig_mask = []
    diag_mask = []
    for method_i in method_names:
        row = []
        annot_row = []
        stat_sig_mask_row = []
        diag_mask_row = []
        for method_j in method_names:
            if method_i == method_j:
                n_best = _compute_num_labels_best(
                    method_i, 
                    method_names, 
                    method_to_metrics, 
                    compute_labels, 
                    metric_name
                )
                row.append(0)
                diag_mask_row.append(False)
                stat_sig_mask_row.append(False)
                annot_row.append("%d" % n_best)
            else:
                win_diff, p_val = _compute_win_diff(
                    method_i,
                    method_j,
                    method_to_metrics,
                    compute_labels,
                    metric_name
                )
                print '%s vs. %s win diff: %d' % (method_i, method_j, win_diff)
                row.append(win_diff)
                all_vals.append(win_diff)
                if p_val < 0.05:
                    stat_sig_mask_row.append(False)
                else:
                    stat_sig_mask_row.append(True)
                diag_mask_row.append(True)
                annot_row.append("%d" % win_diff)
        matrix.append(row)
        annot.append(annot_row)
        stat_sig_mask.append(stat_sig_mask_row)
        diag_mask.append(diag_mask_row)
    with sns.axes_style("white"):
        cmap = sns.diverging_palette(0,255,sep=30, as_cmap=True)
        print "The matrix is: %s" % matrix
        ax = sns.heatmap(
            matrix, 
            mask=np.array(stat_sig_mask), 
            vmin=-max(all_vals), 
            vmax=max(all_vals), 
            cbar=False, 
            cmap=cmap, 
            annot=np.array(annot),
            fmt='',
            yticklabels=method_names,
            xticklabels=method_names,
            annot_kws={"weight": "bold", "fontsize":9}
        )
        ax = sns.heatmap(
            matrix,
            mask=np.array([
                [not x for x in row]
                for row in stat_sig_mask
            ]),
            vmin=-max(all_vals),
            vmax=max(all_vals),
            cbar=False,
            cmap=cmap,
            annot=np.array(annot),
            fmt='',
            yticklabels=method_names,
            xticklabels=method_names,
            annot_kws={"fontsize": "7"}
        )
        dark_middle = sns.diverging_palette(255, 133, l=60, n=7, center="dark")
        ax = sns.heatmap(
            matrix,
            mask=np.array(diag_mask),
            vmin=-max(all_vals),
            vmax=max(all_vals),
            cbar=False,
            cmap=dark_middle,
            annot=np.array(annot),
            fmt='',
            yticklabels=method_names,
            xticklabels=method_names
        )
    fig.savefig("%s.pdf" % out_f_prefix, format='pdf', dpi=1000, bbox_inches='tight')
    fig.savefig("%s.eps" % out_f_prefix, format='eps', dpi=1000, bbox_inches='tight') 
    print "done."


def _compute_num_labels_best(
        targ_method, 
        method_names, 
        method_to_metrics, 
        compute_labels, 
        metric_name
    ):
    n_best = 0
    for label in compute_labels:
        val = method_to_metrics[targ_method].loc[label][metric_name]
        max_other_vals = max([
            method_to_metrics[method].loc[label][metric_name]
            for method in method_names
            if method != targ_method
        ])
        if val > max_other_vals:
            n_best += 1
    return n_best


def _compute_win_diff(
        method_i,
        method_j,
        method_to_metrics,
        compute_labels,
        metric_name
    ):
    diffs = []
    for label in compute_labels:
        i_val = method_to_metrics[method_i].loc[label][metric_name]
        j_val = method_to_metrics[method_j].loc[label][metric_name]
        diffs.append(i_val - j_val)

    n_i_beat_j = len([
        diff 
        for diff in diffs 
        if diff > 0 
        and abs(diff) > 0.05
    ])
    n_j_beat_i = len([
        diff 
        for diff in diffs 
        if diff < 0 
        and abs(diff) > 0.05
    ])
    win_diff = n_i_beat_j - n_j_beat_i
    p_val = wilcoxon(diffs)[1]
    return win_diff, p_val




def draw_collapsed_ontology_w_pr_curves(
        exp_to_labels,
        label_graph, 
        label_to_name,
        label_to_pr_curves, 
        label_to_avg_precisions,
        method_names,
        out_dir
    ):
    tmp_dir = join(out_dir, "tmp_figs")
    _run_cmd("mkdir -p %s" % tmp_dir)

    source_to_targets = label_graph.source_to_targets
    target_to_sources = label_graph.target_to_sources

    method_to_color = {
        method_name: vl.NICE_COLORS[method_i]
        for method_i, method_name in enumerate(method_names)
    }

    label_to_color = {}
    label_to_diff_avg_prec = {}
    label_to_fig = {}
    for label in label_to_pr_curves[0]:
        avg_precs = [
            label_to_avg_precision[label]
            for label_to_avg_precision in label_to_avg_precisions
        ]
        method_w_avg_precs = sorted(
            zip(method_names, avg_precs), 
            key=lambda x: x[1]
        )
       
        label_to_color[label] = method_to_color[method_w_avg_precs[-1][0]]
        print "Label %s" % label_to_name[label]
        print "Best method is %s with avg. precision of %f" % (
            method_w_avg_precs[-1][0], 
            method_w_avg_precs[-1][1]
        )
        if len(method_w_avg_precs) > 1:
            print "Second best method is %s with avg. precision %f" % (
                method_w_avg_precs[-2][0],
                method_w_avg_precs[-2][1]
            )
            label_to_diff_avg_prec[label] = abs(
                method_w_avg_precs[-1][1] - method_w_avg_precs[-2][1]
            )
        else:
            label_to_diff_avg_prec[label] = 0.0

        fig, axarr = plt.subplots(
            1,
            1,
            figsize=(3.0, 3.0),
            squeeze=False
        )
        for curve_i, label_to_pr_curve in enumerate(label_to_pr_curves):
            pr = label_to_pr_curve[label]
            precisions = pr[0]
            recalls = pr[1]
            precisions, recalls = _adjust_pr_curve_for_plot(
                precisions, recalls
            )
            axarr[0][0].plot(
                recalls,
                precisions,
                color='black',
                lw=8.5
            )
            axarr[0][0].plot(
                recalls, 
                precisions, 
                color=vl.NICE_COLORS[curve_i], 
                lw=8
            )
        title = label_to_name[label]
        if len(title) > 25:
            toks = title.split(" ")
            str_1 = ""
            str_2 = ""
            t_i = 0
            for t_i in range(len(toks)):
                if len(str_1) > 25:
                    break
                str_1 += " " + toks[t_i]
            for t_i in range(t_i, len(toks)):
                str_2 += " " + toks[t_i]
            title = "%s\n%s" % (str_1, str_2)

        axarr[0][0].set_title(title, fontsize=26)
        axarr[0][0].set_xlim(0.0, 1.0)
        axarr[0][0].set_ylim(0.0, 1.0)
        out_f = join(tmp_dir, "%s.png" % label_to_name[label].replace(' ', '_').replace('/', '_'))
        fig.savefig(
            out_f, 
            format='png', 
            bbox_inches='tight', 
            dpi=200,
            transparent=True
        )
        label_to_fig[label] = out_f
    result_dot_str = _diff_dot(
        source_to_targets,
        label_to_name, 
        label_to_fig,
        label_to_color,
        label_to_diff_avg_prec 
    )
    dot_f = join(tmp_dir, "collapsed_ontology_by_name.dot")
    graph_out_f = join(out_dir, "pr_curves_on_graph.pdf")
    with open(dot_f, 'w') as f:
        f.write(result_dot_str)
    _run_cmd("dot -Tpdf %s -o %s" % (dot_f, graph_out_f))


def _diff_dot(
        source_to_targets, 
        node_to_label, 
        node_to_image, 
        node_to_color, 
        node_to_color_intensity 
    ):
    max_color_intensity = float(math.log(max(node_to_color_intensity.values())+1.0))

    g = "digraph G {\n"
    all_nodes = set(source_to_targets.keys())
    for targets in source_to_targets.values():
        all_nodes.update(targets)
    for node in all_nodes:
        if node not in node_to_image:
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
        
        # luminance corresponding to maximum intensity
        min_luminance = Color(node_to_color[node]).luminance 
        # difference of luminance we allow between pure white and the target color
        luminance_range = 1.0 - min_luminance 
        luminance = min_luminance + ((1.0 - intensity) * luminance_range)
        color_value = Color(node_to_color[node], luminance=luminance).hex_l
 
        g += '"%s" [style=filled,  fillcolor="%s", image="%s", label=""]\n' % (
            node_to_label[node],
            color_value,
            node_to_image[node]
        )
    for source, targets in source_to_targets.iteritems():
        for target in targets:
            if source in node_to_image and target in node_to_image:
                g += '"%s" -> "%s"\n' % (
                    node_to_label[source],
                    node_to_label[target]
                )
    g += "}"
    return g


def _run_cmd(cmd):
    print "Running: %s" % cmd
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    main()


