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

#import project_data_on_ontology as pdoo
from graph_lib.graph import DirectedAcyclicGraph
from common import the_ontology
from vis_lib_py3 import vis_lib as vl
import metrics as cm


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
        "--out_file", 
        help="File in which to write output"
    )
    (options, args) = parser.parse_args()

    out_f = options.out_file

    pr_curve_f = args[0]
    label_graph_f = args[1]
    labeling_f = args[2]     

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

    # Load PR-curves
    with open(pr_curve_f, 'r') as f:
        label_to_pr_curve = json.load(f)

    # Compute labels on which we will compute metrics
    include_labels = set(label_to_pr_curve.keys()) - BLACKLIST_TERMS

    # Precision recall curves overlaid on label-graph
    draw_collapsed_ontology_w_pr_curves(
        exp_to_labels,
        label_graph,
        label_to_name,
        label_to_pr_curve,
        out_f
    )


def _adjust_pr_curve_for_plot(precisions, recalls):
    new_precisions = [x for x in precisions]
    new_recalls = [x for x in recalls]
    prec_recs = [x for x in zip(precisions, recalls)]
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


def draw_collapsed_ontology_w_pr_curves(
        exp_to_labels,
        label_graph, 
        label_to_name,
        label_to_pr_curve, 
        out_f
    ):
    tmp_dir = "tmp_figs"
    _run_cmd("mkdir -p %s" % tmp_dir)
    source_to_targets = label_graph.source_to_targets
    target_to_sources = label_graph.target_to_sources
    label_to_fig = {}
    for label in label_to_pr_curve:
        fig, axarr = plt.subplots(
            1,
            1,
            figsize=(3.0, 3.0),
            squeeze=False
        )
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
            color=vl.NICE_BLUE, 
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
        curr_out_f = join(tmp_dir, "%s.png" % label_to_name[label].replace(' ', '_').replace('/', '_'))
        fig.savefig(
            curr_out_f, 
            format='png', 
            bbox_inches='tight', 
            dpi=200,
            transparent=True
        )
        plt.close()
        label_to_fig[label] = curr_out_f
    result_dot_str = _diff_dot(
        source_to_targets,
        label_to_name, 
        label_to_fig
    )
    dot_f = join(tmp_dir, "collapsed_ontology_by_name.dot")
    with open(dot_f, 'w') as f:
        f.write(result_dot_str)
    _run_cmd("dot -Tpdf %s -o %s" % (dot_f, out_f))


def _diff_dot(
        source_to_targets, 
        node_to_label, 
        node_to_image 
    ):
    g = "digraph G {\n"
    all_nodes = set(source_to_targets.keys())
    for targets in source_to_targets.values():
        all_nodes.update(targets)
    for node in all_nodes:
        if node not in node_to_image:
            continue
        g += '"%s" [style=filled,  fillcolor="white", image="%s", label=""]\n' % (
            node_to_label[node],
            node_to_image[node]
        )
    for source, targets in source_to_targets.items():
        for target in targets:
            if source in node_to_image and target in node_to_image:
                g += '"%s" -> "%s"\n' % (
                    node_to_label[source],
                    node_to_label[target]
                )
    g += "}"
    return g


def _run_cmd(cmd):
    print("Running: %s" % cmd)
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    main()


