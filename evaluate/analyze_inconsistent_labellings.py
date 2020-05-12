###########################################################################################
#
###########################################################################################

import matplotlib as mpl
mpl.use('Agg')
import os
from os.path import join
import json
from optparse import OptionParser
from collections import defaultdict
from matplotlib import pyplot as plt
import pandas as pd
import subprocess

from common import the_ontology
from vis_lib_py3 import vis_lib as vl
import graph_lib
from graph_lib.graph import DirectedAcyclicGraph

BLACKLIST_TERMS = set([
    "CL:0000010"    # cultured cell
    "CL:0000578"    # experimentally modified cell in vitro
    "CL:0001034"    # cell in vitro
    "CL:0000255"    # eukaryotic cell
    "CL:0000548"    # animal cell
])

EPSILON = 0.000000000001
VERY_INCONS_THRESH = 0.5

def main():
    usage = "usage: %prog <options> <environment dir> <experiment list name> <cross-validation config name>"
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_dir", help="Directory in which to write the output")
    (options, args) = parser.parse_args()
   
    results_f = args[0]
    label_graph_f = args[1]
    prefix = args[2]
    out_dir = options.out_dir

    # Load the results
    confidence_df = pd.read_csv(results_f, sep='\t', index_col=0)

    # Map each label to its name
    og = the_ontology.the_ontology()
    label_to_name = {
        label: og.id_to_term[label].name
        for label in confidence_df.columns
    }

    _run_cmd("mkdir -p %s" % out_dir)

    # Load the label-graph
    with open(label_graph_f, 'r') as f:
        labels_data = json.load(f)
    label_graph = DirectedAcyclicGraph(
        labels_data['label_graph']
    )

    # Compute the labels for which we will compute metrics over.
    # This label set is simply the set of labels for which we have
    # predictions for every sample
    include_labels = set(confidence_df.columns) - BLACKLIST_TERMS

    total_n_incons = 0
    total_n_very_incons = 0
    total_edges = 0
    incons_to_count = defaultdict(lambda: 0)
    very_incons_to_count = defaultdict(lambda: 0)
    exp_child_parent_incons = []
    for exp in confidence_df.index:
        exp_n_incons = 0
        for parent_label in confidence_df.columns:
            parent_conf = confidence_df.loc[exp][parent_label]
            if parent_label in BLACKLIST_TERMS:
                continue
            for child_label in label_graph.source_to_targets[parent_label]:
                if child_label in BLACKLIST_TERMS:
                    continue
                if child_label not in confidence_df.columns:
                    continue
                child_conf = confidence_df.loc[exp][child_label]
                # Don't consider parent-child edges where prediction-scores
                # for both nodes is less than 1%
                if child_conf < 0.01 and parent_conf < 0.01:
                    continue
                # We count the edge as inconsistent if BOTH the child's score is 
                # greater than its parents and ALSO that difference is non-negligeble
                if abs(child_conf - parent_conf) > EPSILON and child_conf > parent_conf:
                    exp_child_parent_incons.append((
                        exp, 
                        child_label, 
                        parent_label, 
                        (child_conf-parent_conf)
                    ))
                    incons_to_count[(parent_label, child_label)] += 1
                    total_n_incons += 1
                    exp_n_incons += 1
                    if child_conf - parent_conf > VERY_INCONS_THRESH:
                        total_n_very_incons += 1
                        very_incons_to_count[(parent_label, child_label)] += 1
                total_edges += 1
    total_fraction_inconsistent = total_n_incons / float(total_edges)
    total_fraction_very_inconsistent = total_n_very_incons / float(total_edges)

    print("Inconsistent edges:")
    for incons, count in sorted([(k,v) for k,v in incons_to_count.items()], key=lambda x: x[1]):
        parent = incons[0]
        child = incons[1]
        print("%s -> %s : %d" % (label_to_name[parent], label_to_name[child], count))
    print("Very inconsistent edges:")
    for incons, count in sorted([(k,v) for k,v in very_incons_to_count.items()], key=lambda x: x[1]):
        parent = incons[0]
        child = incons[1]
        print("%s -> %s : %d" % (label_to_name[parent], label_to_name[child], count))

    summary_df = pd.DataFrame(
        data=[
            [
                total_n_incons, 
                total_edges, 
                total_fraction_inconsistent
            ],
            [
                total_n_very_incons, 
                total_edges, 
                total_fraction_very_inconsistent
            ],
            [
                total_n_very_incons, 
                len(confidence_df.index), 
                (float(total_n_very_incons)/len(confidence_df.index))
            ]
        ],
        columns=["No. enconsistent", "Total edges", "Fraction of total edges"],
        index=[
            "Total edges inconsistent", 
            "Total edges inconsistent >%f" % VERY_INCONS_THRESH, 
            "Avg. very inconsistent per sample"
        ]
    )
    summary_df.to_csv(
        join(out_dir, '{}.inconsistent_edges_stats.tsv'.format(prefix)),
        sep='\t'
    )

    exp_child_parent_incons = sorted(exp_child_parent_incons, key=lambda x: x[3])
    inconss = []
    n_less_eq = []
    l_less_than_1 = 0
    n_great_than_1 = 0
    for i, exp_child_parent_icons in enumerate(exp_child_parent_incons):
        incons = exp_child_parent_icons[3]
        inconss.append(incons)
        n_less_eq.append(float(i)/len(exp_child_parent_incons))

    fig, axarr = plt.subplots( 
        1, 
        1, 
        figsize=(3.0, 3.0), 
        squeeze=False 
    ) 
    axarr[0][0].plot(inconss, n_less_eq, color=vl.NICE_COLORS[1], lw=4)
    axarr[0][0].set_xlabel('Child prob. - Parent prob.')
    axarr[0][0].set_ylabel('Cumulative probability')
    axarr[0][0].set_xlim((0.0, 1.0)) 
    axarr[0][0].set_ylim((0.0, 1.0))
    out_f = join(out_dir, "{}.CDF_inconsistences".format(prefix)) 
    fig.savefig( 
        "%s.eps" % out_f, 
        format='eps', 
        bbox_inches='tight', 
        dpi=100, 
        transparent=True 
    )
    fig.savefig(
        "%s.pdf" % out_f,
        format='pdf',
        bbox_inches='tight',
        dpi=100,
        transparent=True
    ) 
    

def _run_cmd(cmd):
    print("Running: %s" % cmd)
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    main()


