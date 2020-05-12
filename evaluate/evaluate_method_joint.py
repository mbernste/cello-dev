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
        "-c", 
        "--conservative_mode", 
        action="store_true", 
        help="Compute conservative metrics"
    )
    (options, args) = parser.parse_args()

    conservative_mode = options.conservative_mode
    out_dir = options.out_dir
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
    results_df = pd.read_csv(results_f, sep='\t', index_col=0)

     # Create the output directory
    _run_cmd("mkdir -p %s" % out_dir)

    # Compute labels on which we will compute metrics
    include_labels = set(results_df.columns) - BLACKLIST_TERMS

    # Create the assignment matrix where rows are samples, columns
    # are labels, and element (i,j) = True if sample i is annotated
    # with label j
    assignment_df = cm._compute_assignment_matrix(
        results_df,
        exp_to_labels
    )
    assignment_df = assignment_df.loc[results_df.index][results_df.columns]

    precisions, recalls, threshs = cm.compute_joint_metrics(
        results_df,
        assignment_df,
        include_labels,
        label_graph=label_graph,
        label_to_name=label_to_name,
        og=og,
        conservative=conservative_mode
    )

    with open(join(out_dir, 'joint_pr_curve.json'), 'w') as f:
        json.dump(
            {
                'precisions': precisions,
                'recalls': recalls,
                'thresholds': threshs
            },
            f,
            indent=4
        )


def _run_cmd(cmd):
    print("Running: %s" % cmd)
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    main()


