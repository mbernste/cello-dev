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

from graph_lib.graph import DirectedAcyclicGraph, topological_sort
from vis_lib_py3 import vis_lib as vl
import metrics as cm
#import generate_figures as gf



def main():
    usage = "usage: %prog <options> <| delimited results files>, <| delimited method names>"
    parser = OptionParser()
    parser.add_option(
        "-o", 
        "--out_pref", 
        help="Output prefix"
    )
    (options, args) = parser.parse_args()

    # Parse the input
    pr_curve_fs = args[0].split(',')
    method_names = args[1].split(',')
    out_pref = options.out_pref

    # Load the metrics
    pr_curves = []
    for pr_curve_f in pr_curve_fs:
        with open(pr_curve_f, 'r') as f:
            pr_data = json.load(f)
        precisions = pr_data['precisions']
        recalls = pr_data['recalls']
        pr_curves.append((precisions, recalls))

    draw_joint_pr_curves(method_names, pr_curves, out_pref, color_progression=False)

def draw_joint_pr_curves(method_names, pr_curves, out_pref, color_progression=False):
    fig, axarr = plt.subplots(
        1,
        1,
        figsize=(3.0, 3.0),
        squeeze=False
    )
    ax = axarr[0][0]
    for curve_i, pr in enumerate(pr_curves):
        precisions = pr[0]
        recalls = pr[1]
        precisions, recalls = _adjust_pr_curve_for_plot(precisions, recalls)
        if color_progression:
            ax.plot(
                recalls,
                precisions,
                color=sns.color_palette('Blues_r').as_hex()[curve_i],
                lw=1.5
                #where='pre'
            )
        else:
            ax.plot(
                recalls,
                precisions,
                color=vl.NICE_COLORS[curve_i],
                lw=1.5
                #where='pre'
            )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Joint-recall')
    ax.set_ylabel('Joint-precision')
    if color_progression:
        patches = [
            mpatches.Patch(
                color=sns.color_palette('Blues_r').as_hex()[method_i],
                label=method_names[method_i]
            )
            for method_i in range(len(method_names))
        ]
    else:
        patches = [
            mpatches.Patch(
                color=vl.NICE_COLORS[method_i],
                label=method_names[method_i]
            )
            for method_i in range(len(method_names))
        ]
    print('Writing plot to prefix: {}'.format(out_pref))
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        handles=patches
    )
    fig.savefig(
        "%s.pdf" % out_pref,
        format='pdf',
        bbox_inches='tight'
    )
    fig.savefig(
        "%s.eps" % out_pref,
        format='eps',
        bbox_inches='tight'
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
            new_precisions.insert(i+n_inserted, prec)
            new_recalls.insert(i+n_inserted, last_rec)
            n_inserted += 1
    return new_precisions, new_recalls

if __name__ == "__main__":
    main()


