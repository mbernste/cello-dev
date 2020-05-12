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
        "--out_dir", 
        help="Directory in which to write the output. If it doesn't exist, create the directory."
    )
    (options, args) = parser.parse_args()

    # Parse the input
    metrics_fs = args[0].split(',')
    method_names = args[1].split(',')
    metric_name = args[2]
    x_axis = args[3]
    prefix = args[4]
    out_dir = options.out_dir

    # Load the metrics
    metrics_dfs = []
    for metrics_f in metrics_fs:
        metrics_df = pd.read_csv(metrics_f, sep='\t', index_col=0)
        metrics_dfs.append(metrics_df)

     # Create the output directory

    # Average precision box-plots
    draw_boxplot(
        method_names,
        metrics_dfs,
        metric_name,
        join(out_dir, prefix),
        x_axis=x_axis,
        color_progression=True
    )


def draw_boxplot(
        method_names, 
        metrics_dfs, 
        value_name, 
        out_f_prefix, 
        x_axis='Method',
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
            palette='Blues_r',
            order=method_names
        )
    else: 
        sns.boxplot(
            data=df, 
            x='Method', 
            y=value_name, 
            ax=ax, 
            fliersize=0.0,
            order=method_names
        )
    sns.stripplot(
        data=df, 
        x='Method', 
        y=value_name, 
        ax=ax, 
        color=".3", 
        alpha=0.75, 
        size=2.,
        order=method_names
    )
    for method_name in method_names:
        if len(method_name) > 5:
            plt.xticks(rotation=90)
            break
    ax.set_xlabel(x_axis)
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
     


if __name__ == "__main__":
    main()


