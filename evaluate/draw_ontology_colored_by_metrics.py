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

LABEL_NAME_TO_SUCCINCT = {
    'somatic cell': 'somatic',
    'precursor cell': 'precursor',
    'hematopoietic cell': 'hematopoietic',
    'motile cell': 'motile',
    'hematopoietic precursor cell': 'hematopoietic\nprecursor',
    'myeloid cell': 'myeloid',
    'leukocyte': 'leukocyte',
    'nucleate cell': 'nucleated',
    'myeloid leukocyte': 'myeloid\nleukocyte',
    'monocyte': 'monocyte',
    'defensive cell': 'defensive cell',
    'nongranular leukocyte': 'nongranular\nleukocyte',
    'single nucleate cell': 'single\nnucleated',
    'mononuclear cell': 'mononuclear',
    'lymphocyte': 'lymphocyte',
    'CD14-positive monocyte': 'CD14+\nmonocyte',
    'professional antigen presenting cell': 'professional antigen\npresenting cell',
    'lymphocyte of B lineage': 'lymphocyte of\nB lineage',
    'T cell': 'T cell',
    'innate lymphoid cell': 'innate lymphoid\ncell',
    'B cell': 'B cell',
    'mature T cell': 'mature T cell',
    'alpha-beta T cell': 'alpha-beta\nT cell',
    'group 1 innate lymphoid cell': 'group 1 innate\nlymphoid cell',
    'regulatory T cell': 'regulatory\nT cell',
    'effector T cell': 'effector\nT cell',
    'mature alpha-beta T cell': 'mature alpha-beta\nT cell',
    'natural killer cell': 'NK cell',
    'helper T cell': 'helper T cell',
    'CD4-positive, alpha-beta T cell': 'CD4+ T cell',
    'memory T cell': 'memory\nT cell',
    'naive T cell': 'naive T cell',
    'CD8-positive, alpha-beta T cell': 'CD8+ T cell',
    'CD4-positive, CD25-positive, alpha-beta regulatory T cell': 'CD4+CD25+\nregulatory T cell',
    'CD4-positive helper T cell': 'CD4+ helper\nT cell',
    'CD4-positive, alpha-beta memory T cell': 'CD4+ memory\nT cell',
    'naive thymus-derived CD4-positive, alpha-beta T cell': 'CD4+ naive\nT cell',
    'naive thymus-derived CD8-positive, alpha-beta T cell': 'CD8+ naive\nT cell'
}

def main():
    usage = "usage: %prog <options> <| delimited results files>, <| delimited method names>"
    parser = OptionParser()
    parser.add_option(
        "-o", 
        "--out_file", 
        help="Output file"
    )
    (options, args) = parser.parse_args()

    # Parse the input
    metrics_f = args[0]
    metric_name = args[1]
    label_graph_f = args[2]     
    out_f = options.out_file

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

    # Topologically sort the labels and assign them numbers
    topo_sort_labels = topological_sort(label_graph)
    label_to_topo_index = {
        label: index
        for index, label in enumerate(topo_sort_labels)
    }

    # Create text legend for graph
    #legend = ''
    #for label, topo_index in label_to_topo_index.items():
    #    legend += '{} {}'.format(topo_index, og.id_to_term[label].name)
    #with open(join(out_dir, 'graph_node_labels.txt'), 'w') as f:
    #    f.write(legend)

    # Load the metrics
    metrics_df = pd.read_csv(metrics_f, sep='\t', index_col=0)

     # Create the output directory
    #_run_cmd("mkdir -p %s" % out_dir)

    label_to_f1 = {
        label: metrics_df.loc[label][metric_name]
        for label in metrics_df.index
    }

    # F1-score drawn atop ontology
    draw_collapsed_ontology(
        label_graph,
        #label_to_topo_index,
        label_to_name,
        label_to_f1,
        metric_name,
        out_f
    )


def draw_collapsed_ontology(
        label_graph,
        label_to_name,
        label_to_metric,
        metric_name,
        out_f
    ):

    tmp_dir = "tmp_figs"
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
    with open(dot_f, 'w') as f:
        f.write(result_dot_str)
    _run_cmd("dot -Tpdf %s -o %s" % (dot_f, out_f))


def _diff_dot(
        source_to_targets,
        node_to_label,
        metric_name,
        node_to_color_intensity
    ):
    max_color_intensity = 1.0

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
            intensity = node_to_color_intensity[node]

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

        if node_to_color_intensity[node] > 0.55:
            font_color = 'black'
        else:
            font_color = 'white'

        g += '"{}" [style=filled,  fillcolor="{}", fontsize=20, fontcolor={}, fontname = "arial", label="{}"]\n'.format(
            LABEL_NAME_TO_SUCCINCT[node_to_label[node]],
            color_value,
            font_color,
            LABEL_NAME_TO_SUCCINCT[node_to_label[node]]
        )
    for source, targets in source_to_targets.items():
        for target in targets:
            if source in node_to_color_intensity and target in node_to_color_intensity:
                g += '"{}" -> "{}"\n'.format(
                    LABEL_NAME_TO_SUCCINCT[node_to_label[source]],
                    LABEL_NAME_TO_SUCCINCT[node_to_label[target]]
                )
    g += "}"
    return g




def _run_cmd(cmd):
    print("Running: %s" % cmd)
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    main()


