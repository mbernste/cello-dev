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
    'professional antigen presenting cell': 'professional antigen\npresenting',
    'lymphocyte of B lineage': 'lymphocyte of\nB lineage',
    'T cell': 'T',
    'innate lymphoid cell': 'innate lymphoid',
    'B cell': 'B',
    'mature T cell': 'mature T',
    'alpha-beta T cell': 'alpha-beta T',
    'group 1 innate lymphoid cell': 'group 1 innate\nlymphoid cell',
    'regulatory T cell': 'regulatory T',
    'effector T cell': 'effector T',
    'mature alpha-beta T cell': 'mature\nalpha-beta T',
    'natural killer cell': 'NK',
    'helper T cell': 'helper T',
    'CD4-positive, alpha-beta T cell': 'CD4+ T',
    'memory T cell': 'memory T',
    'naive T cell': 'naive T',
    'CD8-positive, alpha-beta T cell': 'CD8+ T',
    'CD4-positive, CD25-positive, alpha-beta regulatory T cell': 'CD4+CD25+\nregulatory T',
    'CD4-positive helper T cell': 'CD4+ helper T',
    'CD4-positive, alpha-beta memory T cell': 'CD4+ memory\nT cell',
    'naive thymus-derived CD4-positive, alpha-beta T cell': 'CD4+ naive T',
    'naive thymus-derived CD8-positive, alpha-beta T cell': 'CD8+ naive T',
    'neural cell': 'neural',
    'female germ cell': 'female germ',
    'glandular epithelial cell': 'glandular\nepithelial',
    'peripheral blood mononuclear cell': 'PBMC',
    'enteroendocrine cell': 'enteroendocrine',
    'oocyte': 'oocyte',
    'efferent neuron': 'efferent\nneuron',
    'electrically responsive cell': 'electrically\nresponsive',
    'electrically signaling cell': 'electrically\nsignaling',
    'circulating cell': 'circulating',
    'epithelial cell': 'epithelial',
    'male germ cell': 'male germ',
    'embryonic cell': 'embryonic',
    'epithelial cell of pancreas': 'pancreatic\nepithelial',
    'germ cell': 'germ',
    'germ line cell': 'germ line',
    'conventional dendritic cell': 'conventional\ndendritic',
    'primordial germ cell': 'primordial\ngerm',
    'pancreatic endocrine cell': 'pancreatic\nendocrine',
    'early embryonic cell': 'early\nembryonic',
    'extraembryonic cell': 'extraembryonic',
    'central nervous system neuron': 'central nervous\nsystem neuron',
    'spinal cord motor neuron': 'spinal cord\nmotor neuron',
    'dendritic cell': 'dendritic',
    'CNS neuron (sensu Vertebrata)': 'CNS neuron\n(sensu Vertebrata)',
    'secretory cell': 'secretory',
    'neuron': 'neuron',
    'trophectodermal cell': 'trophectodermal',
    'stem cell': 'stem',
    'electrically active cell': 'electrically\nactive',
    'motor neuron': 'motor\nneuron',
    'hematopoietic lineage restricted progenitor cell': 'hematopoietic lineage\nrestricted progenitor',
    'blood cell': 'blood',
    'endocrine cell': 'endocrine',
    'germ line stem cell': 'germ line\nstem cell',
    'cell': 'cell',
    'native cell': 'native',
    'morula cell': 'morula',
    'pre-conventional dendritic cell': 'pre-conventional\ndendritic',
    'male germ line stem cell': 'male germ line stem',
    'male germ line stem cell (sensu Vertebrata)': 'male germ line\nstem (sensu Vertebrata)',
    'germ line stem cell (sensu Vertebrata)': 'germ line\nstem (sensu Vertebrata)',
    "smooth muscle cell": "smooth muscle",
    "anucleate cell": "anucleate",
    "antibody secreting cell": "antibody\nsecreting",
    "regulatory B cell": "regulatory B",
    "microglial cell": "microglial",
    "granulocyte": "granulocyte",
    "non-striated muscle cell": "non-striated\nmuscle",
    "plasmacytoid dendritic cell": "plasmacytoid\ndendritic",
    "IgG memory B cell": "IgG memory B",
    "pneumocyte": "pneumocyte",
    "muscle cell": "muscle",
    "androgen binding protein secreting cell": "androgen binding\nprotein secreting",
    "neurecto-epithelial cell": "neurecto-epithelial",
    "biogenic amine secreting cell": "biogenic amine\nsecreting",
    "hematopoietic oligopotent progenitor cell": "hematopoietic\noligopotent\nprogenitor",
    "placental pericyte": "placental\npericyte",
    "gamete": "gamete",
    "central memory CD4-positive, alpha-beta T cell": "central memory\nCD4+ T",
    "type B pancreatic cell": "pancreatic beta",
    "megakaryocyte-erythroid progenitor cell": "megakaryocyte-erythroid\nprogenitor",
    "cumulus cell": "cumulus",
    "seminiferous tubule epithelial cell": "seminiferous tubule\nepithelial",
    "CD8-positive, alpha-beta memory T cell": "memory CD8+ T",
    "contractile cell": "contractile",
    "endo-epithelial cell": "endo-epithelial",
    "non-terminally differentiated cell": "non-terminally\ndifferentiated",
    "memory B cell": "memory B",
    "IgD-negative memory B cell": "IgD-negative\nmemory B",
    "glial cell (sensu Vertebrata)": "glial\n(sensu Vertebrata)",
    "myeloid dendritic cell, human": "myeloid dendritic\n(human)",
    "protein secreting cell": "protein secreting",
    "CD141-positive myeloid dendritic cell": "CD141+ myeloid\ndendritic",
    "connective tissue cell": "connective tissue",
    "cell in vitro": "in vitro",
    "single fate stem cell": "single fate\nstem",
    "meso-epithelial cell": "meso-epithelial",
    "endothelial cell of umbilical vein": "umbilical vein\nendothelial",
    "stratified epithelial cell": "stratified\nepithelial",
    "vascular associated smooth muscle cell": "vascular\nsmooth muscle",
    "stratified squamous epithelial cell": "stratified squamous\nepithelial",
    "sperm": "sperm",
    "myoblast": "myoblast",
    "glucagon secreting cell": "glucagon secreting",
    "ecto-epithelial cell": "ecto-epithelial",
    "plasmablast": "plasmablast",
    "mammary gland epithelial cell": "mammary gland\nepithelial",
    "keratinocyte": "keratinocyte",
    "granulosa cell": "granulosa",
    "pancreatic A cell": "pancreatic alpha",
    "central nervous system macrophage": "central nervous\nsystem macrophage",
    "astrocyte": "astrocyte",
    "visceral muscle cell": "visceral muscle",
    "type A enterocrine cell": "type A enterocrine",
    "hepatocyte": "hepatocyte",
    "myeloid lineage restricted progenitor cell": "myeloid lineage\nrestricted progenitor",
    "squamous epithelial cell": "squamous\nepithelial",
    "platelet": "platelet",
    "plasma cell": "plasma",
    "brain microvascular endothelial cell": "brain microvascular\nendothelial",
    "surfactant secreting cell": "surfactant\nsecreting",
    "cultured cell": "cultured",
    "hematopoietic oligopotent progenitor cell, lineage-negative": "hematopoietic oligopotent\nprogenitor, lineage-negative",
    "glial cell": "glial",
    "oxygen accumulating cell": "oxygen\naccumulating",
    "keratin accumulating cell": "keratin\naccumulating",
    "epithelial cell of alveolus of lung": "epithelial of\nalveolus lung",
    "keratinizing barrier epithelial cell": "keratinizing barrier\nepithelial",
    "phagocyte": "phagocyte",
    "myeloid dendritic cell": "myeloid\ndendritic",
    "basal cell of prostate epithelium": "basal cell of\nprostate epithelium",
    "neuron of the substantia nigra": "substantia nigra\nneuron",
    "microvascular endothelial cell": "microvascular\nendothelial",
    "spermatid": "spermatid",
    "columnar/cuboidal epithelial cell": "columnar/cuboidal\nepithelial",
    "tissue-resident macrophage": "tissue-resident\nmacrophage",
    "bone marrow cell": "bone marrow",
    "class switched memory B cell": "class switched\nmemory B",
    "foreskin keratinocyte": "foreskin\nkeratinocyte",
    "mature neutrophil": "mature\nneutrophil",
    "eukaryotic cell": "eukaryotic",
    "mononuclear cell of bone marrow": "bone marrow\nmononuclear",
    "steroid hormone secreting cell": "steroid hormone\nsecreting",
    "metabolising cell": "metabolising",
    "male gamete": "male gamete",
    "blood vessel endothelial cell": "blood vessel\nendothelial",
    "bone marrow hematopoietic cell": "bone marrow\nhematopoietic",
    "corneal endothelial cell": "corneal\nendothelial",
    "erythrocyte": "erythrocyte",
    "experimentally modified cell in vitro": "experimentally modified\nin vitro",
    "follicular cell of ovary": "ovary follicular",
    "serotonin secreting cell": "serotonin\nsecreting",
    "effector memory CD4-positive, alpha-beta T cell": "effector memory\nCD4+ T",
    "bone cell": "bone",
    "pigment cell": "pigment",
    "luminal cell of prostate epithelium": "luminal of\nprostate epithelium",
    "germinal center B cell": "germinal\ncenter B",
    "CD1c-positive myeloid dendritic cell": "CD1c+ myeloid\ndendritic",
    "activated CD4-positive, alpha-beta T cell": "activated\nCD4+ T",
    "ciliated cell": "ciliated",
    "basal cell": "basal",
    "muscle precursor cell": "muscle precursor",
    "haploid cell": "haploid",
    "somatic stem cell": "somatic stem",
    "mature B cell": "mature B",
    "classical monocyte": "classical\nmonocyte",
    "mesothelial cell": "mesothelial",
    "erythroid lineage cell": "erythroid\nlineage",
    "hematopoietic stem cell": "hematopoietic\nstem",
    "lung secretory cell": "lung secretory",
    "dendritic cell, human": "dendritic\n(human)",
    "peptide hormone secreting cell": "peptide hormone\nsecreting",
    "naive B cell": "naive B",
    "duct epithelial cell": "duct epithelial",
    "IgM memory B cell": "IgM memory B",
    "neuron associated cell": "neuron\nassociated",
    "respiratory epithelial cell": "respiratory\nepithelial",
    "epithelial cell of lung": "lung epithelial",
    "Kupffer cell": "Kupffer",
    "general ecto-epithelial cell": "general\necto-epithelial",
    "fibroblast": "fibroblast",
    "insulin secreting cell": "insulin secreting",
    "lining cell": "lining",
    "epithelial cell of prostate": "prostate\nepithelial",
    "activated CD4-positive, alpha-beta T cell, human": "activated CD4+ T\n(human)",
    "granulocytopoietic cell": "granulocytopoietic",
    "alveolar macrophage": "alveolar\nmacrophage",
    "respiratory basal cell": "respiratory\nbasal",
    "endothelial cell": "endothelial",
    "epidermal cell": "epidermal",
    "dopaminergic neuron": "dopaminergic\nneuron",
    "Sertoli cell": "Sertoli",
    "CD14-positive, CD16-negative classical monocyte": "CD14+CD16- classical\nmonocyte",
    "trophoblast cell": "trophoblast",
    "CD7-negative lymphoid progenitor OR granulocyte monocyte progenitor": "CD7-lymphoid\nprogenitor OR\ngranulocyte monocyte\nprogenitor",
    "CD4-positive, CXCR3-negative, CCR6-negative, alpha-beta T cell": "CD4+CXCR3-CCR6- T",
    "epithelial cell of upper respiratory tract": "upper respiratory\ntract epithelial",
    "barrier cell": "barrier",
    "stuff accumulating cell": "stuff accumulating",
    "epithelial fate stem cell": "epithelial\nfate stem",
    "T-helper 2 cell": "T-helper 2",
    "neutrophil": "neutrophil",
    "oligodendrocyte": "oligodendrocyte",
    "melanocyte": "melanocyte",
    "pericyte cell": "pericyte",
    "pleural macrophage": "pleural\nmacrophage",
    "endopolyploid cell": "endopolyploid",
    "type II pneumocyte": "type II\npneumocyte",
    "vein endothelial cell": "vein\nendothelial",
    "promyelocyte": "promyelocyte",
    "endothelial cell of vascular tree": "vascular tree\nendothelial",
    "animal cell": "animal",
    "macrophage": "macrophage",
    "spermatocyte": "spermatocyte",
    "granulocyte monocyte progenitor cell": "granulocyte\nmonocyte\nprogenitor",
    "supportive cell": "supportive",
    "macroglial cell": "macroglial",
    "polyploid cell": "polyploid"
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
    print('Reading label graph from {}.'.format(label_graph_f))
    with open(label_graph_f, 'r') as f:
        label_data = json.load(f)
    label_graph = DirectedAcyclicGraph(label_data['label_graph'])
    label_to_name = {
        x: og.id_to_term[x].name
        for x in label_graph.get_all_nodes()
    }

    print('\n'.join(set(label_to_name.values()) - set(LABEL_NAME_TO_SUCCINCT.keys())))

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

    label_to_metric = {
        label: metrics_df.loc[label][metric_name]
        for label in metrics_df.index
        if label in label_to_name
    }

    # F1-score drawn atop ontology
    draw_collapsed_ontology(
        label_graph,
        label_to_name,
        label_to_metric,
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
    dot_f = join(tmp_dir, "ontology_graph.dot")
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
        #cmap = mpl.cm.get_cmap('Blues')
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


