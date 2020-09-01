from optparse import OptionParser
import h5py
import json
import pandas as pd

from common import the_ontology
from common import ontology_utils
from graph_lib import graph
from onto_lib_py3 import ontology_graph


CELL_TYPE_TO_TERM = {
    'DENDRITIC': 'CL:0001057',              # myeloid dendritic cell, human 
    'MICROGLIA/MACROPHAGE': 'CL:0000235',   # macrophage 
    'NEUTROPHIL': 'CL:0000775',             # neutrophil
    'NK': 'CL:0000623',                     # natural killer cell 
    'NKT': 'CL:0000814',                    # matural natural killer T cell
    'Breg': 'CL:0000969',                   # regulatory B cell
    'EPITHELIAL': 'CL:0000066',             # epithelial cell           
    'Tm': 'CL:0000813',                     # memory T cell
    'Treg': 'CL:0000815',                   # regulatory T cell
    'PROLIFERATING MESENCHYMAL PROGENITOR': 'CL:0000134',    # mesenchymal stem cell
    'MAST': 'CL:0000097',                   # mast cell 
    'PERICYTE': 'CL:0000669',               # pericyte cell
    'ENDOTHELIAL': 'CL:0000115',            # endothelial cell
    'MDSC': 'CL:0000889',                   # myeloid suppressor cell 
    'Th': 'CL:0000912',                     # helper T cell 
    'MONOCYTE': 'CL:0001054',               # CD14-positive monocyte 
    'MACROPHAGE': 'CL:0000235',             # macrophage 
    'IG': 'CL:0000786',                     # plasma cell 
    'FIBROBLAST': 'CL:0000057',             # fibroblast 
    'DENDRITIC (ACTIVATED)': '0001057'      # myeloid dendritic cell, human 
}

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    parser.add_option("-o", "--out_file", help="File to write labels data")
    (options, args) = parser.parse_args()

    raw_10x_f = args[0]
    out_f = options.out_file

    # Load the ontology
    og = the_ontology.the_ontology()
    
    h = pd.HDFStore(raw_10x_f, mode='r')
    df = h['DF_ALL']
    cells = df.index.get_level_values('Cell ID')
    cell_types = df.index.get_level_values('CELL_TYPE') 

    print(set(cell_types))

    # Label each cell
    cell_id_to_labels = {}
    all_labels = set()
    for cell_id, cell_type in zip(cells, cell_types):
        ms_label = CELL_TYPE_TO_TERM[cell_type]
        labels = sorted(og.recursive_superterms(ms_label))
        cell_id_to_labels[cell_id] = labels
        all_labels.update(labels)

    # Generate label-graph
    label_graph = ontology_utils.ontology_subgraph_spanning_terms(
        all_labels,
        og
    )
    label_graph = graph.transitive_reduction_on_dag(label_graph)

    # Write output
    print("Writing output to {}...".format(out_f))
    with open(out_f, 'w') as  f:
        f.write(json.dumps(
            {
                'labels_config': {},
                'label_graph': {
                    source: list(targets)
                    for source, targets in label_graph.source_to_targets.items()
                },
                'labels': cell_id_to_labels
            },
            indent=4,
            separators=(',', ': ')
        ))



if __name__ == "__main__":
    main()
