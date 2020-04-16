###############################################################################
#   Given a set of experiments, create a label graph that spans these 
#   experiments' mapped ontology terms in the MetaSRA. 
###############################################################################

import os
from os.path import join, basename
import sys
from optparse import OptionParser
import json
import subprocess
from collections import deque, defaultdict

from common import the_ontology as the_og
import graph_lib
from graph_lib import graph
from onto_lib import ontology_graph

DEBUG = False

def main():
    usage = ""
    parser = OptionParser()
    parser.add_option("-o", "--out_file", help="Output file")
    (options, args) = parser.parse_args()
    
    annot_f = args[0]
    exp_set_f = args[1]
    out_f = options.out_file

    # Load metadata
    with open(annot_f, 'r') as f:
        exp_to_info = json.load(f)
    with open(exp_set_f, 'r') as f:
        the_exps = json.load(f)['experiments']

    # Label the experiments
    exp_to_terms = _label_experiments(
        the_exps,
        exp_to_info
    )

    # Generate the labelling-graph induced by this 
    # dataset
    og = the_og.the_ontology()
    all_terms = set()
    for terms in exp_to_terms.values():
        all_terms.update(terms)
    label_graph = _ontology_subgraph_spanning_terms(
        all_terms,
        og
    )
    label_graph = graph.transitive_reduction_on_dag(label_graph)

    # Write output
    exp_set_name = basename(exp_set_f).split('.')[0]
    with open(out_f, 'w') as  f:
        f.write(json.dumps(
            {
                'labels_config': {
                    'experiment_set': exp_set_name
                },
                'label_graph': {
                    source: list(targets)
                    for source, targets in label_graph.source_to_targets.iteritems()
                },
                'labels': exp_to_terms
            },
            indent=4,
            separators=(',', ': ')
        ))
 
 
def _ontology_subgraph_spanning_terms(
        span_terms,
        og
    ):
    """
    Builds the ontology subgraph spanning a set of terms.
    """
    # Get most general terms
    most_general_terms = ontology_graph.most_specific_terms(
        span_terms,
        og,
        sup_relations=["inv_is_a", "inv_part_of"]
    )
    if DEBUG:
        print "Most general terms: %s" % most_general_terms
    q = deque(most_general_terms)
    subgraph_source_to_targets = defaultdict(lambda: set())
    relations = ["inv_is_a", "inv_part_of"]
    #visited_ids = set(most_general_terms)
    while len(q) > 0:
        source_t_id = q.popleft()
        for rel in relations:
            if rel in og.id_to_term[source_t_id].relationships:
                for target_t_id in og.id_to_term[source_t_id].relationships[rel]:
                    target_descendants = set(
                        og.recursive_relationship(target_t_id, relations)
                    )
                    # There exists a descendant of the target represented in the samples
                    if len(target_descendants.intersection(span_terms)) > 0:
                        subgraph_source_to_targets[source_t_id].add(target_t_id)
                        q.append(target_t_id)
        #visited_ids.add(source_t_id)
    return graph.DirectedAcyclicGraph(subgraph_source_to_targets)


def _label_experiments(
        experiment_accs,
        exp_to_info,
        which_terms='mapped_terms'
    ):
    og = the_og.the_ontology()
    exp_to_terms = defaultdict(lambda: set())
    for exp in experiment_accs:
        mapped_terms = set(
            exp_to_info[exp][which_terms]
        )
        # compute all cell-type terms
        all_terms = set()
        for term in mapped_terms:
            all_terms.update(
                og.recursive_relationship(
                    term,
                    recurs_relationships=['is_a', 'part_of']
                )
            )
        all_terms = [
            x
            for x in all_terms
            if x.split(':')[0] == 'CL'
        ]
        exp_to_terms[exp] = all_terms
    return exp_to_terms

 
 
def run_cmd(cmd):
    print cmd
    subprocess.call(cmd, shell=True, env=None)
    

if __name__ == "__main__":
    main()
