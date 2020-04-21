#########################################################################
#   Train a hiearchical classifier on an experiment list and pickle the
#   model.
#########################################################################

import sys
import os
from os.path import join, basename
from optparse import OptionParser
import json
import collections
from collections import defaultdict, Counter
import numpy as np
import dill

from models import model
from common import load_dataset

def main():
    usage = "usage: %prog <configuration_file> <dataset_directory>"
    parser = OptionParser(usage)
    parser.add_option(
        "-o", 
        "--out_dir", 
        help="Directory in which to write the model"
    )
    (options, args) = parser.parse_args()

    config_f = args[0]
    dataset_dir = args[1]
    out_dir = options.out_dir

    # Load the configuration
    print "Reading configuration from %s." % config_f 
    with open(config_f, 'r') as f:
        config = json.load(f)
    features = config['features']
    algorithm = algo_config['algorithm']
    params = algo_config['params']

    # Train model
    mod = _train_model(
        dataset_dir, 
        features, 
        algorithm, 
        params,
        join(out_dir, 'tmp')
    )

    print "Dumping the model with dill..."
    out_f = join(out_dir, 'model.dill')
    with open(out_f, 'w') as f:
        dill.dump(mod, f)
    print "done."

def train_model(dataset_dir, features, algorithm, params, tmp_dir):
    # Load the data
    r = load_dataset.load_dataset(
        dataset_dir,
        features
    )
    og = r[0]
    label_graph = r[1]
    label_to_name = r[2]
    the_exps = r[3]
    exp_to_index = r[4]
    exp_to_labels = r[5]
    exp_to_tags = r[6]
    exp_to_study = r[7]
    study_to_exps = r[8]
    exp_to_ms_labels = r[9]
    data_matrix = r[10]
    gene_names = r[11]

    # Train the classifier
    print 'Training model: %s' % algorithm
    print 'Parameters:\n%s' % json.dumps(params, indent=4)
    mod = model.train_model(
        algorithm,
        params,
        data_matrix,
        the_exps,
        exp_to_labels,
        label_graph,
        item_to_group=exp_to_study,
        tmp_dir=tmp_dir,
        features=gene_names
    )
    print 'done.'
    return mod



if __name__ == "__main__":
    main()
