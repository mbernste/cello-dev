##########################################################################
#   Run a classifier on a test set
##########################################################################

from optparse import OptionParser
import json
from os.path import join
import dill
from collections import defaultdict
import pandas as pd

import train_model
from common import load_dataset

def main():
    usage = "" # TODO 
    parser = OptionParser(usage=usage)
    parser.add_option(
        "-m", 
        "--model_f", 
        help="Load model from file"
    )
    parser.add_option(
        "-t",
        "--train_dir",
        help="Training dataset directory"
    )
    parser.add_option(
        "-p",
        "--train_params",
        help="Training parameters"
    )
    parser.add_option(
        "-c",
        "--classification_threshold",
        help="Classification score to use as the threshold for a positive classification"
    )
    parser.add_option(
        "-f",
        "--classification_threshold_file",
        help="Path to JSON file mapping each label to its classification threshold for calling a positive classification"
    )
    parser.add_option(
        "-o", 
        "--out_dir", 
        help="Directory in which to write output"
    )
    (options, args) = parser.parse_args()

    test_data_dir = args[0]
    out_dir = options.out_dir

    if options.model_f:
        with open(options.model_f, 'rb') as f:
            mod = dill.load(f)
        features = args[1] 
    else:
        assert options.train_dir is not None
        assert options.train_params is not None
        with open(options.train_params, 'r') as f:
            config=json.load(f)
        features = config['features']
        algorithm = config['algorithm']
        params = config['params']
        mod = train_model.train_model(
            options.train_dir,
            features,
            algorithm, 
            params,
            join(out_dir, 'tmp')
        )        

    # Load the test data
    r = load_dataset.load_dataset(
        test_data_dir,
        features
    )
    the_exps = r[3]
    data_matrix = r[10]
    gene_ids = r[11]

    # Re-order columns of data matrix to be same as expected
    # by the model
    assert frozenset(mod.classifier.features) == frozenset(gene_ids)
    if not tuple(mod.classifier.features) == tuple(gene_ids):
        print('Re-ordering columns of data matrix in accordance with classifier input specification...')
        gene_to_index = {
            gene: i
            for i, gene in enumerate(gene_ids)
        }
        indices = [
            gene_to_index[gene]
            for gene in mod.classifier.features
        ]
        data_matrix = data_matrix[:,indices]
        print('done.')
            

    # Apply model
    print('Applying model to test set.')
    confidence_df, score_df = mod.predict(data_matrix, the_exps)
    print('done.')

    # Write output to files
    confidence_df.to_csv(
        join(out_dir, 'classification_results.tsv'),
        sep='\t'
    )
    score_df.to_csv(
        join(out_dir, 'classification_scores.tsv'),
        sep='\t'
    )
 
    # Binarize the classifications 
    if options.classification_threshold \
        or options.classification_threshold_file:
        if options.classification_threshold:
            assert options.classification_threshold_file is None
            classif_thresh = float(options.classification_threshold)
            label_graph = mod.classifier.label_graph
            label_to_thresh = defaultdict(lambda: classif_thresh)
        if options.classification_threshold_file:
            assert options.classification_threshold is None
            with open(options.classification_threshold_file, 'r') as f:
                label_to_thresh = json.load(f)
            label_graph = mod.classifier.label_graph
        binary_df = _binarize_classifiations(
            confidence_df, 
            label_to_thresh, 
            label_graph
        )
        binary_df.to_csv(
            join(out_dir, 'binary_classification_results.tsv'),
            sep='\t'
        ) 

 
def _binarize_classifiations(confidence_df, label_to_thresh, label_graph):
    da = []
    for exp in confidence_df.index:
        # Map each label to its classification-score 
        label_to_conf = {
            label: confidence_df.loc[exp][label]
            for label in confidence_df.columns
        }
        # Compute whether each label is over its threshold
        label_to_is_above = {
            label: int(conf > label_to_thresh[label])
            for label, conf in label_to_conf.items()
            if label in confidence_df.columns
        }
        label_to_bin= {
            label: is_above
            for label, is_above in label_to_is_above.items()
        }
        # Propagate the negative predictions to all descendents
        for label, over_thresh in label_to_is_above.items():
            if not bool(over_thresh):
                desc_labels = label_graph.descendent_nodes(label)
                for desc_label in set(desc_labels) & set(label_to_bin.keys()):
                    label_to_bin[desc_label] = int(False)
        da.append([
            label_to_bin[label]
            for label in confidence_df.columns
        ])
    df = pd.DataFrame(
        data=da,
        index=confidence_df.index,
        columns=confidence_df.columns
    )
    return df

if __name__ == "__main__":
    main()
