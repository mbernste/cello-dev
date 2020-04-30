##########################################################################
#   Run a classifier on a test set
##########################################################################

from optparse import OptionParser
import json
from os.path import join
import dill

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
        "-o", 
        "--out_dir", 
        help="Directory in which to write output"
    )
    (options, args) = parser.parse_args()

    test_data_dir = args[0]
    out_dir = options.out_dir

    if options.model_f:
        with open(options.model_f, 'r') as f:
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
        print 'Re-ordering columns of data matrix to be in line with classifier...'
        gene_to_index = {
            gene: i
            for i, gene in enumerate(gene_ids)
        }
        indices = [
            gene_to_index[gene]
            for gene in mod.classifier.features
        ]
        data_matrix = data_matrix[:,indices]
        print 'done.'
            

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
    

if __name__ == "__main__":
    main()
