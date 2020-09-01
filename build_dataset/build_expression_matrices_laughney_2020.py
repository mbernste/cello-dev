from optparse import OptionParser
import h5py
import pandas as pd
import numpy as np
from collections import defaultdict

def main():
    usage = ""
    parser = OptionParser()
    #parser.add_option("-o", "--out_file", help="Output file")
    (options, args) = parser.parse_args()

    tenx_raw_f = args[0]    
    in_bulk_f = args[1]
    out_f_10x = args[2]
    out_f_bulk = args[3]

    # Load the bulk RNA-seq dataset
    print("Loading bulk RNA-seq data from {}...".format(in_bulk_f))
    with h5py.File(in_bulk_f, 'r') as f:
        X_bulk = f['expression'][:]
        exps_bulk = [
            str(x)[2:-1]
            for x in f['experiment'][:]
        ]
        genes_bulk = [
            str(x)[2:-1]
            for x in f['gene_id'][:]
        ]
    # Compute TPM from log1(TPM)
    X_bulk = np.exp(X_bulk)-1
    print("done.")

    # Load the Laughney et al. 10x data
    print("Loading 10x data from {}...".format(tenx_raw_f))
    h = pd.HDFStore(tenx_raw_f, mode='r')
    df = h['DF_ALL']
    print(df)
    print("done.")
   
    test_genes = list(df.columns)

    # Map each bulk gene to its index
    gene_to_index = {
        gene: index
        for index, gene in enumerate(genes_bulk)
    }

    # Match the training set genes to the test set genes
    final_genes, gene_to_indices = _match_genes(
        test_genes, 
        genes_bulk, 
        gene_to_index
    )
    print('Matched a total of {} genes.'.format(len(final_genes)))

    X_bulk = _expression_matrix_subset(X_bulk, final_genes, gene_to_indices)
    
    print(X_bulk.shape)

    # Compute log1(TPM) from TPM
    X_bulk = np.log(X_bulk+1)

    # Compute the 10x numpy array
    X_10x = np.array(df[final_genes])
    print(X_10x.shape)

    # Extract the cell ID's
    cells = [
        x.encode('utf-8')
        for x in list(df.index.get_level_values('Cell ID'))
    ]
    
    exps_bulk = [
        x.encode('utf-8')
        for x in exps_bulk
    ]
    final_genes = [
        str(x).encode('utf-8')
        for x in final_genes
    ]

    print('Writing data to {}...'.format(out_f_10x))
    with h5py.File(out_f_10x, 'w') as f:
        f.create_dataset('expression', data=X_10x, compression="gzip")
        f.create_dataset('experiment', data=np.array(cells))
        f.create_dataset('gene_id', data=np.array(final_genes)) # Note, the name of the dataset is a bit misleading given that we are now working with gene symbols
    print('done.')
     
    print('Writing data to {}...'.format(out_f_bulk))
    with h5py.File(out_f_bulk, 'w') as f:
        f.create_dataset('expression', data=X_bulk, compression="gzip")
        f.create_dataset('experiment', data=np.array(exps_bulk))
        f.create_dataset('gene_id', data=np.array(final_genes)) # Note, the name of the dataset is a bit misleading given that we are now working with gene symbols
    print('done.')
   
def _match_genes(test_genes, all_genes, gene_to_index):
    genes_f = "biomart_id_to_symbol.tsv"
    with open(genes_f, 'r') as f:
        sym_to_ids = defaultdict(lambda: [])
        for l in f:
            gene_id, gene_sym = l.split('\t')
            gene_id = gene_id.strip()
            gene_sym = gene_sym.strip()
            sym_to_ids[gene_sym].append(gene_id)
    # Gather training genes
    train_ids = []
    train_genes = []
    all_genes_s = set(all_genes)
    not_found = []
    gene_to_indices = defaultdict(lambda: [])
    for sym in test_genes:
        if sym in sym_to_ids:
            ids = sym_to_ids[sym]
            for idd in ids:
                if idd in all_genes_s:
                    train_genes.append(sym)
                    train_ids.append(idd)
                    gene_to_indices[sym].append(gene_to_index[idd])
        else:
            not_found.append(sym)
    gene_to_indices = dict(gene_to_indices)
    print('Of {} genes in test set, found {} of {} training set genes in input file.'.format(
        len(test_genes),
        len(train_ids),
        len(all_genes)
    ))
    return train_genes, gene_to_indices


def _expression_matrix_subset(X, genes, gene_to_indices):
    """
    Take a subset of the columns for the training-genes. Note
    that if a given gene in the test set maps to multiple training
    genes, then we sum over the training genes.
    """
    X_new = []
    for gene in genes:
        indices = gene_to_indices[gene]
        X_new.append(np.sum(X[:,indices], axis=1))
    X_new = np.array(X_new).T
    assert X_new.shape[1] == len(genes)
    return X_new


if __name__ == '__main__':
    main()
