from optparse import OptionParser

import numpy as np
import json
import random

from one_nn import OneNN
from ensemble_binary_classifiers import EnsembleOfBinaryClassifiers 

CLASSIFIERS = {
    'onn': OneNN,
    'ind_one_vs_rest': EnsembleOfBinaryClassifiers 
}

class Model:
    def __init__(self, classifier, dim_reductor=None):
        """
        Parameters:
            classifier: a classifier object that performs
                supervised classification
            dim_reductor: a dimensonality reduction object
                that performs unsupervised dimensionality
                reduction. If this is supplied, all training
                and classification will be performed on the
                reduced dimensional representation of instances
                as learned by this algorithm.
        """
        self.dim_reductor = dim_reductor
        self.classifier = classifier

    def fit(
            self,
            train_X,
            train_items,
            item_to_labels,
            label_graph,
            item_to_group=None,
            verbose=False,
            features=None
        ):
        """
        Parameters: 
            train_X (matrix): an NxM matrix of training data 
                for N items and M features
            train_items (list): a N-length list of item-
                identifiers corresponding to the rows of
                train_X
            item_to_labels (dictionary): a dictionary mapping
                each item to its set of labels
            label_graph (DirectedAcyclicGraph): the graph of
                labels
            features (list): a M-length list of feature names 
        """
        if self.dim_reductor:
            self.dim_reductor.fit(train_X)
            train_X = self.dim_reductor.transform(
                train_X
            )
        self.classifier.fit(
            train_X,
            train_items,
            item_to_labels,
            label_graph,
            item_to_group=item_to_group,
            verbose=verbose,
            features=features
        )

    def predict(self, X, test_items):
        if self.dim_reductor:
            X = self.dim_reductor.transform(X)
        return self.classifier.predict(X, test_items)


def train_model(
        classifier_name, 
        params, 
        train_X, 
        train_items, 
        item_to_labels,
        label_graph,
        dim_reductor_name=None,
        dim_reductor_params=None,
        verbose=False,
        item_to_group=None,
        tmp_dir=None,
        features=None
    ):
    """
    Args:
        algorithm: the string representing the machine learning algorithm
        params: a dictioanry storing the parameters for the algorithm
        train_X: the training feature vectors
        train_items: the list of item identifiers corresponding to each feature
            vector
        item_to_labels: a dictionary mapping each identifier to its labels
        label_graph: a dictionary mapping each label to its neighbors in
            the label-DAG
        verbose: if True, output debugging messages during training and
            predicting
        tmp_dir: if the algorithm requires writing intermediate files
            then the files are placed in this directory
    """
    classifier = CLASSIFIERS[classifier_name](params)
    dim_reductor = None
    if dim_reductor_name:
        assert not dim_reductor_params is None
        dim_reductor = DIM_REDUCTORS[dim_reductor_name](dim_reductor_params)
    model = Model(
        classifier,
        dim_reductor=dim_reductor
    )
    model.fit(
        train_X,
        train_items,
        item_to_labels,
        label_graph,
        item_to_group=item_to_group,
        verbose=verbose,
        features=features
    )
    return model


if __name__ == "__main__":
    main()
