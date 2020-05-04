from optparse import OptionParser

import numpy as np
import json
import random

from .one_nn import OneNN
from .ensemble_binary_classifiers import EnsembleOfBinaryClassifiers 
from .cascaded_discriminative_classifiers import CascadedDiscriminativeClassifiers 
from .isotonic_regression import IsotonicRegression
from .true_path_rule import TruePathRule
from .scale import Scale
from .pca import PCA

CLASSIFIERS = {
    'onn': OneNN,
    'ind_one_vs_rest': EnsembleOfBinaryClassifiers,
    'cdc': CascadedDiscriminativeClassifiers,
    'isotonic_regression': IsotonicRegression,
    'true_path_rule': TruePathRule
}

PREPROCESSORS = {
    'scale': Scale,
    'pca': PCA
}

class Model:
    def __init__(self, classifier, preprocessors=None):
        """
        Parameters:
            classifier: a classifier object that performs
                supervised classification
            preprocessors: a list of preprocessor algorithms
                for transforming the data before fitting
        """
        self.preprocessors = preprocessors
        self.classifier = classifier

    def fit(
            self,
            train_X,
            train_items,
            item_to_labels,
            label_graph,
            item_to_group=None,
            verbose=False,
            features=None,
            model_dependency=None
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
        if self.preprocessors:
            for prep in self.preprocessors:
                prep.fit(train_X)
                train_X = prep.transform(
                    train_X
                )
        self.classifier.fit(
            train_X,
            train_items,
            item_to_labels,
            label_graph,
            item_to_group=item_to_group,
            verbose=verbose,
            features=features,
            model_dependency=model_dependency
        )

    def predict(self, X, test_items):
        if self.preprocessors:
            for prep in self.preprocessors:
                X = prep.transform(X)
        return self.classifier.predict(X, test_items)


def train_model(
        classifier_name, 
        params, 
        train_X, 
        train_items, 
        item_to_labels,
        label_graph,
        preprocessor_names=None,
        preprocessor_params=None,
        verbose=False,
        item_to_group=None,
        tmp_dir=None,
        features=None,
        model_dependency=None
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
    preps = None
    if preprocessor_names:
        assert preprocessor_params is not None
        assert len(preprocessor_params) == len(preprocessor_names)
        preps = [
            PREPROCESSORS[prep_name](prep_params)
            for prep_name, prep_params in zip(preprocessor_names, preprocessor_params)
        ]
    model = Model(
        classifier,
        preprocessors=preps
    )
    model.fit(
        train_X,
        train_items,
        item_to_labels,
        label_graph,
        item_to_group=item_to_group,
        verbose=verbose,
        features=features,
        model_dependency=model_dependency
    )
    return model


if __name__ == "__main__":
    main()
