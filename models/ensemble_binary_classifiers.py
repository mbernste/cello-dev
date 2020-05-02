#################################################################
#   Supervised hierarchical classification using a per-label
#   binary support vector machine. Variants of this algorithm
#   enforce label-graph consistency by propogating positive
#   predictions upward through the graph's 'is_a' relationship
#   edges, and propogates negative predictions downward.
#################################################################

import sys
from collections import defaultdict
import numpy as np
import pandas as pd

import binary_classifier as bc

POS_CLASS = 1
VERBOSE = True

class EnsembleOfBinaryClassifiers(object):
    def __init__(
            self,
            params 
        ):
        self.binary_classif_algo = params['binary_classifier_algorithm']
        self.binary_classif_params = params['binary_classifier_params']
        self.assert_ambig_neg = params['assert_ambig_neg']

        # Per-label model artifacts
        self.label_to_pos_items = None
        self.label_to_neg_items = None
        self.label_to_classifier = None

        # Labels for which all samples in the training set are
        # annotated with
        self.trivial_labels = None

    def fit(
            self, 
            X, 
            train_items, 
            item_to_labels,  
            label_graph,
            item_to_group=None, 
            verbose=False,
            features=None
        ):
        """
        Args:
            X: NxM training set matrix of N samples by M features
            train_items: list of item identifiers, each corresponding
                to a row in X
            item_to_labels: dictionary mapping each item identifier to
                its set of labels in the label_graph
            label_graph: a Graph object representing the label DAG
            item_to_group: a dictionary mapping each item identifier
                to the group it belongs to. These groups may correspond
                to a shared latent variable and should be considered in
                the training process. For example, if the training items
                are gene expression profiles, the groups might be batches.
            verbose: if True, output debugging messages
        """
        self.train_items = train_items 
        self.label_graph = label_graph
        self.features = features

        # Map each item to its index in the item lost 
        item_to_index = {x:i for i,x in enumerate(self.train_items)}

        # Map each label to its items
        label_to_items = defaultdict(lambda: set())
        for item in self.train_items:
            labels = item_to_labels[item]
            for label in labels:
                label_to_items[label].add(item)
        label_to_items = dict(label_to_items)

        # Compute the positive items for each label
        print "Generating sets of positive examples..."
        self.label_to_pos_items = {}
        for label in label_to_items:
            pos_items = _compute_positive_examples(
                label,
                label_to_items,
                self.label_graph
            )
            self.label_to_pos_items[label] = pos_items

        # Compute the negative items for each label
        print "Generating sets of negative examples..."
        self.label_to_neg_items = {}
        for label in label_to_items:
            neg_items = _compute_negative_examples(
                label,
                self.train_items,
                self.label_to_pos_items[label],
                item_to_labels,
                self.label_graph,
                self.assert_ambig_neg
            )
            self.label_to_neg_items[label] = neg_items
          
        # Train a classifier at each node of the ontology
        self.trivial_labels = set()
        self.label_to_classifier = {}
        for label_i, curr_label in enumerate(label_to_items.keys()):
            pos_items = self.label_to_pos_items[curr_label]
            neg_items = self.label_to_neg_items[curr_label]

            pos_classes = list(np.full(len(pos_items), 1))
            neg_classes = list(np.full(len(neg_items), -1))
            train_y = np.asarray(pos_classes + neg_classes)
            train_items = pos_items + neg_items  
 
            pos_inds = [
                item_to_index[item]
                for item in pos_items
            ]
            neg_inds = [
                item_to_index[item]
                for item in neg_items
            ]
            pos_X = X[pos_inds,:]
            neg_X = X[neg_inds,:]
            train_X = np.concatenate([pos_X, neg_X])

            # Train classifier
            if VERBOSE:
                print "(%d/%d) training classifier for label %s..." % (
                    label_i+1, 
                    len(label_to_items), 
                    curr_label
                )
                print "Number of positive items: %d" % len(pos_items)
                print "Number of negative items: %d" % len(neg_items)

            if len(pos_items) > 0 and len(neg_items) == 0:
                self.trivial_labels.add(curr_label)
            else:
                model = bc.build_binary_classifier(
                    self.binary_classif_algo, 
                    self.binary_classif_params
                )
                model.fit(train_X, train_y)
                self.label_to_classifier[curr_label] = model

    def predict(self, X, test_items):
        label_to_scores = {}
        all_labels = sorted(self.label_to_classifier.keys())
        mat = []
        for label in all_labels:
            print "Making predictions for label %s" % label
            classifier = self.label_to_classifier[label]
            pos_index = 0 
            for index, clss in enumerate(classifier.classes_):
                if clss == POS_CLASS:
                    pos_index = index
                    break
            scores = [
                x[pos_index]
                for x in classifier.predict_proba(X)
            ]
            mat.append(scores)
        trivial_labels = sorted(self.trivial_labels)
        for label in trivial_labels:
            mat.append(list(np.full(len(test_items), 1.0)))

        all_labels += trivial_labels
        mat = np.array(mat).T
        df = pd.DataFrame(
            data=mat,
            index=test_items,
            columns=all_labels
        )
        return df, df


def _compute_positive_examples(
        label,
        label_to_items,
        label_graph
    ):
    """
    Compute the positive items for a given label.
    This set consists of all items labelled with a 
    descendent of the label
    """
    positive_items = label_to_items[label].copy()
    desc_labels = label_graph.descendent_nodes(label)
    for desc_label in desc_labels:
        if desc_label in label_to_items:
            positive_items.update(label_to_items[desc_label])
    return list(positive_items)


def _compute_negative_examples(
        label, 
        all_items, 
        pos_items, 
        item_to_labels, 
        label_graph,
        assert_ambig_neg
    ):
    """
    Compute the set of negative examples for a given label.
    This set consists of all items that are not labelled
    with a descendant of the label, but are also not labelled
    with an ancestor of the label (can include siblings).
    """
    anc_labels = label_graph.ancestor_nodes(label)
    candidate_items = set(all_items) - set(pos_items)

    if assert_ambig_neg:
        neg_items = list(candidate_items)
    else:
        # Remove items from the negatives that are labelled 
        # most specifically as an ancestor of the current label
        final_items = set()
        for item in candidate_items:
            ms_item_labels = label_graph.most_specific_nodes(
                item_to_labels[item]
            )
            ms_item_labels = set(ms_item_labels)
            if len(ms_item_labels & anc_labels) == 0:
                final_items.add(item)
        neg_items = list(final_items)
    return neg_items

