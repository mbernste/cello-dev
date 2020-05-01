#################################################################
#   Supervised hierarchical classification using a per-label
#   binary support vector machine. Variants of this algorithm
#   enforce label-graph consistency by propogating positive
#   predictions upward through the graph's 'is_a' relationship
#   edges, and propogates negative predictions downward.
#################################################################
import sys
from optparse import OptionParser
import numpy as np
from quadprog import solve_qp
import dill
import pandas as pd

from ensemble_binary_classifiers import EnsembleOfBinaryClassifiers 

def main():
    parser = OptionParser()
    #parser.add_option("-a", "--a_descrip", action="store_true", help="This is a flat")
    #parser.add_option("-b", "--b_descrip", help="This is an argument")
    (options, args) = parser.parse_args()

    feat_vecs = [
        [1,1,1,2,3],
        [10,23,1,24,32],
        [543,21,23,2,5]
    ]

    items = [
        'a',
        'b',
        'c'
    ]
    item_to_labels = {
        'a':['hepatocyte', 'disease'],
        'b':['T-cell'],
        'c':['stem cell', 'cultured cell']
    }


   
class IsotonicRegression():
    def __init__(
            self,
            params,
            trained_classifiers_f=None 
        ):
        self.params = params
       
    def fit(
            self,
            X,
            train_items,
            item_to_labels,
            label_graph,
            item_to_group=None,
            verbose=False,
            features=None,
            model_dependency=None
        ):
        """
        model_dependency: String, path to a dilled, pretrained ensemble of binary
            classifiers
        """

        # Either provide the model with a pre-trained ensemble of binary
        # classifiers or train them from scratch
        self.features = features
        if model_dependency is not None:
            with open(model_dependency, 'r') as f:
                self.ensemble = dill.load(f)
            # Make sure that this pre-trained model was trained on the 
            # same set of items and labels
            assert _validate_pretrained_model(
                self.ensemble, 
                train_items, 
                label_graph,
                features
            )
            self.features = self.ensemble.classifier.features
            self.train_items = self.ensemble.classifier.train_items
            self.label_graph = self.ensemble.classifier.label_graph
        else:
            self.ensemble = EnsembleOfBinaryClassifiers(self.params) 
            self.ensemble.fit(
                X,
                train_items,
                item_to_labels,
                label_graph,
                item_to_group=item_to_group,
                verbose=verbose,
                features=features
            )
            self.features = features
            self.train_items = train_items
            self.label_graph = label_graph

    def predict(self, X, test_items):

        confidence_df, scores_df = self.ensemble.predict(X, test_items)
        labels_order = confidence_df.columns        

        #label_to_scores = {}
        #for label, classifier in self.label_to_classifier.iteritems():
        #    pos_index = 0
        #    for index, clss in enumerate(classifier.classes_):
        #        if clss == 1:
        #            pos_index = index
        #            break
        #    scores = [
        #        x[pos_index] 
        #        for x in classifier.predict_proba(queries)
        #    ]
        #    label_to_scores[label] = scores
        #
        #labels_order = sorted(label_to_scores.keys())
        #label_to_prob_list = []
        #label_to_score_list = []

        # Create the constraints matrix
        constraints_matrix = []
        for row_label in labels_order:
            for constraint_label in self.label_graph.source_to_targets[row_label]:
                row = []
                for label in labels_order:
                    if label == row_label:
                        row.append(1.0)
                    elif label == constraint_label:
                        row.append(-1.0)
                    else:
                        row.append(0.0)
                constraints_matrix.append(row)
        b = np.zeros(len(constraints_matrix))
        constraints_matrix = np.array(constraints_matrix).T

        print "Label order (%d):" % len(labels_order)
        print labels_order
        print "Constraints matrix (%d, %d):" % (len(constraints_matrix), len(constraints_matrix.T))
        print constraints_matrix.T

        pred_da = []
        for q_i in range(len(X)):
            Q = np.eye(len(labels_order), len(labels_order))
            #predictions = np.array([ # Probabilities
            #    label_to_scores[label][q_i]
            #    for label in labels_order
            #])
            #predictions = np.array(predictions, dtype=np.double)
            predictions = np.array(
                confidence_df[labels_order].iloc[q_i],
                dtype=np.double
            )
            #predictions = np.array(predictions, dtype=np.double)
            print "Running solver on item %d/%d..." % (q_i+1, len(X))
            xf, f, xu, iters, lagr, iact = solve_qp(
                Q, 
                predictions, 
                constraints_matrix, 
                b
            )

            # xf is the final list of probabilities for item q_i
            # ordered by 
            pred_da.append(xf)

            #label_to_prob = {}
            #for label, est_prob in zip(labels_order, xf):
            #    label_to_prob[label] = est_prob
            #
            #label_to_prob_list.append(label_to_prob)
            #label_to_score = {}
            #for label, score in zip(labels_order, predictions):
            #    label_to_score[label] = label_to_scores[label][q_i]
            #label_to_score_list.append(label_to_score)
        pred_df = pd.DataFrame(
            data=pred_da,
            columns=labels_order,
            index=test_items
        )
        return pred_df, confidence_df
        #return  label_to_prob_list, label_to_score_list


def _validate_pretrained_model(ensemble, train_items, label_graph, features):
    # Check that the label-graphs have same set of labels
    classif_labels = frozenset(ensemble.classifier.label_graph.get_all_nodes())
    curr_labels = frozenset(label_graph.get_all_nodes())
    if classif_labels != curr_labels:
        return False
    classif_train_items = frozenset(ensemble.classifier.train_items)
    curr_train_items = frozenset(train_items)
    if classif_train_items != curr_train_items:
        return False
    if tuple(ensemble.classifier.features) != tuple(features):
        return False
    return True

if __name__ == "__main__":
    main()
