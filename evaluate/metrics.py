###########################################################################################
#   Evaluate the output of a leave-study-out cross-validation experiment. More
#   specifically, this script analyzes confidence score outputs.
###########################################################################################

import matplotlib as mpl
mpl.use('Agg')

import os
from os.path import join
import sys

import random
import json
from optparse import OptionParser
import collections
from collections import defaultdict
import seaborn as sns
from matplotlib import pyplot as plt
import pandas
import sklearn
from sklearn import metrics
import numpy as np
import pandas as pd

sys.path.append("/ua/mnbernstein/projects/vis_lib")
sys.path.append("/ua/mnbernstein/projects/tbcp/metadata/ontology/src")
sys.path.append("/ua/mnbernstein/projects/tbcp/phenotyping/common")

import project_data_on_ontology as pdoo
from map_sra_to_ontology import ontology_graph
from map_sra_to_ontology import load_ontology
import vis_lib

PBMC_TERM_ID = 'CL:2000001'
MONONUCLEAR_CELL_TERM_ID = 'CL:0000842'

def main():
    scores = [-1.0, -0.5, -0.5, 0.0, 0.5, 1.0]
    assigneds  = [False, False, True, True, False, True]
    print _precision_recall_curve(assigneds, scores)


def convert_keys_to_label_strs(label_to_metric, og):
    return {
        pdoo.convert_node_to_name(k, og): v
        for k,v in label_to_metric.iteritems()
    }


def data_metrics(
        samples,
        sample_to_labels,
        sample_to_info
    ):
    """
    Compute metrics on the dataset independently of any predictions
    made by a classifier.
    """
    label_to_studies = defaultdict(lambda: set())
    for sample in samples:
        study = sample_to_info[sample]['study_accession']
        for label in sample_to_labels[sample]:
            label_to_studies[label].add(study)
    label_to_num_studies = {
        k:len(v) 
        for k,v in label_to_studies.iteritems()
    }
    return label_to_num_studies


def _compute_assignment_matrix(
        confidence_df,
        item_to_labels
    ):
    items = list(confidence_df.index)
    labels = sorted(confidence_df.columns)
    mat = [
        [
            label in item_to_labels[item]
            for label in labels
        ]
        for item in items
    ]
    assignment_df = pd.DataFrame(
        data=mat,
        columns=labels,
        index=items
    )
    return assignment_df


def compute_label_centric_metrics(
        confidence_df,
        assignment_df,
        compute_labels,
        label_graph=None,
        label_to_name=None,
        og=None,
        conservative=False
    ):
    if conservative:
        assert label_graph is not None
        assert label_to_name is not None
        assert og is not None
        label_to_assigneds, label_to_scores = _compute_conservative_assignment_and_score_lists(
            confidence_df,
            assignment_df,
            label_graph,
            label_to_name,
            og,
            blacklist_labels=None
        )
    else:
        label_to_assigneds = {
            label: assignment_df[label]
            for label in assignment_df.columns
        } 
        label_to_scores = {
            label: confidence_df[label]
            for label in confidence_df.columns
        }

    label_to_pr_curve = {}
    metrics_mat = []
    kept_labels = []
    for label in compute_labels:
        assigneds = label_to_assigneds[label]
        confidences = label_to_scores[label]
        assert len(assigneds) > 0
        # This checks that at least one sample is annotated with the current label
        if len(frozenset(assigneds)) == 1 and list(set(assigneds))[0] == False:
            print "WARNING! No samples are assigned label %s. Skipping computing metrics for this label." %  label
            continue
        if len(set(assigneds)) == 1 and list(set(assigneds))[0] == True:
            print "WARNING! All samples are assigned label %s. Skipping computing metrics for this label." %  label
            continue
        kept_labels.append(label)
        # AUC
        try:
            auc = metrics.roc_auc_score(assigneds, confidences)
        except:
            print "WARNING! Unable to compute AUC for label %s" % label
            auc = -1.0
        # Precision-recall curve
        precisions, recalls, threshs = _precision_recall_curve(
            assigneds,
            confidences
        )
        # Average precision
        avg_precision = _average_precision(
            precisions, 
            recalls
        )

        # Achievable recall at 0.9 precision
        prec_thresh = 0.9
        achiev_rec = _max_recall_at_prec_thresh(
            precisions,
            recalls,
            prec_thresh
        )
        metrics_mat.append((auc, avg_precision, achiev_rec))
        label_to_pr_curve[label] = (precisions, recalls, threshs)
    metrics_df = pd.DataFrame(
        data=metrics_mat,
        columns=['AUC', 'Avg. Precision', 'Achievable Recall at 0.9 Precision'],
        index=kept_labels
    )
    return metrics_df, label_to_pr_curve



def _max_recall_at_prec_thresh(
        precisions,
        recalls,
        thresh
    ):
    achievable_recs = [0.0]
    achievable_recs += [
        rec
        for prec, rec in zip(precisions, recalls)
        if prec > thresh
    ]
    return max(achievable_recs)



def compute_joint_metrics():
    # Compute the pan-dataset precision-recall curve
    global_assigneds = []  
    global_scores = []
    global_labels = []
    global_items = []
    for item, label_to_conf in item_to_label_to_conf.iteritems():
        for label, conf in label_to_conf.iteritems():
            if restrict_to_labels and label not in restrict_to_labels:
                continue
            global_assigneds.append(
                label in item_to_labels[item]
            )
            global_scores.append(conf)
            global_labels.append(list(label)[0])
            global_items.append(item)

    scores_w_classes_w_labels_w_items = zip(global_scores, global_assigneds, global_labels, global_items)
    scores_w_classes_w_labels_w_items = sorted(scores_w_classes_w_labels_w_items, key=lambda x: x[0])
    print "Lowest global scores and truth value:"
    with open('weird.json', 'w') as f:
        print json.dump(scores_w_classes_w_labels_w_items, f, indent=True)
        

    global_precisions, global_recalls, global_threshs = _precision_recall_curve(global_assigneds, global_scores)
    global_pr_curve = (global_precisions, global_recalls)
    global_avg_precision = _average_precision(global_precisions, global_recalls)

    return (
        label_to_auc, 
        label_to_avg_precision, 
        label_to_pr_curve,
        global_pr_curve,
        global_avg_precision
    )


def _compute_conservative_assignment_and_score_lists(
        confidence_df,
        assignment_df,
        label_graph,
        label_to_name,
        og,
        blacklist_labels=None
    ):
    """
    Compute the PR-curve metrics for each label, BUT exclude from each
    label's pool of predictions those samples samples for which one of
    the sample's most-specific labels is an ancestor of the current
    label.
    """
    # Instantiate list of blacklisted labels
    if blacklist_labels is None:
        blacklist_labels = set()

    # Map each item to its set of annotated labels
    item_to_labels = {
        item: [
            label
            for label in assignment_df.columns
            if assignment_df.loc[item][label]
        ]
        for item in assignment_df.index
    }

    # Map each item to its most-specific labels
    print "Mapping items to their most-specific labels..."
    all_labels = set(label_graph.get_all_nodes())
    item_to_ms_labels = {}
    for item, labels in item_to_labels.iteritems():
        ms_item_labels = label_graph.most_specific_nodes(
            set(labels) & all_labels
        )
        ms_item_labels = ms_item_labels - blacklist_labels
        item_to_ms_labels[item] = ms_item_labels
    print "done."

    # Create new nodes in the label graph corresponding to 
    # joint-labels -- that is, sets of labels for which there
    # exists a sample labelled with both labels. For example,
    # if an experiment is labelled with both 'PBMC' and
    # 'T cell', then we create a new label 'PBMC & T cell'
    mod_label_graph = label_graph.copy()
    mod_label_to_name = {
        label: name
        for label, name in label_to_name.iteritems()
    }

    # Create all joint-nodes
    item_to_new_ms_labels = defaultdict(lambda: set())
    for item, ms_labels in item_to_ms_labels.iteritems():
        # Create a joint label
        if len(ms_labels) > 1:
            joint_label = frozenset(ms_labels)
            mod_label_to_name[joint_label] = " & ".join([
                mod_label_to_name[ms_label]
                for ms_label in ms_labels
            ])
            item_to_new_ms_labels[item].add(joint_label)
            # Create from the joint label to the labels that 
            # it includes.
            for ms_label in ms_labels:
                mod_label_graph.add_edge(ms_label, joint_label)
            print "Created joint label '%s' (%s)" % (
                mod_label_to_name[joint_label],
                joint_label
            )

    # Make a 'deep' copy of the mappings from experiments to most-specific 
    # labels. Then recompute the most-specific labels and predictions now 
    # that we have added these new join-labels
    mod_item_to_ms_labels = {
        item: set(ms_labels)
        for item, ms_labels in item_to_ms_labels.iteritems()
    }
    for item, new_ms_labels in item_to_new_ms_labels.iteritems():
        mod_item_to_ms_labels[item].update(new_ms_labels)
    mod_item_to_ms_labels = {
        item: mod_label_graph.most_specific_nodes(labels)
        for item, labels in mod_item_to_ms_labels.iteritems()
    }

    # If the sample is most-specifically labeled as PBMC, then
    # for our purposes, we treat mononuclear cell as its most
    # specific label 
    item_to_ms_labels = mod_item_to_ms_labels
    for item, ms_labels in item_to_ms_labels.iteritems():
        if PBMC_TERM_ID in ms_labels:
            ms_labels.add(
                MONONUCLEAR_CELL_TERM_ID
            ) 

    # For each item, get all of the ancestors of all descendants
    # of it's most-specific labels
    item_to_anc_desc_ms_labels = {}
    for item, ms_labels in item_to_ms_labels.iteritems():
        desc_ms_labels = set()
        for ms_label in ms_labels:
            desc_ms_labels.update(
                mod_label_graph.descendent_nodes(ms_label)
            )
        anc_desc_ms_labels = set()
        for desc_ms_label in desc_ms_labels:
            anc_desc_ms_labels.update(
                mod_label_graph.ancestor_nodes(desc_ms_label)
            )
        # Make sure that the item's labels are not included in
        # this set
        anc_desc_ms_labels = anc_desc_ms_labels - set(item_to_labels[item])
        item_to_anc_desc_ms_labels[item] = anc_desc_ms_labels

    # Iterate over all labels and construct the list of assignment-values 
    # (True or False) and classifier-produced confidence-scores for only the 
    # set of items that are relevant for computing the conservative-metrics
    # for the label
    skipped_pairs = set()
    pair_to_items = defaultdict(lambda: set())
    label_to_cons_assigneds = {}
    label_to_cons_scores = {}
    for curr_label in set(all_labels) & set(assignment_df.columns):
        print "Examining label %s" % og.id_to_term[curr_label].name

        # Assignment-values for this label
        filtered_assignments = []

        # Classifier-scores for this label
        filtered_scores = []

        # The set of items not considered for this label
        skipped_items = set()

        # Ancestors of the current label
        anc_labels = set(mod_label_graph.ancestor_nodes(curr_label)) - set([curr_label])

        # Iterate over each item and determine whether it should be included
        # in the computation of curr_label's metrics
        for item in assignment_df.index:
            assigned = assignment_df.loc[item][curr_label]
            score = confidence_df.loc[item][curr_label]            
            ms_labels = item_to_ms_labels[item]
            anc_desc_ms_labels = item_to_anc_desc_ms_labels[item]
            # NOTE this is the crucial step in which we skip
            # samples that have a most-specific label that is
            # an ancestor of the current label or an ancestor
            # of a descendent of the current label
            if len(set(ms_labels) & anc_labels) > 0:
                for ms_label in set(ms_labels) & set(anc_labels):
                    pair = (ms_label, curr_label)
                    skipped_pairs.add(pair)
                    pair_to_items[pair].add(item)
                skipped_items.add(item)
                continue
            if curr_label in anc_desc_ms_labels:
                skipped_items.add(item)
                continue
            filtered_assignments.append(assigned)
            filtered_scores.append(score)
        label_to_cons_assigneds[curr_label] = filtered_assignments
        label_to_cons_scores[curr_label] = filtered_scores
        print "Label %s" % label_to_name[curr_label]
        print "N samples in ranking: %d" % len(filtered_assignments)
        print "N skipped: %d" % len(skipped_items)
        print "Sample of skipped %s" % list(skipped_items)[0:20]
        print 
    label_to_assigneds = dict(label_to_cons_assigneds)
    label_to_scores = dict(label_to_cons_scores) 

    # Print some data on which samples were filtered from this analysis
    filtering_da = [
        (
            og.id_to_term[pair[0]].name, 
            og.id_to_term[pair[1]].name, 
            len(pair_to_items[pair])
        )
        for pair in skipped_pairs
    ]
    filtering_df = pd.DataFrame(
        data = filtering_da,
        columns = ['', '', 'Number of samples removed'] # TODO
    )
    print 'Computation of conservative metrics:'
    print filtering_df
    return label_to_assigneds, label_to_scores


def conservative_joint_metrics(): # TODO REFACTOR THIS!!!!!!!!!!!!!!!
    for curr_label in all_labels:
        global_assigneds.append(
            curr_label in item_to_labels[item]
        )
        global_scores.append(score)
    global_precisions, global_recalls, global_threshs = _precision_recall_curve(
        global_assigneds, 
        global_scores
    )
    global_pr_curve = (global_precisions, global_recalls)
    global_avg_precision = _average_precision(global_precisions, global_recalls)
    return (
        label_to_auc,
        label_to_avg_precision,
        label_to_pr_curve,
        global_pr_curve,
        global_avg_precision
    )



def compute_per_sample_metrics(
        sample_to_label_to_conf,
        sample_to_labels,
        label_graph,
        sample_to_study,
        study_to_samples,
        label_to_name=None,
        restrict_to_labels=None,
        og=None,
        thresh_range='zero_to_one'
    ):

    # Compute the average per-sample metrics
    #for pred_thresh in np.arange()
    precisions = []
    recalls = []
    sw_precisions = []
    sw_recalls = []
    ms_precisions = []
    ms_recalls = []
    sw_ms_precisions = []
    sw_ms_recalls = []
    if thresh_range == 'zero_to_one':
        pred_threshes = np.concatenate((
            np.arange(0.0, 0.1, 0.0001),
            np.arange(0.1, 0.9, 0.05),
            np.arange(0.9, 0.99, 0.01),
            np.arange(0.99, 0.999, 0.001),
            np.arange(0.999, 0.9999, 0.0001),
            np.arange(0.9999, 0.99999, 0.00001),
            np.array([0.99999999]),
            np.array([0.99999999999]),
            np.array([1.0]),
            np.array([1.1]) # To get the PR-point at P=1, R=0
        ))
    elif thresh_range == '1nn':
        pred_threshes = np.concatenate((
            np.array([float('-inf')]),
            np.arange(-2.0, 2.0, 0.01)
        ))
    elif thresh_range == 'data_driven':
        max_conf = None
        min_conf = None
        for sample, label_to_conf in sample_to_label_to_conf.iteritems():
            curr_max = max(label_to_conf.values())
            curr_min = min(label_to_conf.values())
            if not max_conf or curr_max > max_conf:
                max_conf = curr_max
            if not min_conf or curr_min < min_conf:
                min_conf = curr_min
        pred_threshes = np.array([min_conf, max_conf, 0.05])
    print "Pred threshes: %s" % pred_threshes

    # Determine which samples have no true labels. Also, precompute
    # the most-specific labels for each sample
    skipped_samples = set()
    sample_to_ms_true_pos_labels = {}
    for sample in sample_to_label_to_conf:
        true_pos_labels = set(sample_to_labels[sample])
        if restrict_to_labels:
            true_pos_labels &= restrict_to_labels
        if len(true_pos_labels) == 0:
            skipped_samples.add(sample)
        ms_true_pos_labels = label_graph.most_specific_nodes(true_pos_labels)
        if restrict_to_labels:
            ms_true_pos_labels &= restrict_to_labels
        sample_to_ms_true_pos_labels[sample] = ms_true_pos_labels
    print "%d samples have no true labels." % len(skipped_samples)

    for pred_thresh in pred_threshes:
        all_studies = set()
        n_samples = 0
        avg_sample_prec_sum = 0.0
        avg_sample_recall_sum = 0.0
        avg_sw_sample_prec_sum = 0.0
        avg_sw_sample_recall_sum = 0.0
        avg_sample_ms_prec_sum = 0.0
        avg_sample_ms_recall_sum = 0.0
        avg_sw_sample_ms_prec_sum = 0.0
        avg_sw_sample_ms_recall_sum = 0.0

        for sample, label_to_conf in sample_to_label_to_conf.iteritems():
            if sample in skipped_samples:
                continue
            n_samples += 1.0

            pred_pos_labels = set([
                label
                for label, conf in label_to_conf.iteritems()
                if conf >= pred_thresh
            ])
            ms_pred_pos_labels = label_graph.most_specific_nodes(pred_pos_labels)
            true_pos_labels = set(sample_to_labels[sample])
            ms_true_pos_labels = sample_to_ms_true_pos_labels[sample]
            if restrict_to_labels:
                true_pos_labels &= restrict_to_labels
                pred_pos_labels &= restrict_to_labels
                ms_pred_pos_labels &= restrict_to_labels
            study = sample_to_study[sample]
            study_weight = 1.0 / len(study_to_samples[study] - skipped_samples)
            all_studies.add(study)
            if len(pred_pos_labels) > 0:
                sample_prec = len(pred_pos_labels & true_pos_labels) / float(len(pred_pos_labels))
            else:
                sample_prec = 1.0
            sw_sample_prec = sample_prec * study_weight
            avg_sample_prec_sum += sample_prec
            avg_sw_sample_prec_sum += sw_sample_prec

            if len(true_pos_labels) > 0:
                sample_recall = len(pred_pos_labels & true_pos_labels) / float(len(true_pos_labels))
            else:
                sample_recall = 1.0
            sw_sample_recall = sample_recall * study_weight
            avg_sample_recall_sum += sample_recall
            avg_sw_sample_recall_sum += sw_sample_recall

            if len(ms_pred_pos_labels) > 0:
                sample_ms_prec = len(ms_pred_pos_labels & true_pos_labels) / float(len(ms_pred_pos_labels))
            else:
                sample_ms_prec = 1.0
            sw_sample_ms_prec = sample_ms_prec * study_weight 
            avg_sample_ms_prec_sum += sample_ms_prec
            avg_sw_sample_ms_prec_sum += sw_sample_ms_prec

            if len(ms_true_pos_labels) > 0:
                sample_ms_recall = len(pred_pos_labels & ms_true_pos_labels) / float(len(ms_true_pos_labels))
            else:
                sample_ms_recall = 1.0
            sw_sample_ms_recall = sample_ms_recall * study_weight
            avg_sample_ms_recall_sum += sample_ms_recall
            avg_sw_sample_ms_recall_sum += sw_sample_ms_recall


            #if pred_thresh == 0.9:
            #    print "Sample: %s" % sample
            #    print "True labels: %s" % [label_to_name[x] for x in true_pos_labels]
            #    print "Predicted labels: %s" % [label_to_name[x] for x in pred_pos_labels]
            #    print "Most-specific true labels: %s" % [label_to_name[x] for x in ms_true_pos_labels]
            #    print "Most-specific predicted labels: %s" % [label_to_name[x] for x in ms_pred_pos_labels] 
            #    print "Precision: %f" % sample_prec
            #    print "Recall: %f" % sample_recall
            #    print "Specific terms precision: %f" % sample_ms_prec
            #    print "Specific terms recall: %f" % sample_ms_recall
            #    print

        n_studies = len(all_studies)
        avg_sample_prec = avg_sample_prec_sum / n_samples 
        avg_sample_recall = avg_sample_recall_sum  / n_samples
        avg_sw_sample_prec = avg_sw_sample_prec_sum / n_studies
        avg_sw_sample_recall = avg_sw_sample_recall_sum / n_studies
        avg_sample_ms_prec = avg_sample_ms_prec_sum / n_samples
        avg_sample_ms_recall = avg_sample_ms_recall_sum / n_samples
        avg_sw_sample_ms_prec = avg_sw_sample_ms_prec_sum / n_studies
        avg_sw_sample_ms_recall = avg_sw_sample_ms_recall_sum / n_studies
       
        print "Threshold is %f. Prec: %f. Recall: %f. Ms-prec: %f. Ms-recall: %f" % (
            pred_thresh,
            avg_sample_prec,
            avg_sample_recall,
            avg_sample_ms_prec,
            avg_sample_ms_recall
        )
        precisions.append(avg_sample_prec)
        recalls.append(avg_sample_recall)
        sw_precisions.append(avg_sw_sample_prec)
        sw_recalls.append(avg_sw_sample_recall)
        ms_precisions.append(avg_sample_ms_prec)
        ms_recalls.append(avg_sample_ms_recall)
        sw_ms_precisions.append(avg_sw_sample_ms_prec)
        sw_ms_recalls.append(avg_sw_sample_ms_recall)
    return (
        None,   # TODO REMOVE THIS
        None,   # TODO REMOVE THIS
        (precisions, recalls),
        (sw_precisions, sw_recalls),
        (ms_precisions, ms_recalls),
        (sw_ms_precisions, sw_ms_recalls),
        pred_threshes
    )


def compute_conservative_per_sample_metrics(
        sample_to_label_to_conf,
        sample_to_labels,
        label_graph,
        sample_to_study,
        study_to_samples,
        label_to_name=None,
        restrict_to_labels=None,
        og=None,
        thresh_range='zero_to_one'
    ):

    # Compute the average per-sample metrics
    #for pred_thresh in np.arange()
    precisions = []
    recalls = []
    sw_precisions = []
    sw_recalls = []
    ms_precisions = []
    ms_recalls = []
    sw_ms_precisions = []
    sw_ms_recalls = []
    if thresh_range == 'zero_to_one':
        pred_threshes = np.concatenate((
            np.arange(0.0, 0.1, 0.01),
            np.arange(0.1, 0.9, 0.05),
            np.arange(0.9, 0.99, 0.01),
            np.arange(0.99, 0.999, 0.001),
            np.arange(0.999, 0.9999, 0.0001),
            np.arange(0.9999, 0.99999, 0.00001),
            np.array([0.99999999]),
            np.array([0.99999999999]),
            np.array([1.0]),
            np.array([1.1]) # To get the PR-point at P=1, R=0
        ))
    elif thresh_range == '1nn':
        pred_threshes = np.concatenate((
            np.array([float('-inf')]),
            np.arange(-2.0, 2.0, 0.01)
        ))
    elif thresh_range == 'data_driven':
        max_conf = None
        min_conf = None
        for sample, label_to_conf in sample_to_label_to_conf.iteritems():
            curr_max = max(label_to_conf.values())
            curr_min = min(label_to_conf.values())
            if not max_conf or curr_max > max_conf:
                max_conf = curr_max
            if not min_conf or curr_min < min_conf:
                min_conf = curr_min
        pred_threshes = np.array([min_conf, max_conf, 0.05])
    print "Pred threshes: %s" % pred_threshes

    # Determine which samples have no true labels. Also, precompute
    # the most-specific labels for each sample
    skipped_samples = set()
    sample_to_ms_true_pos_labels = {}
    sample_to_ignore_labels = {}
    for sample in sample_to_label_to_conf:
        true_pos_labels = set(sample_to_labels[sample])
        if restrict_to_labels:
            true_pos_labels &= restrict_to_labels
        if len(true_pos_labels) == 0:
            skipped_samples.add(sample)
        ms_true_pos_labels = label_graph.most_specific_nodes(true_pos_labels)
        if restrict_to_labels:
            ms_true_pos_labels &= restrict_to_labels
        sample_to_ms_true_pos_labels[sample] = ms_true_pos_labels

        # Precompute the labels that are "ambiguous". That is, labels that are
        # are more-specific than the most-specific for the sample and ancestors
        # of these terms (that aren't in the true labels set).
        desc_ms_labels = set()
        for ms_label in ms_true_pos_labels:
            desc_ms_labels.update(
                label_graph.descendent_nodes(ms_label)
            )
        anc_desc_ms_labels = set()
        for desc_ms_label in desc_ms_labels:
            anc_desc_ms_labels.update(
                label_graph.ancestor_nodes(desc_ms_label)
            )
        ignore_labels = anc_desc_ms_labels - true_pos_labels
        sample_to_ignore_labels[sample] = ignore_labels    

    print "%d samples have no true labels." % len(skipped_samples)

    # REMOVEEEEEEEEEEEE
    #samples_w_ignore = [
    #    sample
    #    for sample, ignore in sample_to_ignore_labels.iteritems() 
    #    if len(ignore) > 2
    #]
    #print "Samples and ignore labels"
    #for sample in samples_w_ignore[:10]:
    #    print "%s\t%s" % (sample, [label_to_name[x] for x in sample_to_ignore_labels[sample]])
    #print "%d samples have labels to ignore" % len(samples_w_ignore)
    # REMOVEEEEE

    for pred_i, pred_thresh in enumerate(pred_threshes):
        all_studies = set()
        n_samples = 0
        avg_sample_prec_sum = 0.0
        avg_sample_recall_sum = 0.0
        avg_sw_sample_prec_sum = 0.0
        avg_sw_sample_recall_sum = 0.0
        avg_sample_ms_prec_sum = 0.0
        avg_sample_ms_recall_sum = 0.0
        avg_sw_sample_ms_prec_sum = 0.0
        avg_sw_sample_ms_recall_sum = 0.0

        for sample, label_to_conf in sample_to_label_to_conf.iteritems():
            if sample in skipped_samples:
                continue
            n_samples += 1.0

            pred_pos_labels = set([
                label
                for label, conf in label_to_conf.iteritems()
                if conf >= pred_thresh
            ])
            ignore_labels = sample_to_ignore_labels[sample]
            pred_pos_labels -= ignore_labels
            ms_pred_pos_labels = label_graph.most_specific_nodes(pred_pos_labels)
            true_pos_labels = set(sample_to_labels[sample])
            ms_true_pos_labels = sample_to_ms_true_pos_labels[sample]
            if restrict_to_labels:
                true_pos_labels &= restrict_to_labels
                pred_pos_labels &= restrict_to_labels
                ms_pred_pos_labels &= restrict_to_labels
            #pred_pos_labels -= ignore_labels
            #ms_pred_pos_labels -= ignore_labels
            
            study = sample_to_study[sample]
            study_weight = 1.0 / len(study_to_samples[study] - skipped_samples)
            all_studies.add(study)
            if len(pred_pos_labels) > 0:
                sample_prec = len(pred_pos_labels & true_pos_labels) / float(len(pred_pos_labels))
            else:
                sample_prec = 1.0
            sw_sample_prec = sample_prec * study_weight
            avg_sample_prec_sum += sample_prec
            avg_sw_sample_prec_sum += sw_sample_prec

            if len(true_pos_labels) > 0:
                sample_recall = len(pred_pos_labels & true_pos_labels) / float(len(true_pos_labels))
            else:
                sample_recall = 1.0
            sw_sample_recall = sample_recall * study_weight
            avg_sample_recall_sum += sample_recall
            avg_sw_sample_recall_sum += sw_sample_recall

            if len(ms_pred_pos_labels) > 0:
                sample_ms_prec = len(ms_pred_pos_labels & true_pos_labels) / float(len(ms_pred_pos_labels))
            else:
                sample_ms_prec = 1.0
            sw_sample_ms_prec = sample_ms_prec * study_weight
            avg_sample_ms_prec_sum += sample_ms_prec
            avg_sw_sample_ms_prec_sum += sw_sample_ms_prec

            if len(ms_true_pos_labels) > 0:
                sample_ms_recall = len(pred_pos_labels & ms_true_pos_labels) / float(len(ms_true_pos_labels))
            else:
                sample_ms_recall = 1.0
            sw_sample_ms_recall = sample_ms_recall * study_weight
            avg_sample_ms_recall_sum += sample_ms_recall
            avg_sw_sample_ms_recall_sum += sw_sample_ms_recall


            if pred_thresh == 0.999400:
                print "Sample: %s" % sample
                print "True labels: %s" % [label_to_name[x] for x in true_pos_labels]
                print "Predicted labels: %s" % [label_to_name[x] for x in pred_pos_labels]
                print "Most-specific true labels: %s" % [label_to_name[x] for x in ms_true_pos_labels]
                print "Most-specific predicted labels: %s" % [label_to_name[x] for x in ms_pred_pos_labels] 
                print "Precision: %f" % sample_prec
                print "Recall: %f" % sample_recall
                print "Specific terms precision: %f" % sample_ms_prec
                print "Specific terms recall: %f" % sample_ms_recall
                print

        n_studies = len(all_studies)
        avg_sample_prec = avg_sample_prec_sum / n_samples
        avg_sample_recall = avg_sample_recall_sum  / n_samples
        avg_sw_sample_prec = avg_sw_sample_prec_sum / n_studies
        avg_sw_sample_recall = avg_sw_sample_recall_sum / n_studies
        avg_sample_ms_prec = avg_sample_ms_prec_sum / n_samples
        avg_sample_ms_recall = avg_sample_ms_recall_sum / n_samples
        avg_sw_sample_ms_prec = avg_sw_sample_ms_prec_sum / n_studies
        avg_sw_sample_ms_recall = avg_sw_sample_ms_recall_sum / n_studies

        print "Threshold is %f. Prec: %f. Recall: %f. Ms-prec: %f. Ms-recall: %f" % (
            pred_thresh,
            avg_sample_prec,
            avg_sample_recall,
            avg_sample_ms_prec,
            avg_sample_ms_recall
        )
        precisions.append(avg_sample_prec)
        recalls.append(avg_sample_recall)
        sw_precisions.append(avg_sw_sample_prec)
        sw_recalls.append(avg_sw_sample_recall)
        ms_precisions.append(avg_sample_ms_prec)
        ms_recalls.append(avg_sample_ms_recall)
        sw_ms_precisions.append(avg_sw_sample_ms_prec)
        sw_ms_recalls.append(avg_sw_sample_ms_recall)
    return (
        None,   # TODO REMOVE THIS
        None,   # TODO REMOVE THIS
        (precisions, recalls),
        (sw_precisions, sw_recalls),
        (ms_precisions, ms_recalls),
        (sw_ms_precisions, sw_ms_recalls)
    )


def compute_metrics_hard_thresh(
        exp_to_label_to_confidence,
        exp_to_labels,
        restrict_to_labels=None,
        thresh=0.5
    ):
    exp_order = sorted(exp_to_label_to_confidence.keys())
    label_to_prec = {}
    label_to_rec = {}
    for label in restrict_to_labels:
        true = [
            label in exp_to_labels[exp]
            for exp in exp_order
        ]
        pred = [
            exp_to_label_to_confidence[exp][label] > thresh
            for exp in exp_order
        ]
        if len(set(true)) == 1 and list(set(true))[0] == False:
            # There are no true positives
            continue
        prec = _precision([x for x in zip(pred, true)])
        rec = _recall([x for x in zip(pred, true)])
        label_to_prec[label] = prec
        label_to_rec[label] = rec
    return label_to_prec, label_to_rec
   
 
def _precision(preds_w_trues):
    true_pos = float(len([
        x
        for x in preds_w_trues
        if x[1] and x[0]              
    ]))
    pred_pos = float(len([
        x
        for x in preds_w_trues
        if x[0]
    ]))
    if pred_pos == 0:
        return 0.0
    else:
        return true_pos / pred_pos
    

def _recall(preds_w_trues):
    true_pos = float(len([
        x
        for x in preds_w_trues
        if x[1] and x[0]              
    ]))
    all_pos = float(len([
        x
        for x in preds_w_trues
        if x[1]
    ]))
    return true_pos / all_pos
    

def _precision_recall_curve(classes, scores):
    scores_w_classes = zip(scores, classes)
    scores_w_classes = sorted(scores_w_classes, key=lambda x: x[0])
    curr_score = None
    precisions = []
    recalls = []
    threshs = []
    total_pos = float(len([x for x in scores_w_classes if x[1]]))
    pos_seen = 0.0
    last_score = None
    for i, score_w_class in enumerate(scores_w_classes[:-1]):
        curr_score = score_w_class[0]
        if score_w_class[1]: # Positive instance
            pos_seen += 1.0
        if last_score is None or curr_score != last_score:
            prec = (total_pos  - pos_seen) / float(len(scores_w_classes) - (i+1))
            recall = (total_pos  - pos_seen) / total_pos
            precisions.append(prec)
            recalls.append(recall)
            threshs.append(curr_score)
        last_score = curr_score
    precisions.reverse()
    recalls.reverse()
    threshs.reverse()
    return precisions, recalls, threshs


def _average_precision(precisions, recalls):
    summ = 0.0
    if len(recalls) > 0 and recalls[0] != 0.0:
        precisions = [0.0] + precisions
        recalls = [0.0] + recalls
    for i in range(1, len(recalls)):
        recall_prev = recalls[i-1]
        recall = recalls[i]
        prec = precisions[i]
        summ += (recall - recall_prev)*prec
    return summ


   
    
def compute_included_labels(
        the_exps,
        exp_to_labels,
        exp_to_info,
        og
    ):
    label_to_num_studies = data_metrics(
        the_exps,
        exp_to_labels,
        exp_to_info
    )
    include_labels = set([
        label
        for label, n_studies in label_to_num_studies.iteritems()
        if n_studies > 1
    ])

    print "Label to number of studies:"
    print label_to_num_studies

    # Compute the trivial labels
    trivial_labels = set(
        exp_to_labels[
            list(exp_to_labels.keys())[0]
        ]
    )
    for labels in exp_to_labels.values():
        trivial_labels = trivial_labels & set(labels)
    print "Computed the following trivial labels: %s" % trivial_labels

    include_labels -= trivial_labels
    return include_labels



















def compute_per_sample_performance(
        sample_to_predictions,
        sample_to_labels
    ):
    sample_to_false_positives = defaultdict(lambda: set())
    sample_to_false_negatives = defaultdict(lambda: set())
    sample_to_hamming_loss = {}
    for sample, predictions in sample_to_predictions.iteritems():
        predictions = set(predictions)
        labels = set(sample_to_labels[sample])
        false_positives = predictions - labels
        false_negatives = labels - predictions
        true_positives = labels & predictions
        hamming_loss = len(false_positives) + len(false_negatives)

        sample_to_false_positives[sample] = false_positives
        sample_to_false_negatives[sample] = false_negatives
        sample_to_hamming_loss[sample] = hamming_loss
    return sample_to_false_positives, sample_to_false_negatives, sample_to_hamming_loss


def per_label_recall(
        sample_to_predictions,
        sample_to_labels,
        sample_to_weight=None
    ):

    if not sample_to_weight:
        sample_to_weight = defaultdict(lambda: 1)

    label_to_predicted_samples = defaultdict(lambda: set())
    for sample, predictions in sample_to_predictions.iteritems():
        for label in predictions:
            label_to_predicted_samples[label].add(sample)

    label_to_relavent_samples = defaultdict(lambda: set())
    for sample, labels in sample_to_labels.iteritems():
        for label in labels:
            label_to_relavent_samples[label].add(sample)

    label_to_recall = {}
    for label in label_to_relavent_samples:
        rel_samples = label_to_relavent_samples[label]
        pred_samples = label_to_predicted_samples[label]
        true_positives = rel_samples & pred_samples
        num = float(sum([
            sample_to_weight[x]
            for x in true_positives
        ]))
        den = float(sum([
            sample_to_weight[x]
            for x in rel_samples
        ]))
        recall = num/den
        label_to_recall[label] = recall

    return label_to_recall


def per_label_precision(
        sample_to_predictions,
        sample_to_labels,
        label_to_sample_to_weight=None
    ):

    if not label_to_sample_to_weight:
        label_to_sample_to_weight = defaultdict(lambda: defaultdict(lambda: 1))

    label_to_predicted_samples = defaultdict(lambda: set())
    for sample, predictions in sample_to_predictions.iteritems():
        for label in predictions:
            label_to_predicted_samples[label].add(sample)

    label_to_relavent_samples = defaultdict(lambda: set())
    for sample, labels in sample_to_labels.iteritems():
        for label in labels:
            label_to_relavent_samples[label].add(sample)

    label_to_precision = {}
    for label in label_to_relavent_samples:
        rel_samples = label_to_relavent_samples[label]
        pred_samples = label_to_predicted_samples[label]
        true_positives = rel_samples & pred_samples
        num = float(sum([
            label_to_sample_to_weight[label][x]
            for x in true_positives
        ]))
        den = float(sum([
            label_to_sample_to_weight[label][x]
            for x in pred_samples
        ]))
        if den == 0:
            precision = 0.0
        else:
            precision = num/den
        label_to_precision[label] = precision

    return label_to_precision





if __name__ == "__main__":
    main()


