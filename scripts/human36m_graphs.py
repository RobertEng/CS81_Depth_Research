import pickle
import os
import json
import sys
import itertools
import copy
import difflib
import random
from collections import Counter, defaultdict
import numpy as np
from lean_correction import correct_lean

import matplotlib.pyplot as plt

from constants import (HUMAN_ANNOTATION_PATH, COCO_ANNOTATION_PATH,
    HUMAN_RAW_RESULT_PATH, COCO_RAW_RESULT_PATH, KEYPTS_RELATIVE_DEPTH_PATH,
    KEYCMPS_RESULT_PATH, HUMAN_OUTPUT_PATH)


def fraction_of_correct_comparisons_per_trial(data, _human_dataset, subj_id_to_yvals, plots=True):
    fractions_of_correct_comps_per_trial = []
    for d in data:
        for t in d['trials']:
            subj_id = t['human_subj_id']
            yvals = subj_id_to_yvals[subj_id]
            trial_comp_res = []
            for comp, comp_res in t['depth']['keypoint_comparisons_res'].iteritems():
                kpt1, kpt2 = [int(_k) for _k in comp.split(',')]
                kpt1_depth = yvals[kpt1]
                kpt2_depth = yvals[kpt2]
                depth_diff = kpt1_depth - kpt2_depth
                trial_comp_res.append(int(np.sign(depth_diff) == np.sign(comp_res)))
            fractions_of_correct_comps_per_trial.append(sum(trial_comp_res) * 1.0 / len(trial_comp_res))
    
    if plots:
        plt.figure(random.randint(0, 1000))
        bins = np.arange(-0.1, 1.1, 0.2)
        n, bins, patches = plt.hist(fractions_of_correct_comps_per_trial, bins, facecolor='green')
        plt.figure(random.randint(0, 1000))
        n = np.array([_i * 1.0 / sum(n) for _i in n])
        
        labels = np.arange(0.0, 1.1, 0.2)
        pos = np.arange(len(labels)) / 5.0
        width = 0.2     # gives histogram aspect to the bar diagram
        ax = plt.axes()
        ax.set_xticks(pos)
        ax.set_xticklabels(labels)
        plt.bar(pos, n, width, color='r', linewidth=1, edgecolor='black')
        plt.xlabel('Correct / Total Comparisons per Trial')
        plt.ylabel('Proportion of Trials')
        plt.title('Fraction of Correct Comparisons per Trial')
        ax.yaxis.grid(True)
        plt.show()

    return fractions_of_correct_comps_per_trial

def fraction_of_correct_comparisons_per_assignment(data, _human_dataset, subj_id_to_yvals, plots=True):
    fractions_of_correct_comps_per_assignment = []

    for d in data:
        assignment_comp_res = []
        for t in d['trials']:
            subj_id = t['human_subj_id']
            yvals = subj_id_to_yvals[subj_id]
            for comp, comp_res in t['depth']['keypoint_comparisons_res'].iteritems():
                kpt1, kpt2 = [int(_k) for _k in comp.split(',')]
                kpt1_depth = yvals[kpt1]
                kpt2_depth = yvals[kpt2]
                depth_diff = kpt1_depth - kpt2_depth
                assignment_comp_res.append(int(np.sign(depth_diff) == np.sign(comp_res)))
        fractions_of_correct_comps_per_assignment.append(sum(assignment_comp_res) * 1.0 / len(assignment_comp_res))

    if plots:
        plt.figure(random.randint(0, 1000))
        bins = np.arange(0.0, 1.1, 0.2)
        n, bins, patches = plt.hist(fractions_of_correct_comps_per_assignment, bins, facecolor='green')
        plt.figure(random.randint(0, 1000))
        n = np.array([_i * 1.0 / sum(n) for _i in n])

        labels = np.arange(0.0, 1.0, 0.2)
        pos = np.arange(len(labels)) / 5.0
        width = 0.2     # gives histogram aspect to the bar diagram
        ax = plt.axes()
        ax.set_xticks(pos)
        ax.set_xticklabels(labels)
        plt.bar(pos, n, width, color='r', align='edge')
        plt.xlabel('Correct / Total Comparisons per Assignment')
        plt.ylabel('Proportion of Assignments')
        plt.title('Fraction of Correct Comparisons per Assignment')
        plt.grid(True)
        plt.show()

    return fractions_of_correct_comps_per_assignment

def fraction_of_correct_comparisons_per_subject(data, _human_dataset, subj_id_to_yvals, subj_id_to_trials, plots=True):
    fractions_of_correct_comps_per_subject = []

    for subj_id in subj_id_to_trials.keys():
        subj_comp_res = []
        trials = subj_id_to_trials[subj_id]
        yvals = subj_id_to_yvals[subj_id]
        for t in trials:
            for comp, comp_res in t.iteritems():
                kpt1, kpt2 = [int(_k) for _k in comp.split(',')]
                kpt1_depth = yvals[kpt1]
                kpt2_depth = yvals[kpt2]
                depth_diff = kpt1_depth - kpt2_depth
                subj_comp_res.append(int(np.sign(depth_diff) == np.sign(comp_res)))
        fractions_of_correct_comps_per_subject.append(sum(subj_comp_res) * 1.0 / len(subj_comp_res))

    if plots:
        plt.figure(random.randint(0, 1000))
        bins = np.arange(0.0, 1.1, 0.2)
        n, bins, patches = plt.hist(fractions_of_correct_comps_per_subject, bins, facecolor='green')
        plt.figure(random.randint(0, 1000))
        n = np.array([_i * 1.0 / sum(n) for _i in n])
        
        labels = np.arange(0.0, 1.0, 0.2)
        pos = np.arange(len(labels)) / 5.0
        width = 0.2     # gives histogram aspect to the bar diagram
        ax = plt.axes()
        ax.set_xticks(pos)
        ax.set_xticklabels(labels)
        plt.bar(pos, n, width, color='r', align='edge')
        plt.xlabel('Correct / Total Comparisons per Subject')
        plt.ylabel('Proportion of Subjects')
        plt.title('Fraction of Correct Comparisons per Subject')
        plt.grid(True)
        plt.show()        

    return fractions_of_correct_comps_per_subject

def fraction_of_correct_comparisons_per_hit(data, _human_dataset, subj_id_to_yvals, hit_id_to_trials, plots=True):
    fractions_of_correct_comps_per_hit = []

    for hit_id in hit_id_to_trials.keys():
        hit_comp_res = []
        trials = hit_id_to_trials[hit_id]
        for trial in trials:
            for t in trial:
                subj_id = t['human_subj_id']
                yvals = subj_id_to_yvals[subj_id]
                for comp, comp_res in t['depth']['keypoint_comparisons_res'].iteritems():
                    kpt1, kpt2 = [int(_k) for _k in comp.split(',')]
                    kpt1_depth = yvals[kpt1]
                    kpt2_depth = yvals[kpt2]
                    depth_diff = kpt1_depth - kpt2_depth
                    hit_comp_res.append(int(np.sign(depth_diff) == np.sign(comp_res)))
        fractions_of_correct_comps_per_hit.append(sum(hit_comp_res) * 1.0 / len(hit_comp_res))

    if plots:
        plt.figure(random.randint(0, 1000))
        bins = np.arange(0.0, 1.1, 0.2)
        n, bins, patches = plt.hist(fractions_of_correct_comps_per_hit, bins, facecolor='green')
        plt.figure(random.randint(0, 1000))
        n = np.array([_i * 1.0 / sum(n) for _i in n])
        
        labels = np.arange(0.0, 1.0, 0.2)
        pos = np.arange(len(labels)) / 5.0
        width = 0.2     # gives histogram aspect to the bar diagram
        ax = plt.axes()
        ax.set_xticks(pos)
        ax.set_xticklabels(labels)
        plt.bar(pos, n, width, color='r', align='edge')
        plt.xlabel('Correct / Total Comparisons per HIT')
        plt.ylabel('Proportion of HITs')
        plt.title('Fraction of Correct Comparisons per HIT')
        plt.grid(True)
        plt.show()        

    return fractions_of_correct_comps_per_hit

def fraction_of_correct_comparisons_per_worker(data, _human_dataset, subj_id_to_yvals, worker_id_to_trials, plots=True):
    fractions_of_correct_comps_per_worker = []

    for worker_id in worker_id_to_trials.keys():
        worker_comp_res = []
        trials = worker_id_to_trials[worker_id]
        for trial in trials:
            for t in trial:
                subj_id = t['human_subj_id']
                yvals = subj_id_to_yvals[subj_id]
                for comp, comp_res in t['depth']['keypoint_comparisons_res'].iteritems():
                    kpt1, kpt2 = [int(_k) for _k in comp.split(',')]
                    kpt1_depth = yvals[kpt1]
                    kpt2_depth = yvals[kpt2]
                    depth_diff = kpt1_depth - kpt2_depth
                    worker_comp_res.append(int(np.sign(depth_diff) == np.sign(comp_res)))
        fractions_of_correct_comps_per_worker.append(sum(worker_comp_res) * 1.0 / len(worker_comp_res))

    if plots:
        plt.figure(random.randint(0, 1000))
        bins = np.arange(0.0, 1.1, 0.1)
        n, bins, patches = plt.hist(fractions_of_correct_comps_per_worker, bins, facecolor='green')
        plt.figure(random.randint(0, 1000))
        n = np.array([_i * 100 / sum(n) for _i in n])
        
        labels = np.arange(0, 100, 10)
        pos = np.arange(len(labels)) / 5.0
        width = 0.2     # gives histogram aspect to the bar diagram
        ax = plt.axes()
        ax.set_xticks(pos)
        ax.set_xticklabels(labels)
        plt.bar(pos, n, width, color='r', align='edge')
        plt.xlabel('Overall % of Correct Comparisons')
        plt.ylabel('Percent of Turkers')
        plt.title('Accuracy per Turker')
        plt.grid(True)
        plt.show()        

    return fractions_of_correct_comps_per_worker

def fraction_of_correct_comparisons_per_metatrial(data, _human_dataset, subj_id_to_yvals, subj_id_to_metatrials,plots=True):
    fractions_of_correct_comps_per_metatrial = []

    for subj_id, t in subj_id_to_metatrials.iteritems():
        yvals = subj_id_to_yvals[subj_id]
        trial_comp_res = []
        for comp, comp_res in t.iteritems():
            kpt1, kpt2 = [int(_k) for _k in comp.split(',')]
            kpt1_depth = yvals[kpt1]
            kpt2_depth = yvals[kpt2]
            depth_diff = kpt1_depth - kpt2_depth
            trial_comp_res.append(int(np.sign(depth_diff) == np.sign(comp_res)))
        fractions_of_correct_comps_per_metatrial.append(sum(trial_comp_res) * 1.0 / len(trial_comp_res))
    
    if plots:
        plt.figure(random.randint(0, 1000))
        bins = np.arange(-0.1, 1.1, 0.2)
        n, bins, patches = plt.hist(fractions_of_correct_comps_per_metatrial, bins, facecolor='green')
        plt.figure(random.randint(0, 1000))
        n = np.array([_i * 1.0 / sum(n) for _i in n])
        
        labels = np.arange(0.0, 1.1, 0.2)
        pos = np.arange(len(labels)) / 5.0
        width = 0.2     # gives histogram aspect to the bar diagram
        ax = plt.axes()
        ax.set_xticks(pos)
        ax.set_xticklabels(labels)
        plt.bar(pos, n, width, color='r', linewidth=1, edgecolor='black')
        plt.xlabel('Correct / Total Comparisons per MetaTrial')
        plt.ylabel('Proportion of MetaTrials')
        plt.title('Fraction of Correct Comparisons per MetaTrial')
        ax.yaxis.grid(True)
        plt.show()    

def fraction_of_correct_comparisons_per_metahit(data, _human_dataset, subj_id_to_yvals, subj_id_to_metatrials, hit_id_to_subj_ids, plots=True):
    fractions_of_correct_comps_per_metahit = []

    for hit_id, subj_ids in hit_id_to_subj_ids.iteritems():
        metahit_comp_res = []
        for subj_id in subj_ids:
            t = subj_id_to_metatrials[subj_id]
            yvals = subj_id_to_yvals[subj_id]
            for comp, comp_res in t.iteritems():
                kpt1, kpt2 = [int(_k) for _k in comp.split(',')]
                kpt1_depth = yvals[kpt1]
                kpt2_depth = yvals[kpt2]
                depth_diff = kpt1_depth - kpt2_depth
                metahit_comp_res.append(int(np.sign(depth_diff) == np.sign(comp_res)))
        fractions_of_correct_comps_per_metahit.append(sum(metahit_comp_res) * 1.0 / len(metahit_comp_res))

    if plots:
        plt.figure(random.randint(0, 1000))
        bins = np.arange(0.0, 1.1, 0.2)
        n, bins, patches = plt.hist(fractions_of_correct_comps_per_metahit, bins, facecolor='green')
        plt.figure(random.randint(0, 1000))
        n = np.array([_i * 1.0 / sum(n) for _i in n])
        
        labels = np.arange(0.0, 1.0, 0.2)
        pos = np.arange(len(labels)) / 5.0
        width = 0.2     # gives histogram aspect to the bar diagram
        ax = plt.axes()
        ax.set_xticks(pos)
        ax.set_xticklabels(labels)
        plt.bar(pos, n, width, color='r', align='edge')
        plt.xlabel('Correct / Total Comparisons per MetaHIT')
        plt.ylabel('Proportion of MetaHITs')
        plt.title('Fraction of Correct Comparisons per MetaHIT')
        plt.grid(True)
        plt.show()    

def wrongness_distance_of_comps(data, _human_dataset, subj_id_to_yvals, subj_id_to_metatrials, workers_ranked_worst_to_best, plots=True):
    distance_of_wrong_comps = []
    distance_of_correct_comps = []

    FRACTION_OF_WORKERS_TO_EXCLUDE = 0.1
    accepted_worker_ids = workers_ranked_worst_to_best[int(len(workers_ranked_worst_to_best) * FRACTION_OF_WORKERS_TO_EXCLUDE):]

    for d in data:
        if d['worker_id'] not in accepted_worker_ids:
            continue
        for t in d['trials']:
            subj_id = t['human_subj_id']
            yvals = subj_id_to_yvals[subj_id]
            for comp, comp_res in t['depth']['keypoint_comparisons_res'].iteritems():
                kpt1, kpt2 = [int(_k) for _k in comp.split(',')]
                kpt1_depth = yvals[kpt1]
                kpt2_depth = yvals[kpt2]
                depth_diff = kpt1_depth - kpt2_depth
                if np.sign(depth_diff) == np.sign(comp_res):
                    distance_of_correct_comps.append(abs(depth_diff))
                else:
                    distance_of_wrong_comps.append(abs(depth_diff))

    distance_of_wrong_metacomps = []
    distance_of_correct_metacomps = []

    for subj_id, t in subj_id_to_metatrials.iteritems():
        yvals = subj_id_to_yvals[subj_id]
        for comp, comp_res in t.iteritems():
            kpt1, kpt2 = [int(_k) for _k in comp.split(',')]
            kpt1_depth = yvals[kpt1]
            kpt2_depth = yvals[kpt2]
            depth_diff = kpt1_depth - kpt2_depth
            if np.sign(depth_diff) == np.sign(comp_res):
                distance_of_correct_metacomps.append(abs(depth_diff))
            else:
                distance_of_wrong_metacomps.append(abs(depth_diff))

    if plots:
        plt.figure(random.randint(0, 1000))
        # bins = np.arange(-1, 1251, 250)
        bins = np.arange(0, 1001, 10)
        n_wrong, bins, patches = plt.hist(distance_of_wrong_comps, bins, facecolor='green')
        n_correct, bins, patches = plt.hist(distance_of_correct_comps, bins, facecolor='green')
        n_wrong_meta, bins, patches = plt.hist(distance_of_wrong_metacomps, bins, facecolor='green')
        n_correct_meta, bins, patches = plt.hist(distance_of_correct_metacomps, bins, facecolor='green')
        plt.figure(random.randint(0, 1000))
        # n_wrong = np.array([_i * 1.0 / (sum(n_wrong) + sum(n_correct)) for _i in n_wrong])
        n_correct = np.array([n_c * 100 / (n_w + n_c) for n_w, n_c in zip(n_wrong, n_correct)])
        n_correct_meta = np.array([n_c * 100 / (n_w + n_c) for n_w, n_c in zip(n_wrong_meta, n_correct_meta)])

        labels = np.arange(0, 1000, 10)
        pos = np.arange(len(labels)) / 5.0
        width = 0.2     # gives histogram aspect to the bar diagram
        ax = plt.axes()
        ax.set_xticks(pos)
        ax.set_xticklabels(labels)
        for i, label in enumerate(ax.xaxis.get_ticklabels()): # Erase every nth tick
            if i % 10 != 0:
                label.set_visible(False)

        # bar_wrong = plt.bar(pos, n_wrong, width, color='r', align='edge')
        # bar_correct = plt.bar(pos, n_correct, width, bottom=n_wrong, color='g', align='edge')
        plot_correct = plt.plot(pos, n_correct, color='blue')
        plot_correct_meta = plt.plot(pos, n_correct_meta, color='red')
        plt.xlabel('Distance between Keypoints in mm')
        plt.ylabel('Overall % of Correct Pairs')
        plt.title('Accuracy per Distance')
        plt.legend((plot_correct[0], plot_correct_meta[0]), ('Worker', 'Metaworker'))
        plt.grid(True)
        plt.show()

def wrongness_distance_of_metacomps(data, _human_dataset, subj_id_to_yvals, subj_id_to_metatrials, plots=True):
    distance_of_wrong_metacomps = []
    distance_of_correct_metacomps = []

    for subj_id, t in subj_id_to_metatrials.iteritems():
        yvals = subj_id_to_yvals[subj_id]
        for comp, comp_res in t.iteritems():
            kpt1, kpt2 = [int(_k) for _k in comp.split(',')]
            kpt1_depth = yvals[kpt1]
            kpt2_depth = yvals[kpt2]
            depth_diff = kpt1_depth - kpt2_depth
            if np.sign(depth_diff) == np.sign(comp_res):
                distance_of_correct_metacomps.append(abs(depth_diff))
            else:
                distance_of_wrong_metacomps.append(abs(depth_diff))

    if plots:
        plt.figure(random.randint(0, 1000))
        bins = np.arange(-1, 1251, 250)
        n_wrong, bins, patches = plt.hist(distance_of_wrong_metacomps, bins, facecolor='green')
        n_correct, bins, patches = plt.hist(distance_of_correct_metacomps, bins, facecolor='green')
        plt.figure(random.randint(0, 1000))
        n_wrong = np.array([_i * 1.0 / (sum(n_wrong) + sum(n_correct)) for _i in n_wrong])
        n_correct = np.array([_i * 1.0 / (sum(n_wrong) + sum(n_correct)) for _i in n_correct])

        labels = np.arange(0, 1250, 250)
        pos = np.arange(len(labels)) / 5.0
        width = 0.2     # gives histogram aspect to the bar diagram
        ax = plt.axes()
        ax.set_xticks(pos)
        ax.set_xticklabels(labels)
        bar_wrong = plt.bar(pos, n_wrong, width, color='r', align='edge')
        bar_correct = plt.bar(pos, n_correct, width, bottom=n_wrong, color='g', align='edge')
        plt.xlabel('Distance between Compared Keypoints')
        plt.ylabel('Proportion of Comparisons')
        plt.title('Proportion of Correct/Incorrect Metaperson Comparisons by Keypoint Pair Distance')
        plt.legend((bar_wrong[0], bar_correct[0]), ('Incorrect', 'Correct'))
        plt.grid(True)
        plt.show()

def worker_experience_vs_accuracy(data, subj_id_to_yvals, worker_id_to_trials, workers_ranked_worst_to_best, worker_accuracy, plots=True):
    x = [worker_accuracy[worker_id] * 100 for worker_id in worker_accuracy.keys()]
    y = [len(worker_id_to_trials[worker_id]) * 25 for worker_id in worker_accuracy.keys()]

    if plots:
        plt.figure(random.randint(0, 1000))
        ax = plt.axes()
        plt.scatter(x, y)
        plt.xlabel('% of Correct Comparisons')
        plt.ylabel('Total Number of Comparisons per Turker')
        plt.title('Accuracy vs Number of Comparisons per Turker')
        ax.set_yscale('log')

        plt.figure(random.randint(0, 1000))
        ax = plt.axes()
        plt.scatter(x, y)
        plt.xlabel('% of Correct Comparisons')
        plt.ylabel('Total Number of Comparisons per Turker')
        plt.title('Accuracy vs Number of Comparisons per Turker')
        plt.show()

def image_difficulty(data, _human_dataset, subj_id_to_trials, subj_id_accuracy, subjects_ranked_hard_to_easy, plots=True):
    x = np.arange(len(subjects_ranked_hard_to_easy))
    y = [subj_id_accuracy[subj_id] * 100 for subj_id in subjects_ranked_hard_to_easy]

    subj_id_to_image_path = {}
    for d in _human_dataset['images']:
        subj_id_to_image_path[d['id']] = d['filename']

    print("The hardest 5 subjects are {}".format(subjects_ranked_hard_to_easy[:5]))
    hardest_paths = [subj_id_to_image_path[subj_id] for subj_id in subjects_ranked_hard_to_easy[:5]]
    print("The hardest 5 images' paths are {}".format(hardest_paths))

    print("The easiest 5 subjects are {}".format(subjects_ranked_hard_to_easy[-5:]))
    hardest_paths = [subj_id_to_image_path[subj_id] for subj_id in subjects_ranked_hard_to_easy[-5:]]
    print("The easiest 5 images' paths are {}".format(hardest_paths))

    if plots:
        plt.plot(x, y)
        plt.ylim(ymin=0)
        plt.xlabel('Images Sorted Hardest to Easiest')
        plt.ylabel('% of Correct Comparisons per Image')
        plt.title('Accuracy per Human3.6 Subject i.e. Image Difficulty')
        plt.grid(True)
        plt.show()

