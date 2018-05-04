# use_analysis.py
# uses analysis functions from postprocess_original_utils.py

import json
import numpy as np
import random

from matplotlib import pyplot as plt

from postprocess_original_utils import (load_data, wrongness,
    get_keypoint_comparison_depths, metaperson_comparisons, worker_comparisons)
    
from human36m_graphs import (
    fraction_of_correct_comparisons_per_trial,
    fraction_of_correct_comparisons_per_assignment,
    fraction_of_correct_comparisons_per_subject,
    fraction_of_correct_comparisons_per_hit,
    fraction_of_correct_comparisons_per_worker,
    fraction_of_correct_comparisons_per_metatrial,
    fraction_of_correct_comparisons_per_metahit,
    wrongness_distance_of_comps,
    wrongness_distance_of_metacomps,
    worker_experience_vs_accuracy,
    image_difficulty,
    visualize_difficult_images)

from lean_correction import correct_lean
from constants import (HUMAN_ANNOTATION_PATH, COCO_ANNOTATION_PATH,
    HUMAN_RAW_RESULT_PATH, COCO_RAW_RESULT_PATH, KEYPTS_RELATIVE_DEPTH_PATH,
    KEYCMPS_RESULT_PATH, HUMAN_OUTPUT_PATH)

with open(HUMAN_ANNOTATION_PATH) as f:
        _human_dataset = json.load(f)
        correct_lean(_human_dataset)

data = load_data(load_from_file=True, full_ordering=False)


subj_id_to_yvals = {}
axis_to_sort = 1
for d in _human_dataset['annotations']:
    subj_id_to_yvals[d['i_id']] = list(d['kpts_3d'])
    # subj_id_to_yvals[d['i_id']][3:6] = [] # Remove neck keypoint
    subj_id_to_yvals[d['i_id']] = subj_id_to_yvals[d['i_id']][axis_to_sort::3]


subj_id_to_trials = {}
for d in data:
    for t in d['trials']:
        subj_id = t['human_subj_id']
        if subj_id in subj_id_to_trials:
            subj_id_to_trials[subj_id].append(t['depth']['keypoint_comparisons_res'])
        else:
            subj_id_to_trials[subj_id] = [t['depth']['keypoint_comparisons_res']]


hit_id_to_trials = {}
for d in data:
    hit_id = d['hit_id']
    if hit_id in hit_id_to_trials:
        hit_id_to_trials[hit_id].append(d['trials'])
    else:
        hit_id_to_trials[hit_id] = [d['trials']]


worker_id_to_trials = {}
for d in data:
    worker_id = d['worker_id']
    if worker_id in worker_id_to_trials:
        worker_id_to_trials[worker_id].append(d['trials'])
    else:
        worker_id_to_trials[worker_id] = [d['trials']]


subj_id_to_metatrials = {}
for subj_id in subj_id_to_trials.keys():
    metatrial = {}
    for t in subj_id_to_trials[subj_id]:
        for comp, comp_res in t.iteritems():
            if comp in metatrial:
                metatrial[comp] += comp_res
            else:
                metatrial[comp] = comp_res
    for comp, comp_res in metatrial.iteritems():
        metatrial[comp] = np.sign(comp_res) if comp_res != 0 else random.choice([-1, 1])
    subj_id_to_metatrials[subj_id] = metatrial


hit_id_to_subj_ids = {}
for d in data:
    hit_id_to_subj_ids[d['hit_id']] = d['human_subj_ids']


workers_ranked_worst_to_best = []
worker_accuracy = {}
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
    worker_accuracy[worker_id] = sum(worker_comp_res) * 1.0 / len(worker_comp_res)
workers_ranked_worst_to_best = [key for key, value in sorted(worker_accuracy.iteritems(), key=lambda (k,v): (v,k))]


subj_id_accuracy = {}
subjects_ranked_hard_to_easy = []
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
    subj_id_accuracy[subj_id] = sum(subj_comp_res) * 1.0 / len(subj_comp_res)
subjects_ranked_hard_to_easy = [key for key, value in sorted(subj_id_accuracy.iteritems(), key=lambda (k,v): (v,k))]


# fraction_of_correct_comparisons_per_trial(data, _human_dataset, subj_id_to_yvals)
# fraction_of_correct_comparisons_per_assignment(data, _human_dataset, subj_id_to_yvals)
# fraction_of_correct_comparisons_per_subject(data, _human_dataset, subj_id_to_yvals, subj_id_to_trials)
# fraction_of_correct_comparisons_per_hit(data, _human_dataset, subj_id_to_yvals, hit_id_to_trials)
# fraction_of_correct_comparisons_per_worker(data, _human_dataset, subj_id_to_yvals, worker_id_to_trials)
# fraction_of_correct_comparisons_per_metatrial(data, _human_dataset, subj_id_to_yvals, subj_id_to_metatrials)
# fraction_of_correct_comparisons_per_metahit(data, _human_dataset, subj_id_to_yvals, subj_id_to_metatrials, hit_id_to_subj_ids)
# wrongness_distance_of_comps(data, _human_dataset, subj_id_to_yvals, subj_id_to_metatrials, workers_ranked_worst_to_best)
# wrongness_distance_of_metacomps(data, _human_dataset, subj_id_to_yvals, subj_id_to_metatrials)
# worker_experience_vs_accuracy(data, subj_id_to_yvals, worker_id_to_trials, workers_ranked_worst_to_best, worker_accuracy)
# image_difficulty(data, _human_dataset, subj_id_to_trials, subj_id_accuracy, subjects_ranked_hard_to_easy)
visualize_difficult_images(data, _human_dataset, subj_id_to_trials, subj_id_accuracy, subjects_ranked_hard_to_easy)


# from constants import CALTECH_OUTPUT_PATH
# data = json.load(open(CALTECH_OUTPUT_PATH, 'r'))
# lab = ['nonAMT_607266', 'nonAMT_368102', 'nonAMT_6599', 'nonAMT_687008', 'nonAMT_764039', 'nonAMT_700986']
# data = [d for d in data if d['_worker_id'] in lab]
# '''
# {u'Amanda Lin: nonAMT_808135',
#  u'Caltech: ',
#  u'Jennifer - Hi! I wou: nonAMT_607266',
#  u'Matteo: nonAMT_368102',
#  u'Milan: nonAMT_6599',
#  u'Oisin: nonAMT_687008',
#  u'Ron: nonAMT_764039',
#  u'This is Jalani. I wo: nonAMT_158235',
#  u'caltech: ',
#  u'lucy: nonAMT_700986'}
# '''

# wrongness(data)
# metaperson_comparisons(data)
# worker_comparisons(data)
