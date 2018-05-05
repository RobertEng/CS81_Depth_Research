'''
post_process_depth_hits.py

Process all the gathered depth information from AMT.
'''

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

################################################################################
# PROCESS FUNCTIONS
################################################################################

def load_data(load_from_file=True, full_ordering=True):
    '''
    Reads in the data gathered from the GUI. Does basic calculations over each
    hit which may be helpful. Returns data grouped by hit.

    full_ordering: True if all comparisons were collected, otherwise assume
        only a few comparisons were found and the full ordering cannot be used.

    Step 1: Load the raw result data
    Step 2: Recreate relative depth ordering
    Step 3: Associate the ground truth data
    Step 4: Perform helpful calculations
    Step 5: Write to a file
    '''
    if load_from_file and os.path.isfile(HUMAN_OUTPUT_PATH):
        return json.load(open(HUMAN_OUTPUT_PATH, 'r'))
        
    ### Step 1
    data = pickle.load(open(HUMAN_RAW_RESULT_PATH, 'r'))
    print "Read in {} HIT total assignments".format(len(data['_all_assignments']))
    print "{} good, {} flagged, {} error, {} rejected".format(
        len(data['_good_assignments']), len(data['_flagged_assignments']),
        len(data['_error_assignments']), len(data['_rejected_assignments']))
    data = data['_good_assignments']

    ### Step 2
    if full_ordering:
        # Recreate the relative depth ordering with a local version of the GUI.
        # TODO: Collect this data from the GUI directly.
        if not os.path.isfile(KEYPTS_RELATIVE_DEPTH_PATH):
            ## DUMP KEYPOINTCOMARPISONS ORDER AND RES TO FILE TO BE READ BY JS
            keycmp_orders = [h['trials'][0]['depth']['keypoint_comparisons_order'] for h in data]
            keycmp_reses = [h['trials'][0]['depth']['keypoint_comparisons_res'] for h in data]
            keycmps = {'keycmp_orders': keycmp_orders, 'keycmp_reses': keycmp_reses}

            with open(KEYCMPS_RESULT_PATH, 'w') as f:
                f.write(json.dumps(keycmps))
            print "Output keypoint comparisons to {}".format(KEYCMPS_RESULT_PATH)
            sys.exit()

        _relative_depth = json.load(open(KEYPTS_RELATIVE_DEPTH_PATH, 'r'))
        for d, kpts_relative_depth in zip(data, _relative_depth):
            d['trials'][0]['kpts_relative_depth'] = kpts_relative_depth
        print "Relative depth ordering for Turkers associated with the data"

    ### Step 3
    # Get the img_ids of images annotated by turkers to match with the ground
    # truth data
    img_ids = [h['trials'][0]['img_id'] for h in data]

    # Get the ground truth annotations from human dataset
    with open(HUMAN_ANNOTATION_PATH) as f:
        _human_dataset = json.load(f)
        correct_lean(_human_dataset)
    # >>> _human_dataset.keys()
    # [u'images', u'pose', u'annotations', u'actions']
    # >>> _human_dataset['images'][0].keys()
    # [u'c_id', u's_id', u'frame', u'height', u'width', u'video', u'filename', u'id']
    # >>> _human_dataset['annotations'][0].keys()
    # [u'i_id', u's_id', u'a_id', u'kpts_2d', u'id', u'kpts_3d']

    # Match the ground truth annotations with the turker data annotations
    for i in range(len(_human_dataset['images'])):
        if _human_dataset['images'][i]['id'] in img_ids:
            img_id_idxes = [img_id_idx for img_id_idx, img_id in enumerate(img_ids)
                            if img_id == _human_dataset['images'][i]['id']]
            for img_id_idx in img_id_idxes:
                data[img_id_idx]['images_truth'] = _human_dataset['images'][i]
                data[img_id_idx]['annotations_truth'] = copy.deepcopy(_human_dataset['annotations'][i])

    print "{} images in human3.6 dataset".format(len(_human_dataset['images']))
    print "{} annotations matched with ground truth".format(len(data))

    # Remove the neck keypoint. Arrange the depth data in a useful way.
    # >>> _human_dataset['pose']
    # [{u'original_index': [15, 13, 17, 25, 18, 26, 19, 27, 6, 1, 7, 2, 8, 3],
    # u'keypoints': [u'head', u'neck', u'left_shoulder', u'right_shoulder',
    #   u'left_elbow', u'right_elbow', u'left_wrist', u'right_wrist', u'left_hip',
    #   u'right_hip', u'left_knee', u'right_knee', u'left_ankle', u'right_ankle'],
    # u'skeleton': [[0, 1], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [2, 8],
    #   [3, 9], [8, 10], [9, 11], [10, 12], [11, 13]]}]
    for d in data:
        # Remove neck keypoint
        # d['annotations_truth']['kpts_2d'][2:4] = []
        # d['annotations_truth']['kpts_3d'][3:6] = []
        # Grab every third kpts_3d which corresponds to depth
        axis_to_sort = 1 # change this if it looks like we're sorting along the wrong axis
        d['annotations_truth']['kpts_depth'] = d['annotations_truth']['kpts_3d'][axis_to_sort::3]
        d['annotations_truth']['kpts_relative_depth'] = \
            [i[0] for i in sorted(enumerate(d['annotations_truth']['kpts_depth']),
                                  key=lambda x:x[1])]

    ### Step 4
    # Perform helpful calculations


    ### Step 5
    with open(HUMAN_OUTPUT_PATH, 'w') as f:
        json.dump(data, f)
    print "Output data to {}".format(HUMAN_OUTPUT_PATH)

    return data


def group_data_by_image(data_by_hit):
    # Group hits by image. 3 Turkers were assigned the same image to annotate.
    # Derive useful calculations from this data ie majority vote it, etc.
    img_ids = [h['trials'][0]['img_id'] for h in data_by_hit]
    img_ids_to_data = {img_id: [] for img_id in img_ids}
    [img_ids_to_data[d['trials'][0]['img_id']].append(d) for d in data_by_hit]

    # Calculate the metaperson's annotations for this picture
    data_by_image = []
    for img_id, hits in img_ids_to_data.iteritems():
        image_data = {"hits": hits}
        metaperson = {'keypoint_comparisons_res': {}, 'img_id': img_id}
        metaperson['annotations_truth'] = hits[0]['annotations_truth']

        # For each comp, get the majority vote
        for comp in hits[0]['trials'][0]['depth']['keypoint_comparisons_res'].keys():
            votes = []
            for m in hits:
                # If this comparison doesn't exists in this hit, then calculate it.
                if comp not in m['trials'][0]['depth']['keypoint_comparisons_res']:
                    kpt1, kpt2 = [int(_k) for _k in comp.split(',')]
                    m['trials'][0]['depth']['keypoint_comparisons_res'][comp] = np.sign(m['trials'][0]['kpts_relative_depth'].index(kpt2) - m['trials'][0]['kpts_relative_depth'].index(kpt1))
                # Add it to votes
                votes.append(m['trials'][0]['depth']['keypoint_comparisons_res'][comp])
            metaperson['keypoint_comparisons_res'][comp] = find_majority(votes)
        image_data['metaperson'] = metaperson
        data_by_image.append(image_data)
    return data_by_image


def group_data_by_worker(data_by_hit):
    # Group hits by worker.
    worker_ids = set([h['worker_id'] for h in data_by_hit])
    worker_id_to_data = {worker_id: [] for worker_id in worker_ids}
    [worker_id_to_data[d['worker_id']].append(d) for d in data_by_hit]

    data_by_worker = []
    for worker_id, hits in worker_id_to_data.iteritems():
        worker_data = {"hits": hits}
        data_by_worker.append(worker_data)
    return data_by_worker


def calculate_naive_score(d):
    lst1 = d['trials'][0]['kpts_relative_depth']
    lst2 = d['annotations_truth']['kpts_relative_depth']
    score = 0
    for n in lst1:
        score += abs(lst2.index(n) - lst1.index(n))
    return score


def calculate_dist_score(d):
    lst1 = d['trials'][0]['kpts_relative_depth']
    lst2 = d['annotations_truth']['kpts_relative_depth']
    depths = d['annotations_truth']['kpts_depth']
    score = 0
    for kpt1, kpt2 in zip(lst1, lst2):
        score += abs(depths[kpt1] - depths[kpt2])
    return 1.0 * score / len(depths)


def find_majority(votes):
    vote_count = Counter(votes)
    top_two = vote_count.most_common(2)
    if len(top_two) > 1 and top_two[0][1] == top_two[1][1]:
        # It is a tie
        return 0
    return top_two[0][0]



def get_keypoint_comparison_depths(data, threshold):
    kpt_pair_dist_incorrect_lbl = []
    kpt_pair_dist_correct_lbl = []
    kpt_pair_dist_incorrect_lbl_gen_comps = []
    kpt_pair_dist_correct_lbl_gen_comps = []

    for d in data:
        human_made_comps    = d['trials'][0]['depth']['keypoint_comparisons_order']
        kpts_depth          = d['annotations_truth']['kpts_depth']
        kpts_relative_depth = d['annotations_truth']['kpts_relative_depth']

        # Loop through all the comparisons and calculate their distances
        for comp, comp_res in d['trials'][0]['depth']['keypoint_comparisons_res'].iteritems():
            kpt1, kpt2 = [int(_k) for _k in comp.split(',')]
            kpt1_depth = kpts_depth[kpts_relative_depth.index(kpt1)]
            kpt2_depth = kpts_depth[kpts_relative_depth.index(kpt2)]
            depth_diff = kpt1_depth - kpt2_depth

            # Collect whatever data is needed for the graphs
            # If the comparison was correct
            if ((abs(depth_diff) < threshold and comp_res == 0) or np.sign(comp_res) == np.sign(depth_diff)):
                # If the comparison was turker made (as opposed to generated)
                if comp in human_made_comps:
                    kpt_pair_dist_correct_lbl.append(depth_diff)
                else:  # if comparison was generated (not turker made)
                    kpt_pair_dist_correct_lbl_gen_comps.append(depth_diff)

            else:  # Comparison was not correct
                # If the comparison was turker made (as opposed to generated)
                if comp in human_made_comps:
                    if depth_diff > 600 and depth_diff < 700:
                        print (d['annotations_truth']['i_id'])
                    kpt_pair_dist_incorrect_lbl.append(depth_diff)
                else:  # if comparison was generated (not turker made)
                    kpt_pair_dist_incorrect_lbl_gen_comps.append(depth_diff)

    return (kpt_pair_dist_incorrect_lbl,
            kpt_pair_dist_correct_lbl,
            kpt_pair_dist_incorrect_lbl_gen_comps,
            kpt_pair_dist_correct_lbl_gen_comps)


def lookup_hits_from_file_names(data, file_names):
    '''
    NOTE: One caveat is it only takes one of the annotations for the image
    rather than all of them.
    '''
    hits = []
    files_seen = set()
    for d in data:
        if (d['images_truth']['filename'] in file_names
                and d['images_truth']['filename'] not in files_seen):
            files_seen.add(d['images_truth']['filename'])
            hits.append(d)
    return hits


def lookup_file_names_from_img_ids(img_ids):
    with open(HUMAN_ANNOTATION_PATH) as f:
        _human_dataset = json.load(f)

    filenames = []
    for h in _human_dataset['images']:
        if h['id'] in img_ids:
            filenames.append(h['filename'])
    return filenames


def lookup_file_names_from_worker_ids(data, worker_ids):
    img_ids = set()
    for d in data:
        if d['worker_id'] in worker_ids:
            img_ids.add(d['trials'][0]['img_id'])
    # print img_ids # set([408524, 668287])
    return lookup_file_names_from_img_ids(list(img_ids))


def lookup_hit_id_from_img_ids(data, img_ids):
    hit_ids = []
    for img in img_ids:
        for d in data:
            if d['trials'][0]['img_id'] == img:
                hit_ids.append(d['hit_id'])
                break
    return hit_ids


def find_images_for_mini_experiment(data):
    (worker_id_to_correct_human_made_comparison_count,
    worker_id_to_total_human_made_comparisons,
    worker_id_to_correct_generated_comparison_count,
    worker_id_to_total_generated_comparisons) = worker_comparisons(data, plots=False)

    worker_ids = worker_id_to_correct_human_made_comparison_count.keys()
    worker_ids_to_lookup = []
    ### This doesn't work. The mistake is I took the worst turkers rather than
    ### the worst images.
    # x = sorted([worker_id for worker_id in worker_ids if 100 * worker_id_to_correct_human_made_comparison_count[worker_id] / worker_id_to_total_human_made_comparisons[worker_id] < 30])
    # worker_ids_to_lookup.extend(x)
    # print x
    # x = sorted([worker_id for worker_id in worker_ids if 100 * worker_id_to_correct_human_made_comparison_count[worker_id] / worker_id_to_total_human_made_comparisons[worker_id] > 90])
    # worker_ids_to_lookup.extend(x)
    # print x
    # print worker_ids_to_lookup
    # filenames = lookup_file_names_from_worker_ids(data, worker_ids_to_lookup)

    # agreement(data, plots=False)
    # img_ids = [465672, 44510]

    # filenames.extend(lookup_file_names_from_img_ids(img_ids))


    # Find the image ids to run for the lab mini experiment
    # smallest and largest at img id [408524, 668287]
    # Median one as 56%. 
    x = sorted([100 * worker_id_to_correct_human_made_comparison_count[worker_id] / worker_id_to_total_human_made_comparisons[worker_id] for worker_id in worker_ids])
    x = sorted([worker_id for worker_id in worker_ids if 100 * worker_id_to_correct_human_made_comparison_count[worker_id] / worker_id_to_total_human_made_comparisons[worker_id] == 56])
    # [u'A1EG2FJJUWAYJU', u'A2RXPHOE9V9634'] # both are the median
    x = ['A2RXPHOE9V9634'] # arbitrarily choose this one.
    lookup_file_names_from_worker_ids(data, x)


    # Across the images, which ones is the hardest, easiest, and median.
    # TODO: DEPRECATED USE OF GROUP_DATA_BY_IMAGE
    img_ids_to_data = group_data_by_image(data)
    (img_id_to_correct_generated_majority_vote_comparison_count,
    img_id_to_total_generated_majority_vote_comparisons) = metaperson_comparisons(data, plots=False)
    # Sort the images based on % correct on average. Grab the image ids from this
    x = sorted([(100 * img_id_to_correct_generated_majority_vote_comparison_count[img_id] / img_id_to_total_generated_majority_vote_comparisons[img_id], img_id) for img_id in img_ids_to_data.keys()])
    # x = 571176 # Smallest at 29%
    # x = 12750 # Median at 59%
    # x = 849671 # Largest at 85%
    [571176, 12750, 849671]
    # Also choose 4 more random pics. random choose one to do twice.
    [56833, 965922, 649263, 896082]
    repeatd = 965922

    img_ids = [56833, 965922, 849671, 649263, 12750, 896082, 571176, 965922]
    filenames = lookup_file_names_from_img_ids(img_ids)
    # [u'human36m_train_0000849671.jpg', u'human36m_train_0000012750.jpg', u'human36m_train_0000571176.jpg']

    # Get hit ids of the images
    x = lookup_hit_id_from_img_ids(data, img_ids)
    hit_ids = [41, 198, 111, 158, 19, 181, 135, 198]

    # random1, random2, best_performance, random3, median_performance, random4, worst_performance, random2_repeat


def get_action_data_from_human(_human_dataset, subj_ids, action, version=0):
    # human
    _human_dataset['images'] = [d for d in _human_dataset['images'] if d['s_id'] in subj_ids]
    _human_dataset['annotations'] = [d for d in _human_dataset['annotations'] if d['s_id'] in subj_ids]

    action_key = [_act for _act in _human_dataset['actions'] if _act['name'] == action and _act['version'] == version][0]
    action_id = action_key['id']
    
    action_data_annotations = []
    action_data_images = []
    for d_ann, d_img in zip(_human_dataset['annotations'], _human_dataset['images']):
        if d_ann['a_id'] == action_id:
            action_data_annotations.append(d_ann)
            action_data_images.append(d_img)
    return action_data_annotations, action_data_images




################################################################################
# ANALYSIS FUNCTIONS
################################################################################

def calculate_base_depth_statistics(data):
    '''
    This function serves to analyze the ground truth depth data and to better
    understand it. In doing so, it will shed some light on the larger analysis.
    '''
    # Calculate average range of depth in each image
    ranges = [max(d['annotations_truth']['kpts_depth']) - 
        min(d['annotations_truth']['kpts_depth']) for d in data]
    mean_range = 1.0 * sum(ranges) / len(data)
    median_range = np.median(ranges)

    x = ranges
    # the histogram of the data
    n, bins, patches = plt.hist(x, 6, facecolor='blue')
    plt.axvline(x=median_range, color='red')

    plt.xlabel('Depth Ranges (unknown units, probably mm)')
    plt.ylabel('Count')
    plt.title('Ranges of keypoint depths Human3.6m N={}.\n(ie the distance between closest and furthest keypoint)'.format(len(data)))
    plt.grid(True)
    plt.show()
    print ("Range of keypoint depth for a subject ie closest keypoint" \
        " minus furthest keypoint: mean {}, median {}"
        .format(mean_range, median_range))

    # TODO: analysis on the average changes in depth between two adjacent keypoints


def scores_histogram(scores):
    x = scores
    # the histogram of the data
    n, bins, patches = plt.hist(x, 6, facecolor='green')
    plt.xlabel('Scores')
    plt.ylabel('Count')
    plt.title('Average scores for depth annotated Human3.6m N={}'.format(len(data)))
    plt.grid(True)
    plt.show()


def worker_comparisons(data, plots=True):
    THRESHOLD = 500  # 500mm or 50cm
    thresholds = [1000, 500, 200, 150, 100]
    for threshold in thresholds:
        worker_id_to_correct_hum_comp_count = defaultdict(int)
        worker_id_to_total_hum_comp_count   = defaultdict(int)
        worker_id_to_correct_gen_comp_count = defaultdict(int)
        worker_id_to_total_gen_comp_count   = defaultdict(int)
        worker_ids = set([d['worker_id'] for d in data])
        for d in data:
            worker_id           = d['worker_id']
            human_made_comps    = d['trials'][0]['depth']['keypoint_comparisons_order']
            kpts_depth          = d['annotations_truth']['kpts_depth']
            kpts_relative_depth = d['annotations_truth']['kpts_relative_depth']

            # Loop through all the comparisons and calculate their distances
            for comp, comp_res in d['trials'][0]['depth']['keypoint_comparisons_res'].iteritems():
                kpt1, kpt2 = [int(_k) for _k in comp.split(',')]
                kpt1_depth = kpts_depth[kpts_relative_depth.index(kpt1)]
                kpt2_depth = kpts_depth[kpts_relative_depth.index(kpt2)]
                depth_diff = kpt2_depth - kpt1_depth

                ### Collect whatever data is needed for the graphs
                # If the comparison was correct
                if ((abs(depth_diff) < threshold and comp_res == 0) or
                        np.sign(comp_res) == np.sign(depth_diff)):
                    # If the comparison was human made
                    if comp in human_made_comps:
                        worker_id_to_correct_hum_comp_count[worker_id] += 1
                    else: # It was a generated comparison
                        worker_id_to_correct_gen_comp_count[worker_id] += 1
                else: # It was an incorrect comparison
                    if comp in human_made_comps: # Human made comparison
                        worker_id_to_total_hum_comp_count[worker_id] += 1
                    else: # It was a generated comparison
                        worker_id_to_total_gen_comp_count[worker_id] += 1

        if plots:
            plt.figure(random.randint(0, 1000))
            # Percentage right during comparisons. Binned ~50cm each, histogram.
            x = sorted([100 * worker_id_to_correct_hum_comp_count[worker_id] / worker_id_to_total_hum_comp_count[worker_id] for worker_id in worker_ids if worker_id_to_total_hum_comp_count[worker_id] > 0])
            # the histogram of the data
            bins = np.arange(0, 110, 10)
            n, bins, patches = plt.hist(x, bins, facecolor='green')
            plt.xlabel('Percentage Correct during Comparisons By Worker')
            plt.ylabel('Worker Count')
            plt.title('Human3.6m Human-made Comparisons by Worker\nnum_workers={}, threshold={}'.format(len(worker_ids), threshold))
            plt.grid(True)
            # plt.show()

            # Percentage right during comparisons. Binned ~50cm each, histogram. Cumulative.
            plt.figure(random.randint(0, 1000))
            num_bins = 10
            bins = np.arange(0.0, 1.1, 0.1)
            x = sorted([1.0 * worker_id_to_correct_hum_comp_count[worker_id] / worker_id_to_total_hum_comp_count[worker_id] for worker_id in worker_ids if worker_id_to_total_hum_comp_count[worker_id] > 0])
            counts, bins = np.histogram(x, bins=bins, normed=True)
            cdf = np.cumsum(counts)
            plt.bar(bins[:-1] + 0.05, cdf/10, width=0.1)
            plt.xlabel('Percentage Correct during Comparisons By Worker')
            plt.ylabel('Cumulative Proportion of Workers')
            plt.title('Human3.6m Human-made Comparisons by Worker CDF\nnum_workers={}, threshold={}'.format(len(worker_ids), threshold))
            plt.grid(True)
            # plt.show()

            # # Same as above but with inferred comparisions too
            # x = sorted([100 * worker_id_to_correct_gen_comp_count[worker_id] / worker_id_to_total_gen_comp_count[worker_id] for worker_id in worker_ids if worker_id_to_total_gen_comp_count[worker_id] > 0])
            # # the histogram of the data
            # bins = np.arange(0, 110, 10)
            # n, bins, patches = plt.hist(x, bins, facecolor='green')
            # plt.xlabel('Percentage Correct during Comparisons By Worker')
            # plt.ylabel('Worker Count')
            # plt.title('Human3.6m\nInferred Comparisons by Worker, num_workers={}, threshold={}'.format(len(worker_ids), threshold))
            # plt.grid(True)
            # plt.show()

            # # Percentage right during inferred comparisons. Binned ~50cm each, histogram. Cumulative.
            # num_bins = 10
            # bins = np.arange(0.0, 1.1, 0.1)
            # x = sorted([1.0 * worker_id_to_correct_gen_comp_count[worker_id] / worker_id_to_total_gen_comp_count[worker_id] for worker_id in worker_ids if worker_id_to_total_gen_comp_count[worker_id] > 0])
            # counts, bins = np.histogram(x, bins=bins, normed=True)
            # cdf = np.cumsum(counts)
            # plt.bar(bins[:-1] + 0.05, cdf/10, width=0.1)
            # plt.xlabel('Percentage Correct during Comparisons By Worker')
            # plt.ylabel('Cumulative Proportion of Workers')
            # plt.title('Human3.6m\nInferred Comparisons by Worker CDF, num_workers={}, threshold={}'.format(len(worker_ids), threshold))
            # plt.grid(True)
            # plt.show()
    plt.show()
    # return (worker_id_to_correct_human_made_comparison_count,
    #     worker_id_to_total_human_made_comparisons,
    #     worker_id_to_correct_generated_comparison_count,
    #     worker_id_to_total_generated_comparisons)


def metaperson_comparisons(data, plots=True):
    # Group hits by image. 3 Turkers were assigned the same image to annotate. Majority vote it.
    data_by_image = group_data_by_image(data)

    # With majority vote comparison results, count correct and total comparisons
    # for each metaperson.
    img_id_to_correct_generated_majority_vote_comparison_count = {}
    img_id_to_total_generated_majority_vote_comparisons = {}

    img_ids = set()
    THRESHOLD = 1000 # 500mm or 50cm
    for d in data_by_image:
        d = d['metaperson']
        img_id = d['img_id']
        img_ids.add(img_id)
        # human_made_comps = d['trials'][0]['depth']['keypoint_comparisons_order']
        kpts_depth = d['annotations_truth']['kpts_depth']
        kpts_relative_depth = d['annotations_truth']['kpts_relative_depth']

        # Loop through all the comparisons and calculate their distances
        for comp, comp_res in d['keypoint_comparisons_res'].iteritems():
            kpt1, kpt2 = [int(_k) for _k in comp.split(',')]
            kpt1_depth = kpts_depth[kpts_relative_depth.index(kpt1)]
            kpt2_depth = kpts_depth[kpts_relative_depth.index(kpt2)]
            depth_diff = kpt2_depth - kpt1_depth

            # Collect whatever data is needed for the graphs
            img_id_to_total_generated_majority_vote_comparisons.setdefault(img_id, 0)
            img_id_to_total_generated_majority_vote_comparisons[img_id] += 1

            # If the comparison was correct
            if (abs(depth_diff) < THRESHOLD and comp_res == 0) or np.sign(comp_res) == np.sign(depth_diff):
                img_id_to_correct_generated_majority_vote_comparison_count.setdefault(img_id, 0)
                img_id_to_correct_generated_majority_vote_comparison_count[img_id] += 1

    if plots:
        # Plot majority vote histograms
        # Same as above but with inferred comparisions too
        x = sorted([100 * img_id_to_correct_generated_majority_vote_comparison_count[img_id] / img_id_to_total_generated_majority_vote_comparisons[img_id] for img_id in list(set(img_ids))])
        # the histogram of the data
        bins = np.arange(0, 110, 10)
        n, bins, patches = plt.hist(x, bins, facecolor='green')
        plt.xlabel('Percentage Correct during Comparisons By Metaperson')
        plt.ylabel('Image Count')
        plt.title('Human3.6m, Inferred Comparisons by Metaperson, num_imgs={}'.format(len(img_ids)))
        plt.grid(True)
        plt.show()

        # Percentage right during inferred comparisons. Binned ~50cm each, histogram. Cumulative.
        bins = np.arange(0.0, 1.1, 0.1)
        x = sorted([1.0 * img_id_to_correct_generated_majority_vote_comparison_count[img_id] / img_id_to_total_generated_majority_vote_comparisons[img_id] for img_id in img_ids])
        counts, bins = np.histogram(x, bins=bins, normed=True)
        cdf = np.cumsum(counts)
        plt.bar(bins[:-1] + 0.05, cdf/10, width=0.1)
        plt.xlabel('Percentage Correct during Comparisons By Metaperson')
        plt.ylabel('Cumulative Proportion of Images')
        plt.title('Human3.6m, Inferred Comparisons by Metaperson CDF, num_imgs={}'.format(len(img_ids)))
        plt.grid(True)
        plt.show()

    return (img_id_to_correct_generated_majority_vote_comparison_count,
            img_id_to_total_generated_majority_vote_comparisons)


def wrongness(data, absval=True, proportion=True):
    '''
    Analyze to what magnitude Turkers are wrong.

    Args:
        absval:     bool. Indicates whether to take absolute value of distances.
        proportion: bool. Indicates whether to show proportions of correct
                    in each bin rather than a count.
    '''
    THRESHOLD = 500
    BIN_WIDTH = 200

    (hum_kpt_pair_dist_wrong_lbl,
     hum_kpt_pair_dist_correct_lbl,
     gen_kpt_pair_dist_wrong_lbl,
     gen_kpt_pair_dist_correct_lbl) = get_keypoint_comparison_depths(data, THRESHOLD)

    if absval:
        hum_kpt_pair_dist_wrong_lbl = map(abs, hum_kpt_pair_dist_wrong_lbl)
        hum_kpt_pair_dist_correct_lbl = map(abs, hum_kpt_pair_dist_correct_lbl)
        gen_kpt_pair_dist_wrong_lbl = map(abs, gen_kpt_pair_dist_wrong_lbl)
        gen_kpt_pair_dist_correct_lbl = map(abs, gen_kpt_pair_dist_correct_lbl)

    hum_hist_dataset = [hum_kpt_pair_dist_wrong_lbl, hum_kpt_pair_dist_correct_lbl]
    all_hist_dataset = [gen_kpt_pair_dist_wrong_lbl + hum_kpt_pair_dist_wrong_lbl,
                        gen_kpt_pair_dist_correct_lbl + hum_kpt_pair_dist_correct_lbl]
    hum_flattened = [elem for lst in hum_hist_dataset for elem in lst]
    all_flattened = [elem for lst in all_hist_dataset for elem in lst]
    lower_bound = int(min(hum_flattened)) - int(min(hum_flattened)) % BIN_WIDTH
    upper_bound = int(max(hum_flattened)) - int(max(hum_flattened)) % BIN_WIDTH + BIN_WIDTH
    hum_bins = range(lower_bound, upper_bound, BIN_WIDTH)
    lower_bound = int(min(all_flattened)) - int(min(all_flattened)) % BIN_WIDTH
    upper_bound = int(max(all_flattened)) - int(max(all_flattened)) % BIN_WIDTH + BIN_WIDTH
    all_bins = range(lower_bound, upper_bound, BIN_WIDTH)

    if proportion:
        # TODO: Implement this.
        pass

    n, bins, patches = plt.hist(hum_hist_dataset, bins=hum_bins,
                                edgecolor='black', lw=1.2, stacked=True,
                                label=["human incorrect comparisons",
                                       "human correct comparisons"])
    plt.legend()
    plt.grid(True, axis='y')
    if absval:
        plt.xlabel('Distance between incorrectly labeled keypoint pairs (mm)')
    else:
        plt.xlabel('Absolute Distance between incorrectly labeled keypoint pairs (mm)')
    plt.ylabel('Number of Keypoint Pairs Comparisons')
    num_hum_comps = len(hum_kpt_pair_dist_wrong_lbl) + len(hum_kpt_pair_dist_correct_lbl)
    plt.title('Human3.6m, Human-made Comparisons, num_comparisons={}'.format(num_hum_comps))


    plt.figure()
    n, bins, patches = plt.hist(all_hist_dataset, bins=all_bins,
                                edgecolor='black', lw=1.2, stacked=True,
                                label=["human+generated incorrect comparisons",
                                       "human+generated correct comparisons"])
    plt.legend()
    plt.grid(True, axis='y')
    if absval:
        plt.xlabel('Distance between incorrectly labeled keypoint pairs (mm)')
    else:
        plt.xlabel('Absolute Distance between incorrectly labeled keypoint pairs (mm)')
    plt.ylabel('Number of Keypoint Pairs Comparisons')
    num_all_comps = len(gen_kpt_pair_dist_wrong_lbl) + len(gen_kpt_pair_dist_correct_lbl) + num_hum_comps
    plt.title('Human3.6m, Human-made Comparisons, num_comparisons={}'.format(num_all_comps))

    plt.show()


def agreement(data, plots=True):
    '''
    Calculate similarity metric amongst each of the turker's rankings and group
    and graph for analysis.
    
    TODO: Use Cayley distance as accurate similarity metric
    NOTE: difflib similarity ratio calculated by 2.0 * Matches / Total
         elements over both lists
    '''
    # Group hits by image. 3 Turkers were assigned the same image to annotate.
    img_ids = [h['trials'][0]['img_id'] for h in data]
    img_ids_to_data = { img_id: [] for img_id in img_ids }
    for d in data:
        img_ids_to_data[d['trials'][0]['img_id']].append(d)

    # Create the majority vote comparison results.
    avg_order_agreement_errors = [] # Average similarity score amongst each image
    all_order_agreement_errors = [] # Take all the similarity scores
    best_pair_order_agreemnt_errors = [] # Take most similar pair of orderings
                                         # in each picture.
    worst_pair_order_agreement_errors = [] # Take the least similar pair of
                                           # orderings in each picture.
    random_order_agreement_errors = [] # Compare orderings to random ordering
    for img_id, mini_data_batch in img_ids_to_data.iteritems():
        metaperson = { 'kpts_relative_depth':[], 'img_id':img_id }
        # metaperson['annotations_truth'] = mini_data_batch[0]['annotations_truth']

        for m in mini_data_batch:
            metaperson['kpts_relative_depth'].append(m['trials'][0]['kpts_relative_depth'])
        
        # if any([metaperson['kpts_relative_depth'][0] == ordering for ordering in metaperson['kpts_relative_depth'][1:]]):
        #     print "WE FOUND ONE? A suspicious image where the Turker got them all right"
        #     print metaperson['kpts_relative_depth']
        #     print mini_data_batch[0]['trials'][0]['img_id']

        img_agreement_errors = []
        permutations = itertools.combinations(metaperson['kpts_relative_depth'], 2)
        for ordering1, ordering2 in permutations:
            sm = difflib.SequenceMatcher(None, ordering1, ordering2)
            img_agreement_errors.append(sm.ratio())

        if len(img_agreement_errors) > 0:
            all_order_agreement_errors.extend(img_agreement_errors)
            avg_agreement_error = 1.0 * sum(img_agreement_errors) / len(img_agreement_errors)
            avg_order_agreement_errors.append(avg_agreement_error)
            best_pair_order_agreemnt_errors.append(max(img_agreement_errors))
            worst_pair_order_agreement_errors.append(min(img_agreement_errors))

        # Find similarity metrics against a random permutation as a baseline
        for ordering in metaperson['kpts_relative_depth']:
            random_ordering = np.random.permutation(ordering)
            sm = difflib.SequenceMatcher(None, ordering, random_ordering)
            random_order_agreement_errors.append(sm.ratio())

    if plots:
        hist_datasets = [avg_order_agreement_errors,
                         all_order_agreement_errors,
                         best_pair_order_agreemnt_errors,
                         worst_pair_order_agreement_errors,
                         random_order_agreement_errors]
        hist_titles = ['Human3.6m\nMean Agreement Errors of Turkers on same ' \
                       'Image, num_imgs={}'.format(len(img_ids)),
                       'Human3.6m\nAll Agreement Errors of Turkers on same ' \
                       'Image, num_imgs={}'.format(len(img_ids)),
                       'Human3.6m\nBest Agreement Errors of Turkers on same ' \
                       'Image, num_imgs={}'.format(len(img_ids)),
                       'Human3.6m\nWorst Agreement Errors of Turkers on same ' \
                       'Image, num_imgs={}'.format(len(img_ids)),
                       'Human3.6m\nRandom Agreement Errors of Turkers on same ' \
                       'Image, num_imgs={}'.format(len(img_ids))]

        for x, hist_title, f in zip(hist_datasets, hist_titles, range(len(hist_datasets))):
            plt.figure(f)
            bins = np.arange(0.0, 1.05, 0.1)
            n, bins, patches = plt.hist(x, bins, facecolor='green', lw=1.2,
                                        edgecolor='black', normed=True)
            for item in patches:
                item.set_height(item.get_height()/sum(n))
            plt.ylim(0, 0.45)
            plt.xlabel('Similarity ratio. (2.0 * Matches / Total Elems in Both Lists)')
            plt.ylabel('PDF')
            plt.title(hist_title)
            plt.grid(True, axis='y')
        plt.show()

    return (avg_order_agreement_errors, all_order_agreement_errors,
            best_pair_order_agreemnt_errors, worst_pair_order_agreement_errors,
            random_order_agreement_errors)





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



# worker_comparisons(data_by_hit)
# metaperson_comparisons(data_by_hit)
# wrongness(data_by_hit)
# agreement(data_by_hit)

# scores = [calculate_naive_score(d) for d in data]
# scores = [calculate_dist_score(d) for d in data]

# calculate_base_depth_statistics(data)
# scores_histogram(scores)

# find_images_for_mini_experiment(data)

