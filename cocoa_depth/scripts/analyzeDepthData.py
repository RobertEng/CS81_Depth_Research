'''
Helper files to take the output pickle file from processAssignments() and
analyze it.
'''

################################################################################
# Config

pickle_file = "../hits/human/cocoa_test_completed_50_DepthHITS_2017-06-09_11-12-18.pkl"

# dataset = 'coco'
dataset = 'human'

HUMAN_ANNOTATION_FILE = '/home/ubuntu/datasets/human3.6/annotations/human36m_train.json'

################################################################################

def clean_pickle_data(pickle_file):
    assert (dataset == 'human'), "clean_pickle_data only accepts human3.6 data"
    import pickle
    pickle_data = pickle.load(open(pickle_file, 'r'))
    pickle_data = pickle_data['_good_assignments']

    # Sample image id
    # pickle_data[0]['trials'][0]['img_id']

    # Get the img_ids to be looking for from pickle_data
    humpickleids = [h['trials'][0]['img_id'] for h in pickle_data]

    # Get the ground truth annotations from human dataset
    import json
    with open( HUMAN_ANNOTATION_FILE ) as f:
        _human_dataset = json.load(f)

    # Match the ground truth annotations with the pickle data annotations
    for i, h in enumerate(_human_dataset['images']):
        if h['id'] in humpickleids:
            pickle_data[humpickleids.index(h['id'])]['images_truth'] = h
            pickle_data[humpickleids.index(h['id'])]['annotations_truth'] = _human_dataset['annotations'][i]

    return pickle_data

    # Remove the data which doesn't have an annotation associated with it.
    tmp = []
    for h in pickle_data:
        if 'annotations_truth' in h:
            tmp.append(h)
        else:
            print 'uhohs'
    pickle_data = tmp

    # Clean up annotations_truth data. Remove neck keypoint.
    for i, h in enumerate(pickle_data):
        myList = h['annotations_truth']['kpts_3d'][1::3]
        myList[1:2] = [] # Remove neck keypoint
        ordered_truth.append([i[0] for i in sorted(enumerate(myList), key=lambda x:x[1], reverse=True)])


    ordered_truth = []
    for h in pickle_data:
        if 'annotations_truth' not in h:
            ordered_truth.append([])
            print 'uhohs'
        else:
            myList = h['annotations_truth']['kpts_3d'][1::3]
            myList[1:2] = [] # Remove neck keypoint
            ordered_truth.append([i[0] for i in sorted(enumerate(myList), key=lambda x:x[1], reverse=True)])
    print ordered_truth

    ## DUMP KEYPOINTCOMARPISONS ORDER AND RES TO FILE TO BE READ BY JS
    import json
    keycmp_orders = [h['trials'][0]['depth']['keypoint_comparisons_order'] for h in pickle_data]
    keycmp_reses = [h['trials'][0]['depth']['keypoint_comparisons_res'] for h in pickle_data]
    keycmps = {'keycmp_orders':keycmp_orders, 'keycmp_reses':keycmp_reses}

    with open("keycmps.json", 'w') as f:
        f.write(json.dumps(keycmps))



pickle_data = clean_pickle_data(pickle_file)





