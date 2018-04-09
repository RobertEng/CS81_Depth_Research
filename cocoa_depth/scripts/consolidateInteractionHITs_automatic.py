########################################################################
# GENERAL IMPORTS

import os
import sys
import pickle

#######################################################################
# CONSTANTS

MAX_NUM_ANNOTATORS = 5
AGREEMENT_VALUE    = 3

########################################################################
# MONGO CLIENT DATABASE SETUP

from pymongo import MongoClient
_mongo_client = MongoClient()

_cocoa_interactions_db  = _mongo_client.cocoa_interactions
_cocoa_interactions_tmp = _cocoa_interactions_db.cocoa_interactions_to_consolidate
_cocoa_interactions     = _cocoa_interactions_db.cocoa_interactions

########################################################################

def main():
    if len(sys.argv) != 3:
        print "Usage: $> python saveInteractions.py \'assignments file\' \'dictionary key\'"
        print "I.e. : $> python saveInteractions.py ../hits/dict.pkl _rejected_assignments"
        sys.exit(1)

    dict_path = sys.argv[1]
    dict_key  = sys.argv[2]

    saveInteractionHITs( dict_path, dict_key )
    consolidateInteractions()

def consolidateInteractions():

    # extract all the subjects that have been annotated at least once
    coco_subj_id_list = list(set( [s['coco_subj_id'] for s in _cocoa_interactions_tmp.find()] ))
    num_subjects = len(coco_subj_id_list)    

    count           = 0
    inserted        = 0
    not_enough_anno = 0
    num_flagged     = 0

    for subj_id in coco_subj_id_list:
        count += 1
        if count%500 == 0:
            print ' - Completed [%d / %d]' %(count,num_subjects)

        res = _cocoa_interactions_tmp.find({'coco_subj_id':subj_id})
        if res.count() < MAX_NUM_ANNOTATORS:
            not_enough_anno += 1
            continue

        tmp_anno = [elem for elem in res]
        if res.count() > MAX_NUM_ANNOTATORS:
            tmp_anno = sorted( tmp_anno, key=lambda k: k['worker_exp'], reverse=True )
            tmp_anno = tmp_anno[0:MAX_NUM_ANNOTATORS]

        # now tmp_anno contains the mongodb documents of exactly MAX_NUM_ANNOTATORS
        coco_subj_id = tmp_anno[0]['coco_subj_id']
        coco_img_id = tmp_anno[0]['coco_img_id']

        # interactions -> list of lists containing the interactions provided by the 5 annotators
        # i.e. interactions = [[a, b], [b], [b], [a, c], [a, b, c]]    
        interactions = [x['interactions'] for x in tmp_anno]
        # flag -> list containing the flags provided by the 5 annotators
        # i.e. flag = [True, False, False, True, False]
        flags = [x['flag'] for x in tmp_anno]

        # flat_interactions contains all the interactions in a unique list
        # flat_interactions = [a, b, b, b, a, c, a, b, c]
        flat_interactions = [item for sublist in interactions for item in sublist]

        # consolidated annotations[k] keeps only the object ids that have
        # more than k repetitions
        consolidated_interactions = \
            list(set([x for x in flat_interactions if flat_interactions.count(x) >= AGREEMENT_VALUE]))
        consolidated_flag = flags.count(True) >= AGREEMENT_VALUE

        #if DEBUG:
        #    print "============================================================"
        #    print "coco_subj_id: [%d]"%coco_subj_id
        #    print "coco_img_id: [%d]"%coco_img_id
        #    print "------------------------------------------------------------"
        #    print interactions
        #    print flags
        #    print "------------------------------------------------------------"
        #    print "flag: [%s]"%str(consolidated_flag)
        #    print "interactions: [%s]"%str(consolidated_interactions)
        #    print "============================================================" 

        #if consolidated_flag == False:
        mongo_entry = \
        {'coco_subj_id':coco_subj_id,
        'coco_img_id':coco_img_id,
        'flag':consolidated_flag,
        'interactions':consolidated_interactions
        }
        look_up = _cocoa_interactions.find( mongo_entry ).limit(1)
        if look_up.count() == 0:
            _cocoa_interactions.insert( mongo_entry )
            inserted += 1
        elif look_up.count() > 1:
            for elem in look_up:
                print "================================================"
                print elem
                print "================================================"
            sys.exit("Multiple records found in DB. Check Data for Duplicates.")
        #else:
        if consolidated_flag:
            num_flagged += 1
            inserted -= 1

    # print some stats
    print "================================================"
    print " - Number subjects Analyzed:                [%d]" %(num_subjects)
    print " - Subjects with less than [%d] annotators:  [%d]" %(MAX_NUM_ANNOTATORS,not_enough_anno)
    print " - Subjects that were flagged by turkers:   [%d]" %(num_flagged)
    print " - Subjects inserted in cocoa_interactions: [%d]" %(inserted)
    print "================================================"

def saveInteractionHITs( dict_path, dict_key ):

    assignments_dict = pickle.load( open( dict_path, "rb" ) )

    # list with the assignments that are good
    assignments = assignments_dict[dict_key]
    num_assignments = len(assignments)
    count = 0

    for ass in assignments:
        count += 1
        
        if count%500 == 0:
            print " - Inserted [%d] / [%d] assignments"%(count,num_assignments)
    
        worker_id   = ass['worker_id']
        worker_exp  = ass['worker_exp']
        ass_id      = ass['assignment_id']
        trials_data = ass['trials']
    
        for trial_key in trials_data.keys():
            trial = trials_data[trial_key]
        
            mongo_entry = \
            {'coco_subj_id':trial['coco_subj_id'],
            'coco_img_id':trial['coco_img_id'],
            'interactions':trial['interactions'],
            'flag':trial['flag'],
            'worker_id':worker_id,
            'worker_exp':worker_exp,
            'assignment_id':ass_id
            }
        
            look_up = _cocoa_interactions_tmp.find( mongo_entry ).limit(1)
            if look_up.count() == 0:
                _cocoa_interactions_tmp.insert( mongo_entry )
            elif look_up.count() > 1:
                for elem in look_up:
                    print "================================================"
                    print elem
                    print "================================================"
                sys.exit("Multiple records found in DB. Check Data for Duplicates.")

if __name__ == '__main__':
    main()
