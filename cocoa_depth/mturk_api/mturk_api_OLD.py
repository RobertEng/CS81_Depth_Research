from boto.mturk.connection import MTurkConnection
from boto.mturk.question import ExternalQuestion
from boto.mturk.qualification import Qualifications, PercentAssignmentsApprovedRequirement, LocaleRequirement, NumberHitsApprovedRequirement
from boto.mturk.connection import MTurkRequestError

import os
import pickle
from pprint import pprint
import time

from pymongo import MongoClient
import json

#######################################################################
# MECHANICAL TURK SETUP
 
#_host = 'mechanicalturk.sandbox.amazonaws.com'
_host = 'mechanicalturk.amazonaws.com'

#######################################################################
# CONSTANTS

_STARTING_HIT = 11
_NUMBER_HITS = 2683
_NUMBER_HIT_ASSIGNMENTS = 1

########################################################################
# MONGO CLIENT DATABASE SETUP

_mongo_client = MongoClient()
_mongo_db = _mongo_client.cocoa_5000

_mongo_coll_1 = _mongo_db.interactions_amt_gui_data

_mongo_coll_2 = _mongo_db.coco_subj_id2coco_img_id

_mongo_coll_3 = _mongo_db.interactions_hit_id2coco_subj_id

_mongo_coll_4 = _mongo_db.coco_subj_id2interactions_hit_id

_mongo_coll_5 = _mongo_db.interactions_amt_gui_workers
_mongo_coll_6 = _mongo_db.interactions_amt_gui_blocked_workers


def getHITType():
    # Changing this will add another hit type and might mess up later fetches...
    # Only change if you know what you are doing...

    _mtc = MTurkConnection( host = _host )

    _title = "Select Person-Object Interactions in Images"
    _description="Please click on all the objects (or other people) that the highlighted person is interacting with."
    _reward = _mtc.get_price_as_price(0.10)
    _duration = 60 * 10
    _keywords = "person, people, image, images, object, objects, actions, interactions"
    _approval_delay = 60 * 60 * 24 * 5
    _qualifications = Qualifications()
    _qualifications.add(PercentAssignmentsApprovedRequirement('GreaterThanOrEqualTo', 98, required_to_preview=True))
    _qualifications.add(NumberHitsApprovedRequirement('GreaterThanOrEqualTo', 100, required_to_preview=True))
    _qualifications.add(LocaleRequirement('EqualTo', 'US', required_to_preview=True))

    return _mtc.register_hit_type(title=_title, description=_description, reward=_reward, duration=_duration, keywords=_keywords, approval_delay=_approval_delay, qual_req=_qualifications)


def createHITs():
    hostDomain = 'https://aws-ec2-cit-mronchi.org'
    
    setIds = [1,3,4,6,8,9]
    setIds.extend(range(_STARTING_HIT, _STARTING_HIT+_NUMBER_HITS))

    mtc = MTurkConnection( host = _host )

    hits = []

    hitType = getHITType()[0]
    max_assignments = _NUMBER_HIT_ASSIGNMENTS
    lifetime = 60 * 60 * 24 * 5

    max_hits = 2694
    count = 0

    for setId in setIds:
        external_url = hostDomain + '/mturk_interactions/' + str(setId)
        print external_url
        q = ExternalQuestion(external_url=external_url,
                             frame_height=1000)

        hit = mtc.create_hit(hit_type=hitType.HITTypeId,
                             question=q,
                             max_assignments=max_assignments,
                             lifetime=lifetime)

        hits.append(hit[0])
        count += 1
        if count >= max_hits:
            pass

    if 'MTURK_STORAGE_PATH' in os.environ:
        time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        hit_name = 'cocoa_5000_' + str(_NUMBER_HITS) + '_InteractionHITS'
        filename = os.path.join(os.environ['MTURK_STORAGE_PATH'], hit_name + '_' + time_stamp + ".pkl")
        print "Storing created hit data at %s" % (filename, )
        with open(filename, 'wb') as f:
            pickle.dump(hits, f)
    else:
        print "WARNING: MTURK_STORAGE_PATH not set in env. Unable to save hit data."


def getReviewableHITs(verbose=True):

    mtc = MTurkConnection( host = _host )
    page_size = 50
    hitType = getHITType()[0]
    hits = mtc.get_reviewable_hits( page_size = page_size )

    if verbose:
        print "Total results to fetch %s " % hits.TotalNumResults
        print "Request hits page %i" % 1

    total_pages = float(hits.TotalNumResults)/page_size
    int_total = int(total_pages)
    if(total_pages - int_total > 0):
        total_pages = int_total + 1
    else:
        total_pages = int_total

    pn = 1
    while pn < total_pages:
        pn = pn + 1
        if verbose:
            print "Request hits page %i" % pn
            temp_hits = mtc.get_reviewable_hits(hit_type=hitType.HITTypeId,
                                                page_size=page_size,
                                                page_number=pn)
            hits.extend(temp_hits)

    return hits


def deleteAllHits():

    # this function should probably take an input parameter of a pickle file with the hits to be disposed...
    mtc = MTurkConnection(host=_host)
    for hit in mtc.get_all_hits():
        mtc.disable_hit(hit.HITId)


def processHITs(verbose=True, approveAll=False, deleteAll=False, insertComparisons=False):

	mtc = MTurkConnection(host=_host)
	hits = getReviewableHITs(verbose)
	# store hit info here, for persistence
	_hits_vector = []
	_rejected_hits = []
	_flagged_hits = []
	# stats variables
	worker_ids = set()

	for hit in hits:
		assignments = mtc.get_assignments(hit.HITId,page_size=50)
		for assignment in assignments:
			worker_ids.add(assignment.WorkerId)
			if verbose:
				print "Answers of the worker: [%s]" % assignment.WorkerId
			
			_worker_id = ''
			_worker_exp = 0
			_hit_id = 0
			_assignment_id = ''
			_gui_rating = ''
			_hit_comment = ''
			_hit_rt = 0
			_hit_it = 0
			_trials_results = ''
			_hit_interactions_str  = ''
			_hit_reject_flag = False
			_hit_flag = False
			
			for question_form_answer in assignment.answers[0]:
				key = question_form_answer.qid
				value = question_form_answer.fields
				
				if key == '_worker_id':
					_worker_id = value[0]
					if verbose:
						print " - Worker ID: [%s]" % (_worker_id)
				elif key == '_worker_exp':
					_worker_exp = int(value[0])
					if verbose:
						print " - Worker experience: [%d]" % (_worker_exp)
				elif key == '_hit_id':
					_hit_id = int(value[0])
					if verbose:
						print " - HIT ID: [%d]" % (_hit_id)   
				elif key == '_assignment_id':
					_assignment_id = value[0]
					if verbose:
						print " - Assignment ID: [%s]" % (_assignment_id)
				elif key == '_gui_rating':
					_gui_rating = value[0]
					try: 
						_gui_rating = int(_gui_rating)
					except ValueError:
						_gui_rating = -1
					if verbose:
						print " - GUI rating: [%d/10]" % (_gui_rating)         
				elif key == '_hit_comment':
					_hit_comment = value[0]
					if verbose:
						print " - HIT comment: [%s]" % (_hit_comment)
				elif key == '_hit_rt':
					_hit_rt = int(value[0])
					if verbose:
						print " - HIT response time: [%d]" % (_hit_rt)
				elif key == '_hit_it':
					_hit_it = int(value[0])
					if verbose:
						print " - HIT instruction time: [%d]" % (_hit_it)
				elif key == '_trials_results':
					_trials_results = value[0]
					if verbose:
						print " - All HIT's trials results: [%s]" % (_trials_results)    
				elif key == '_hit_interactions_str':
					_hit_interactions_str = value[0]
					if verbose:
						print " - HIT interactions string: [%s]" % (_hit_interactions_str)    
				elif key == '_hit_reject_flag':
					_hit_reject_flag = value[0]
					if str(_hit_reject_flag) == 'false':
						_hit_reject_flag = False
					else:
						_hit_reject_flag = True
					if verbose:
						print " - HIT reject flag: [%s]" % (str(_hit_reject_flag))
				elif key == '_hit_flag':
					_hit_flag = value[0]
					if _hit_flag == 'Yes':
						_hit_flag = True
					else:
						_hit_flag = False
					if verbose:
						print " - HIT information flag: [%s]" % (str(_hit_flag))    
				else:
					print "<----------------------------->"
					print "ERROR: unknown key [%r]" % (key,)
					print "Relevant info:"
					pprint(vars(assignment))
					pprint(vars(question_form_answer))
					print "Exiting..."
					print "<----------------------------->"
					return
			
            #if insertComparisons:
            #    pass
                # insert the comparisons into the database

			_hit_data = assignment.__dict__.copy()
			del _hit_data['answers']

			_hit_data['_worker_id'] = _worker_id
			_hit_data['_worker_exp'] = _worker_exp
			_hit_data['_hit_id'] = _hit_id
			_hit_data['_assignment_id'] = _assignment_id
			_hit_data['_gui_rating'] = _gui_rating
			_hit_data['_hit_comment'] = _hit_comment
			_hit_data['_hit_rt'] = _hit_rt
			_hit_data['_hit_it'] = _hit_it
			_hit_data['_trials_results'] = _trials_results
			_hit_data['_hit_interactions_str'] = _hit_interactions_str
			_hit_data['_hit_reject_flag'] = _hit_reject_flag
			_hit_data['_hit_flag'] = _hit_flag

			_hits_vector.append(_hit_data)
			
			if _hit_reject_flag:
				_rejected_hits.append(_hit_data)
				print "<----------------------------->"
				print "This HIT is low quality - Will be rejected."
				print "Relevant info:"
				pprint(vars(assignment))
				for question_form_answer in assignment.answers[0]:
					pprint(vars(question_form_answer))				
				print "<----------------------------->"
				try:
					mtc.reject_assignment(assignment.AssignmentId)
				except MTurkRequestError:
					print "Could not reject [%s]" %(assignment.AssignmentId) 
			else:			
				if _hit_flag:
					_flagged_hits.append(_hit_data)
					print "<----------------------------->"
					print "This HIT has been flagged by turker."
					print "Relevant info:"
					pprint(vars(assignment))
					for question_form_answer in assignment.answers[0]:
						pprint(vars(question_form_answer))				
					print "<----------------------------->"

				if approveAll:
					try:
						mtc.approve_assignment(assignment.AssignmentId)
					except MTurkRequestError:
						print "Could not approve [%s]" %(assignment.AssignmentId) 
			if verbose:
				print "<----------------------------->"
					
			if deleteAll:
				mtc.disable_hit(hit.HITId)

	# print out some stats
	print "Number of HITs = [%d]" % (len(_hits_vector),)
	print "Number of distinct workers = [%d]" % (len(worker_ids),)
	print "Number of rejected HITs = [%d]" % (len(_rejected_hits),)
	print "Number of flagged HITs = [%d]" % (len(_flagged_hits),)
	
	return_dict = {"_all_hits":_hits_vector,"_rejected_hits":_rejected_hits,"_flagged_hits":_flagged_hits}

	if 'MTURK_STORAGE_PATH' in os.environ:
		time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
		hit_name = "completed_cocoa_5000"
		filename = os.path.join(os.environ['MTURK_STORAGE_PATH'], hit_name + '_' + time_stamp + ".pkl")
		print "Storing collected hit data at %s" % (filename)
		with open(filename, 'wb') as f:
			pickle.dump(return_dict, f)
	else:
		print "WARNING: MTURK_STORAGE_PATH not set in env. Unable to save hit data."
	
	return return_dict
	
def getReviewableAssignments( verbose = False ):
	
	_assignments = []
	
	_mtc = MTurkConnection( host = _host )
	
	_num_hits = sum(1 for _ in _mtc.get_all_hits())
	print "Number HITs: [%d]" %(_num_hits)
	count = 0
	for hit in _mtc.get_all_hits():
		count += 1
		print count
		if verbose:
			print "-------------------------------------"
			print hit.HITId
			print ""
			_hit_assignments = _mtc.get_assignments(hit.HITId,page_size=50)
			_num_assignments = sum(1 for _ in _hit_assignments)
			print " -  number assignments: [%d]" %(_num_assignments)
		
		_assignments.extend([_assignment for _assignment in _mtc.get_assignments(hit.HITId,page_size=50)])
	return _assignments
	   
def processAssignments( workerID = '', exp_name = '', verbose = False ):

	_mtc = MTurkConnection( host = _host )
	
	_assignments = getReviewableAssignments(verbose)
	
	# store assignments info here, for persistence
	_all_assignments = []
	_flagged_assignments = []
	_rejected_assignments = []
	_good_assignments = []
	_error_assignments = []
	
	# stats variables
	worker_ids = set()
	
	print "Got Assignments."
	count = 0
	for _assignment in _assignments:
		count += 1
		
		#if count % 500 == 0:
		print "Analyzing [%d]" %count
		
		if workerID != '':
			if _assignment.WorkerId != workerID:
				continue		
		
		worker_ids.add(_assignment.WorkerId)
		if verbose:
			print "Answers of the worker: [%s]" % _assignment.WorkerId
		
		_worker_id = ''
		_worker_exp = 0
		_hit_id = 0
		_assignment_id = ''
		_gui_rating = ''
		_hit_comment = ''
		_hit_rt = 0
		_hit_it = 0
		_trials_results = ''
		_hit_interactions_str  = ''
		_hit_reject_flag = False
		_hit_flag = False
			
		for question_form_answer in _assignment.answers[0]:
			key = question_form_answer.qid
			value = question_form_answer.fields
				
			if key == '_worker_id':
				_worker_id = value[0]
				if verbose:
					print " - Worker ID: [%s]" % (_worker_id)
			elif key == '_worker_exp':
				_worker_exp = int(value[0])
				if verbose:
					print " - Worker experience: [%d]" % (_worker_exp)
			elif key == '_hit_id':
				_hit_id = int(value[0])
				if verbose:
					print " - HIT ID: [%d]" % (_hit_id)   
			elif key == '_assignment_id':
				_assignment_id = value[0]
				if verbose:
					print " - Assignment ID: [%s]" % (_assignment_id)
			elif key == '_gui_rating':
				_gui_rating = value[0]
				try: 
					_gui_rating = int(_gui_rating)
				except ValueError:
					_gui_rating = -1
				if verbose:
					print " - GUI rating: [%d/10]" % (_gui_rating)         
			elif key == '_hit_comment':
				_hit_comment = value[0]
				if verbose:
					print " - HIT comment: [%s]" % (_hit_comment)
			elif key == '_hit_rt':
				_hit_rt = int(value[0])
				if verbose:
					print " - HIT response time: [%d]" % (_hit_rt)
			elif key == '_hit_it':
				_hit_it = int(value[0])
				if verbose:
					print " - HIT instruction time: [%d]" % (_hit_it)
			elif key == '_trials_results':
				_trials_results = value[0]
				if verbose:
					print " - All HIT's trials results: [%s]" % (_trials_results)    
			elif key == '_hit_interactions_str':
				_hit_interactions_str = value[0]
				if verbose:
					print " - HIT interactions string: [%s]" % (_hit_interactions_str)    
			elif key == '_hit_reject_flag':
				_hit_reject_flag = value[0]
				if str(_hit_reject_flag) == 'false':
					_hit_reject_flag = False
				else:
					_hit_reject_flag = True
				if verbose:
					print " - HIT reject flag: [%s]" % (str(_hit_reject_flag))
			elif key == '_hit_flag':
				_hit_flag = value[0]
				if _hit_flag == 'Yes':
					_hit_flag = True
				else:
					_hit_flag = False
				if verbose:
					print " - HIT information flag: [%s]" % (str(_hit_flag))    
			else:
				print "<----------------------------->"
				print "ERROR: unknown key [%r]" % (key,)
				print "Relevant info:"
				pprint(vars(_assignment))
				pprint(vars(question_form_answer))
				print "Exiting..."
				print "<----------------------------->"
				#return
		
		#if insertComparisons:
		#    pass
		# insert the comparisons into the database
		_assignment_data = _assignment.__dict__.copy()
		
		del _assignment_data['answers']
		
		_assignment_data['_worker_id'] = _worker_id
		_assignment_data['_worker_exp'] = _worker_exp
		_assignment_data['_hit_id'] = _hit_id
		_assignment_data['_assignment_id'] = _assignment_id
		_assignment_data['_gui_rating'] = _gui_rating
		_assignment_data['_hit_comment'] = _hit_comment
		_assignment_data['_hit_rt'] = _hit_rt
		_assignment_data['_hit_it'] = _hit_it
		_assignment_data['_trials_results'] = _trials_results
		_assignment_data['_hit_interactions_str'] = _hit_interactions_str
		_assignment_data['_hit_reject_flag'] = _hit_reject_flag
		_assignment_data['_hit_flag'] = _hit_flag

		_all_assignments.append(_assignment_data)

		_polished_data = {}
		_polished_data['response_time'] = _assignment_data['_hit_rt']
		_polished_data['worker_id'] = _assignment_data['_worker_id']
		_polished_data['worker_exp'] = _assignment_data['_worker_exp']
		_polished_data['assignment_id'] = _assignment_data['_assignment_id']
		_polished_data['hit_id'] = _assignment_data['_hit_id']
		_polished_data['response_time'] = _assignment_data['_hit_rt']
		
		_hit_coco_subj_ids = _mongo_coll_3.find_one({'_amt_hit_id':_hit_id})['_coco_subjs_ids']
		_polished_data['coco_subj_ids'] = _hit_coco_subj_ids
		
		_hit_coco_img_ids = [_mongo_coll_2.find_one({'_coco_subj_id':x})['_coco_img_id'] for x in _hit_coco_subj_ids]
		_polished_data['coco_img_ids'] = _hit_coco_img_ids
		
		_trials_results_dict = json.loads(_assignment_data['_trials_results'])
		_hit_trials = {}
		_error = False
		for key in _trials_results_dict.keys():
		
			_trial = _trials_results_dict[key]
			
			try:
				_objs_info = json.loads(_mongo_coll_1.find_one({'_coco_img_id':_hit_coco_img_ids[_hit_coco_subj_ids.index(_trial['_coco_subj_id'])]})['_obj_info_dict'])
				_interactions = [ _objs_info[x]['_coco_ann_id'] for x in _trial['_interactions_str'].split(',')[:-1] ]
			except:
				_interactions = _trial['_interactions_str'].split(',')[:-1]
				_error = True
				
			_hit_trials[_trial['_coco_subj_id']] = \
			{'coco_subj_id': _trial['_coco_subj_id'],
			'coco_img_id': _mongo_coll_2.find_one({'_coco_subj_id':_trial['_coco_subj_id']})['_coco_img_id'],
			'flag': _trial['_trial_flag'],
			'response_time': _trial['_trial_rt'],
			'interactions': _interactions }
		
		_polished_data['trials'] = _hit_trials
		if _error:
			_error_assignments.append( _polished_data )
		else:
			if _hit_reject_flag:
				_rejected_assignments.append( _polished_data )
				if verbose:
					print "<----------------------------->"
					print "This assignment is low quality - Will be rejected."
					print "Relevant info:"
					pprint(vars(_assignment))
					for question_form_answer in _assignment.answers[0]:
						pprint(vars(question_form_answer))
					print "<----------------------------->"
			else:
				if _hit_flag:
					_flagged_assignments.append( _polished_data )
					if verbose:
						print "<----------------------------->"
						print "This assignment has been flagged by turker."
						print "Relevant info:"
						pprint(vars(_assignment))
						for question_form_answer in _assignment.answers[0]:
							pprint(vars(question_form_answer))
						print "<----------------------------->"
				else:
					_good_assignments.append( _polished_data )

		if verbose:
			print "<----------------------------->"

	# print out some stats
	print "Distinct workers:               [%d]" % (len(worker_ids),)
	print "Total number of assignments:    [%d]" % (len(_all_assignments),)
	print "Rejected assignments:           [%d]" % (len(_rejected_assignments),)
	print "Flagged assignments:            [%d]" % (len(_flagged_assignments),)
	print "Good assignments:               [%d]" % (len(_good_assignments),)
	print "Error assignments:              [%d]" % (len(_error_assignments),)
	
	return_dict = {
		"_all_assignments":_all_assignments,
		"_rejected_assignments":_rejected_assignments,
		"_flagged_assignments":_flagged_assignments,
		"_good_assignments":_good_assignments,
		"_error_assignments":_error_assignments
	} 

	if 'MTURK_STORAGE_PATH' in os.environ:
		
		time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
		name = "completed"
		filename = os.path.join(os.environ['MTURK_STORAGE_PATH'], name + '_' + exp_name + '_' + time_stamp + ".pkl")
		print "Storing collected hit data at %s" % (filename)
		with open(filename, 'wb') as f:
			pickle.dump(return_dict, f)
	else:
		print "WARNING: MTURK_STORAGE_PATH not set in env. Unable to save hit data."
	
	return return_dict
