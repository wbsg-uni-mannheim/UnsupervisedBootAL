from __future__ import unicode_literals
from sklearn.metrics import f1_score, precision_recall_fscore_support, make_scorer
import random
import numpy as np
import libact 
from libact.utils import zip, inherit_docstring_from
import scipy.sparse as sp
from libact.query_strategies import *
from libact.base.dataset import Dataset
from libact.base.interfaces import QueryStrategy, ContinuousModel, ProbabilisticModel
from collections import Counter
import copy
from itertools import compress
import math
from copy import deepcopy
import time
from learningalgos import *

#extension of libact classes
class ExtDataset(Dataset):


    def __init__(self, X=None, y=None, ids=None):
        
        Dataset.__init__(self, X, y)


        
class UnsupervisedPoolDataset(Dataset):
    
    def __init__(self, X=None, y=None, y_unsupervised = None, ids=None, pool_unsupervised_weights=[], reweight='default'):       
        
        Dataset.__init__(self, X, y)

        if y_unsupervised is None: y_unsupervised = []
        y_unsupervised = np.array(y_unsupervised)   
        
        self._y_unsupervised = y_unsupervised
        self._sample_weights = np.full(len(self._y_unsupervised), 1)
        
        if reweight=='score_based': 
            self._sample_weights = pool_unsupervised_weights #every unsupervised data point is weighted by its distance to the decision boundary
    
    def get_unsupervised_labeled_mask(self):
        return ~np.fromiter((e is None for e in self._y_unsupervised), dtype=bool)
        
    def get_unsupervised_labeled_entries(self):
        return self._X[self.get_unsupervised_labeled_mask()], self._y_unsupervised[self.get_unsupervised_labeled_mask()].tolist()
    
    def get_sample_weights(self):
        return self._sample_weights[self.get_unsupervised_labeled_mask()]

    def len_unsupervised_labeled(self):
        return self.get_unsupervised_labeled_mask().sum()
    
    def update_unsupervised(self, entry_id, new_label):
        self._y_unsupervised[entry_id] = new_label
        
    def update_weights (self, weight):
        self._sample_weights[self.get_labeled_mask()] = weight
    
    def update_single_weight (self, entry_id, weight):
        self._sample_weights[entry_id] = weight
   
    def unsupervised_format_sklearn(self):
        X, y = self.get_unsupervised_labeled_entries()
        return X, np.array(y)
    


    def _vote_disagreement(self, votes, models_count):
        ret = []
        for candidate in votes:
            ret.append(0.0)
            lab_count = {}
            for lab in candidate:
                lab_count[lab] = lab_count.setdefault(lab, 0) + 1

            # Using vote entropy to measure disagreement
            for lab in lab_count.keys():
                ret[-1] -= float(lab_count[lab]) / float(models_count) * \
                    math.log(float(lab_count[lab]) / float(models_count))
        return ret


    
class NoisyUncertaintySampling(libact.query_strategies.UncertaintySampling):
    
    def _get_scores(self):
        
        dataset = self.dataset
        self.model.train(dataset, sample_weight=dataset._sample_weights[dataset.get_unsupervised_labeled_mask()])
        
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()

        if isinstance(self.model, ProbabilisticModel):
            dvalue = self.model.predict_proba(X_pool)
        elif isinstance(self.model, ContinuousModel):
            dvalue = self.model.predict_real(X_pool)

        if self.method == 'lc':  # least confident
            score = -np.max(dvalue, axis=1)

        elif self.method == 'sm':  # smallest margin
            if np.shape(dvalue)[1] > 2:
                # Find 2 largest decision values
                dvalue = -(np.partition(-dvalue, 2, axis=1)[:, :2])
            score = -np.abs(dvalue[:, 0] - dvalue[:, 1])

        elif self.method == 'entropy':
            score = np.sum(-dvalue * np.log(dvalue), axis=1)
        return zip(unlabeled_entry_ids, score)

class RandomSampling_(QueryStrategy):

   #no seeding

    def __init__(self, dataset, **kwargs):
        super(RandomSampling_, self).__init__(dataset, **kwargs)


    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        unlabeled_entry_ids, _ = dataset.get_unlabeled_entries()
        entry_id = unlabeled_entry_ids[
            random.randint(0, len(unlabeled_entry_ids))]
        return entry_id
    
class QueryByCommittee_(QueryByCommittee):

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
        
        X_un, y_un = self.dataset.get_unsupervised_labeled_entries()

        if self.disagreement == 'vote':
            # Let the trained students vote for unlabeled data
            votes = np.zeros((len(X_pool), len(self.students)))
            
            for i, student in enumerate(self.students):
                votes[:, i] = student.predict(X_pool)
               

            vote_entropy = self._vote_disagreement(votes)
                   
          
            committee_pool = list(zip(unlabeled_entry_ids, np.array(y_un)[unlabeled_entry_ids], votes, dataset.get_sample_weights()[unlabeled_entry_ids],vote_entropy))
            max_entropy = np.max(vote_entropy)
            
            
            candidates = filter(lambda x, max_entropy=max_entropy: x[4]==max_entropy and len(np.where(x[2]==x[1])[0])<0.5*len(x[2]), committee_pool)

            candidates_ids = map(lambda x:x[0], candidates)
            
            if len(candidates_ids)==0 :
            
                ask_idx = self.random_state_.choice(
                    np.where(np.isclose(vote_entropy, np.max(vote_entropy)))[0])
                return unlabeled_entry_ids[ask_idx]
            
            else:  
            
                ask_idx = self.random_state_.choice(candidates_ids)
                return ask_idx
            
        elif self.disagreement == 'kl_divergence':
            proba = []
            for student in self.students:
                proba.append(student.predict_proba(X_pool))
            proba = np.array(proba).transpose(1, 0, 2).astype(float)

            avg_kl = self._kl_divergence_disagreement(proba)
            ask_idx = random.choice(
                    np.where(np.isclose(avg_kl, np.max(avg_kl)))[0])
            return unlabeled_entry_ids[ask_idx]
        
    
