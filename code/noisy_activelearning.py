from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.labelers import IdealLabeler

from sklearn.metrics import f1_score, precision_recall_fscore_support, make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from libact.query_strategies import *
from sklearn.metrics import classification_report
import libact
from sklearn.model_selection import cross_val_score, StratifiedKFold,cross_validate
import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from displayutils import*
from libact_datasetext import *
from learningalgos import *
import time
from sklearn.metrics import confusion_matrix
import collections
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
import xgboost as xgb
from committee_models import *
from sklearn.tree import _tree
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
from learning import *
from scipy.spatial import distance


def noisy_active_learning(al, query_strategy, max_quota, num_runs, model, setting, reweight = None, warm_start=False):

    X= al['pool_data'].values
    y= al['pool_labels']
    ids = np.asarray(al['ids'])

    X_val= al['validation_data'].values
    y_val=al['validation_labels']
    
    evaluation_ds = Dataset(X_val, y_val)
    
    training_accuracy_scores= []
    training_f1_scores= []
    test_accuracy_scores=[]
    test_f1_scores= []
    pool_correctness = []
    
    
    for i in range(num_runs):
        oracle = IdealLabeler(ExtDataset(X,y,ids))
        
        pool,labeled_weight = initializePool(al, reweight, X, ids, y, warm_start)
        
        if (max_quota == -1): max_quota = pool_labels.size

        model_type = getLearningModel(model, warm_start)
              
        qs = getQueryStrategy(query_strategy, pool, model)
        train_f1, test_f1, pool_correctness_, model_, pool_ids = run_noisy_al (pool, evaluation_ds, oracle, model_type, qs, max_quota, i, al, setting, query_strategy,reweight, labeled_weight, warm_start=warm_start)

        training_f1_scores.append(train_f1)
        test_f1_scores.append(test_f1)
        pool_correctness.append(pool_correctness_)
        
        if (i == num_runs-1):
            print"Feature importances for the last AL run"
            showFeatureImportances(al['pool_data'].columns.values.tolist(), model_, model)
        
    plotResultsMultipleRuns(max_quota, training_f1_scores,  test_f1_scores, ('Experiment Result: '+model+' w '+query_strategy+' sampling'))
    
    return test_f1_scores, pool_correctness


def initializePool(al, reweight, X, ids, oracle, warm_start):
    print("Initialize pool")
    pool_labels = np.asarray([None]*len(al['pool_labels']))
       
    pool_unsupervised_labels = np.asarray([None]*len(al['pool_labels']))
    pool_unsupervised_labels[al['bootstrapping_indices']] = al['bootstrapping_labels']
    pool_unsupervised_weights= np.asarray([0.0]*len(al['pool_labels']))

   
    if reweight == 'score_based':
        bootstrapping_weights = []

        for w in al['bootstrapping_scores']:
            #normalize weights so that they are between 0-1
            nom = abs(w-al['bootstrapping_threshold']) 
            if w<al['bootstrapping_threshold']: den = al['bootstrapping_threshold'] - np.min(al['bootstrapping_scores'])
            else: den = np.max(al['bootstrapping_scores']) - al['bootstrapping_threshold']
            score = nom/den
            if score < 0.0: score=0.0
            
            weight = round(score,1)
            bootstrapping_weights.append(weight)
            

        pool_unsupervised_weights[al['bootstrapping_indices']] = bootstrapping_weights

        #add most certain positive and most certain negative in the pool just for the initialization of QBC strategy: to fix
        combined = list(zip(pool_unsupervised_labels,pool_unsupervised_weights, np.arange(0,len(pool_unsupervised_weights))))
        pos  = filter(lambda x: x[0]==1, combined)
        neg = filter(lambda x: x[0]==0, combined)
        pos_weight = map(lambda x: x[1], pos)
        neg_weight = map(lambda x: x[1], neg)
        sure_positive =filter(lambda x,pos_w =max(pos_weight): (x[0]==1 and x[1]==pos_w) ,combined)
        sure_negative =filter(lambda x,neg_w =max(neg_weight): (x[0]==0 and x[1]==neg_w) ,combined)

        sure_positive_indices = map(lambda x:x[2],sure_positive)
        sure_negative_indices = map(lambda x:x[2],sure_negative)
        
        #take only one
        sure_negative_indices_sample = sure_negative_indices[0]
        sure_positive_indices_sample = sure_positive_indices[0]
        
        pool_labels[sure_positive_indices_sample]=1
        pool_labels[sure_negative_indices_sample]=0
        pool_unsupervised_labels[sure_positive_indices] = 1 
        pool_unsupervised_labels[sure_negative_indices_sample] = 0
        pool_unsupervised_weights[sure_positive_indices_sample] = max(pos_weight)
        pool_unsupervised_weights[sure_negative_indices_sample] = max(neg_weight)
       
        #fix weight for labeled data points
        weights_sum = np.sum(pool_unsupervised_weights)
        
        if warm_start:
            labeled_weight = 1
        else: labeled_weight = weights_sum/10
            
        ## new block above
        pool =  UnsupervisedPoolDataset(X, pool_labels, pool_unsupervised_labels, ids, pool_unsupervised_weights, reweight)

    else: pool =  UnsupervisedPoolDataset(X, pool_labels, pool_unsupervised_labels, ids, reweight)
    

    return pool,labeled_weight

def getQueryStrategy(query_strategy, pool, model=None):
    print("Initialize Query Strategy")
    if query_strategy == 'uncertainty':        
        qs = NoisyUncertaintySampling(pool, method='lc', model=RandomForest_(noisy=True))
    elif query_strategy == 'random':
        qs = RandomSampling(pool)    
    elif 'default_committee' in query_strategy:
        qs = QueryByCommittee_(pool, # Dataset object
            models=[
                RandomForest_(),
                DecisionTree_(),
                SVC_(gamma='auto',kernel='linear',probability=True),
                LogisticRegression_(),
                XGBClassifier_(objective="binary:logistic")
            ]
        )
    
  
    return qs
    
def getLearningModel(model, warm_start=False):
    print("Initialize Learning Model")
    if model == "rf" and warm_start: model_type = RandomForest_(warm_start=True, n_estimators =10)
    elif model == "logr": model_type = LogisticRegression_(random_state=1)
    elif model == "dt": model_type = DecisionTree_(random_state=1)
    elif model == "svm": model_type = SVC_(random_state=1, gamma='auto',kernel='linear')
    elif model == "xgboost": model_type = XGBClassifier_(objective="binary:logistic", random_state=1)
    elif model == "linr": model_type = LinearRegression_()           
    else : print "Unknown model type."
    return model_type
    

    
def run_noisy_al(pool, test, oracle, model, qs, quota, run, al, setting,qs_name,reweight=None, labeled_weight=2.0, warm_start=False):

    if 'model_validation' in qs_name:
        model_validation = True
    if 'relabeling' in qs_name:
        relabeling = True
    
    start_time = time.time()
    pool_ids = []
    
    E_in_f1, E_out_f1, E_out_P, E_out_R, unsupervised_correctness = [], [], [], [], []
    E_in_f1_regressor, E_out_f1_regressor = [], []  
    labels = []
    correctedlabels = []
    confidence_queries = []
    initial_weights = deepcopy(pool.get_sample_weights())
    initial_weights_queries = []
    print("Labeled weight:", labeled_weight)
    print("Warm start:", model.model.warm_start)

    print_progress(0, quota, prefix = 'AL RUN: '+str(run), suffix = 'Complete')
    
    for x in range(quota): 
                
        print_progress(x + 1, quota, prefix = 'AL RUN: '+str(run), suffix = 'Complete')                     
        # Standard usage of libact objects
        ask_id = qs.make_query()
        query_conf = pool._sample_weights[ask_id]
        confidence_queries.append(query_conf)
        initial_weights_queries.append(initial_weights[ask_id])
        X, _ = zip(*pool.data)
        
        
        lb = oracle.label(X[ask_id])
        pool.update(ask_id, lb)
        
        corrected = 0
        if pool._y_unsupervised[ask_id] != lb :
            corrected = 1
            pool.update_unsupervised(ask_id, lb)
                 
        correctedlabels.append(corrected)
        labels.append(lb)

        # update weights of labeled pairs
        if reweight!= None:
            weight = labeled_weight
            pool.update_single_weight(ask_id, weight)
            

        if model.model.warm_start and 1 in pool._y and 0 in pool._y and x>0 :
            model.model.n_estimators +=2
            model.train(pool)
        
        else: 
            
            model.train(Dataset(pool._X,pool._y_unsupervised), sample_weight=pool.get_sample_weights())
                
        poolx, pooly = pool.get_unsupervised_labeled_entries()
        testx, testy = test.format_sklearn()
        
        E_in_f1 = np.append(E_in_f1, f1_score(pooly, model.predict(poolx), pos_label=1, average='binary'))
        prec, recall, fscore, support = precision_recall_fscore_support(testy, model.predict(testx), average='binary')
        
        E_out_f1 = np.append(E_out_f1, fscore)
        E_out_P = np.append(E_out_P, prec)
        E_out_R = np.append(E_out_R, recall)
        
       
        #correctness of unsupervised pool
        correct_elements = len([i for i, j in zip(oracle.y, pool._y_unsupervised) if i == j])
       
        correctness = float(correct_elements)/float(len(oracle.y))
        unsupervised_correctness = np.append(unsupervised_correctness, correctness)
        
          
        if x == quota-1:
            print("F1: % 2.3f " % E_out_f1[-1])
            print("Initial weights of queried pairs:", Counter(initial_weights_queries))
        

    print('Run', run)
    print('Runtime: % 2.3f seconds' % (time.time() - start_time))
    print('Results for last iteration:')
    print('Training F1 score: % 2.3f' % E_in_f1[-1])
    print('Test F1 score: % 2.3f' % E_out_f1[-1])
    print('Test Precision score: % 2.3f' % E_out_P[-1])
    print('Test Recall score: % 2.3f' % E_out_R[-1])
    print('Correctness of unsupervised Pool: % 2.3f' % unsupervised_correctness[-1])
    
    print("Labels")
    display(Counter(labels).keys())
    display(Counter(labels).values())
    
    return  E_in_f1, E_out_f1, unsupervised_correctness, model, pool_ids

