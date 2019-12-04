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
import libact_datasetext as ext
import learningalgos as la
import time
from sklearn.metrics import confusion_matrix
import collections
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
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


#rf for random forest
#lr for logistic regression

def active_learning(al, query_strategy, max_quota, num_runs, model, warm_start=False):
    
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
    
    
    for i in range(num_runs):
       
        oracle = IdealLabeler(ext.ExtDataset(X, y, ids))
        
        pool_labels = np.asarray([None]*len(al['pool_labels']))

        pool_labels[al['bootstrapping_indices']] = al['bootstrapping_labels']

        pool =  ext.ExtDataset(X, pool_labels, ids)
        
        initial_seed_runs = 0
        while (1 not in pool._y) or (0 not in pool._y):
            
            qs_random = ext.RandomSampling_(pool)
            random_id = qs_random.make_query()
            X, _ = zip(*pool.data)
            lb = oracle.label(X[random_id])
            pool.update(random_id, lb)
            initial_seed_runs +=1
        
        print('Resample until the oracle has at least one positive and one negative label: %d times' %initial_seed_runs)
        if (max_quota == -1): max_quota = pool_labels.size - al['bootstrapping_labels'].size
        if model == "rf": 
            if warm_start:
                model_type = la.RandomForest_(warm_start=True, n_estimators =10)
            else: model_type = la.RandomForest_(n_estimators =10)
        elif model == "logr": model_type = la.LogisticRegression_()
        elif model == "dt": model_type = la.DecisionTree_()
        elif model == "svm": model_type = la.SVC_( gamma='auto',kernel='linear')
        elif model == "xgboost": model_type = la.XGBClassifier_(objective="binary:logistic")
        elif model == "linr": model_type = la.LinearRegression_()           
        else : print "Unknown model type."
       
        if query_strategy == 'uncertainty':
            qs = UncertaintySampling(pool, method='lc', model=la.RandomForest_())
        elif query_strategy == 'random':
            qs = RandomSamplingNoSeed(pool)
        elif query_strategy == 'dwus':
            qs = DWUS(pool)
        elif query_strategy == 'quire':
            qs = QUIRE(pool)
        elif query_strategy == 'variance':
            qs = VarianceReduction(pool, model=la.LogisticRegression_())
        elif query_strategy == 'dwus_dynamic':
            qs = DWUS(pool)
        elif query_strategy == 'default_committee':
            qs = QueryByCommittee(pool, # Dataset object
                models=[
                    la.RandomForest_(),
                    la.DecisionTree_(),
                    la.SVC_(gamma='auto',probability=True),
                    la.LogisticRegression_(),
                    la.XGBClassifier_(objective="binary:logistic")
                ],
            )
        elif query_strategy == 'committee':
            committee = CommitteeModels(model)
            
            qs = QueryByCommittee(pool, # Dataset object
                models= committee.committee['models'],
            )

        train_acc, train_f1, test_acc, test_f1, model_, pool_ids = run (pool, evaluation_ds, oracle, model_type, qs, max_quota, i, al, query_strategy, initial_seed_runs, warm_start)

        training_accuracy_scores.append(train_acc)
        training_f1_scores.append(train_f1)
        test_accuracy_scores.append(test_acc)
        test_f1_scores.append(test_f1)
        
        if (i == num_runs-1):
            print"Feature importances for the last AL run"
            showFeatureImportances(al['pool_data'].columns.values.tolist(), model_, model)
        
    
    plotResultsMultipleRuns(max_quota,training_f1_scores, test_f1_scores, ('Experiment Result: '+model+' w '+query_strategy+' sampling'))
    
    return test_f1_scores
 
    
def run(pool, test, oracle, model, qs, quota, run, al,qs_name, initial_seed_runs =0, warm_start=False):
    
    start_time = time.time()
    pool_ids = []
    
    E_in, E_in_f1, E_out, E_out_f1, E_out_P, E_out_R = [], [], [], [], [], []
        
    labels = []
    
    print_progress(0, quota, prefix = 'AL RUN: '+str(run), suffix = 'Complete')
    for x in range(initial_seed_runs):
        E_out_f1 = np.append(E_out_f1, 0.0)
        E_out_P = np.append(E_out_P, 0.0)
        E_out_R = np.append(E_out_R, 0.0)
        E_in_f1 = np.append(E_in_f1, 0.0)
        E_in = np.append(E_in, 0.0)
    
    for x in range(quota-initial_seed_runs):        
                
        print_progress(x + 1, quota, prefix = 'AL RUN: '+str(run), suffix = 'Complete')

        # Standard usage of libact objects
        ask_id = qs.make_query()
        X, _ = zip(*pool.data)
        lb = oracle.label(X[ask_id])
        pool.update(ask_id, lb)

        labels.append(lb)        
        
        if warm_start: model.model.n_estimators +=2
        model.train(pool)
        poolx, pooly = pool.format_sklearn()
        testx, testy = test.format_sklearn()
        
        E_in = np.append(E_in, model.score(pool))
        E_in_f1 = np.append(E_in_f1, f1_score(pooly, model.predict(poolx), pos_label=1, average='binary', sample_weight=None))
      
        E_out = np.append(E_out, model.score(test))
        prec, recall, fscore, support = precision_recall_fscore_support(testy, model.predict(testx), average='binary')
        
        E_out_f1 = np.append(E_out_f1, fscore)
        E_out_P = np.append(E_out_P, prec)
        E_out_R = np.append(E_out_R, recall)
        
        
       
    print'Run', run
    print'Runtime: % 2.3f seconds' % (time.time() - start_time)
    print'Results for last iteration:'
    print'Training accuracy score: % 2.3f' % E_in[-1]
    print'Training F1 score: % 2.3f' % E_in_f1[-1]
    print'Test accuracy score: % 2.3f' % E_out[-1]
    print'Test F1 score: % 2.3f' % E_out_f1[-1]
    print'Test Precision score: % 2.3f' % E_out_P[-1]
    print'Test Recall score: % 2.3f' % E_out_R[-1]

    print"Labels"
    display(Counter(labels).keys())
    display(Counter(labels).values())
    
    
    return E_in, E_in_f1, E_out, E_out_f1, model, pool_ids

def batchTraining(X,y, model_type, optimization=False):
    clf = getClassifier(model_type, optimization)
    scoring = ['precision', 'recall', 'f1']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_validate(clf, X, y, scoring=scoring, cv=cv, return_train_score=False)
    display(scores)
    print"F1 mean:", "% 2.3f" % np.mean(scores['test_f1'])," Standard Deviation:", "% 2.3f" % np.std(scores['test_f1'])

    
def getClassifier(model_type, optimization=False):
    if optimization:
        f1_scorer = make_scorer(f1_score)

        if model_type == "rf":
            clf=RandomForestClassifier(random_state=1)
            grid_values = {'n_estimators' : [10, 50, 100], 'max_depth' : [3, 5, 10, 15], 'min_samples_leaf' : [3, 5, 10]}
            model = GridSearchCV( clf, cv=5,param_grid = grid_values,scoring = f1_scorer)
        elif model_type == "logr":
            clf=LogisticRegression(random_state=1)
            grid_values = {'penalty' : ['l1', 'l2'], 'fit_intercept' : [True, False], 'solver' : ['liblinear', 'saga'], 'max_iter' : [50, 100, 150]}
            model = GridSearchCV(clf, cv=5,param_grid = grid_values,scoring = f1_scorer)
        elif model_type == "dt":
            clf=DecisionTreeClassifier(random_state=1)
            grid_values = {'max_depth' : [None, 3, 5, 10, 15], 'min_samples_leaf' : [3, 5, 10], 'criterion' : ['gini', 'entropy']}
            model = GridSearchCV(clf, cv=5,param_grid = grid_values,scoring = f1_scorer) 
        elif model_type == "svm":
            clf=SVC(random_state=1)
            grid_values = {'kernel' : ['linear', 'rbf'], 'gamma' : [0.1, 1, 5, 10], 'C' : [0.5, 1, 5, 10]}
            model = GridSearchCV(clf, cv=5,param_grid = grid_values,scoring = f1_scorer)
        elif model_type == "xgboost":
            clf=xgb.XGBClassifier(random_state=1)
            grid_values = {'n_estimators' : [100, 150, 200], 'learning_rate' : [0.05, 0.1, 0.15], 'max_depth' : [3, 5, 7]}
            model = GridSearchCV(clf, cv=5,param_grid = grid_values,scoring = f1_scorer)
    else:      
        if model_type == "rf": model = RandomForestClassifier(random_state=1)
        elif model_type == "logr": model = sklearn.linear_model.LogisticRegression(random_state=1)
        elif model_type == "dt": model = DecisionTreeClassifier(random_state=1)
        elif model_type == "svm": model= SVC(random_state=1, gamma='auto', kernel='linear')
        elif model_type == "linr": model = LinearRegression() 
        elif model_type == "xgboost": model = xgb.XGBClassifier(objective="binary:logistic", random_state=1)
        else : print "Unknown model type."
   
    return model

def batchTraining(X,y, X_val, y_val, model_type,optimization=False, printResults = True, showMisclassifications = False, ids = None,  interestingness_model= None, returnModel=False):
    display("Training size:", X.shape[0])
    display("Validation size:", X_val.shape[0])
   
    
    if (interestingness_model is not None):
        predict_prob = interestingness_model.predict_proba(X)
        training_labels = interestingness_model.predict(X)
        correct_prob = []
        for p in zip(predict_prob, training_labels):
            correct_prob.append(p[0][p[1]])
        interest_level = 1.0 - np.mean(correct_prob)
    
    clf = getClassifier(model_type, optimization)
    model = clf.fit(X,y)
    predictions = clf.predict(X_val)
    prec, recall, fscore, support  = precision_recall_fscore_support(y_val, predictions, average='binary')
    
    if (showMisclassifications):
        falsePositives = []
        falseNegatives = []
        for i in range(0, len(predictions)-1):
            if predictions[i] == 1 and y_val[i] == 0:
                falsePositives.append(i)
            if predictions[i] == 0 and y_val[i] == 1:
                falseNegatives.append(i)
        
    if printResults:
        display("Precision:", prec)
        display("Recall:", recall)
        display("F1:", fscore)
        if optimization:
            showFeatureImportances(X.columns.values.tolist(), model.best_estimator_, model_type)
        else: showFeatureImportances(X.columns.values.tolist(), model, model_type)
        print(classification_report(y_val, predictions))
    if model_type == "dt":
        tree = printTreeRules(X.columns.values.tolist(), model.best_estimator_)
    if (showMisclassifications):
        #if (ids == None): print("You need to pass the ids column to show the misclassified elements.")
        print("False positives ( %i ) ids:" % len(falsePositives))
        display(ids.iloc[falsePositives])

        print("False negatives ( %i ) ids:" % len(falseNegatives))
        display(ids.iloc[falseNegatives])
    
    if (interestingness_model is not None): return (prec, recall, fscore, support, interest_level)
    elif (returnModel): return  (prec, recall, fscore, support, model)
    elif (model_type=="dt"): return (prec, recall, fscore, support, tree)
    else: return  (prec, recall, fscore, support)

def printTreeRules(feature_names, tree):
    tree_model = []
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    #tree_model =  tree_model +"def tree({}):".format(", ".join(feature_names))
    
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            tree_model.append("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            tree_model.append("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            tree_model.append("{}return {}".format(indent, tree_.value[node]))
    
    recurse(0, 1)
    return ''.join(tree_model)
    
#for the setting n_to_one we are allowed to add all the negative links for every positive link that we label
def additionalPoolKnowledgeNto1(pool, ask_id, oracle, X, labels):
    idlabel = pool.dataWIDs[ask_id]
    sourceid = idlabel[2].split('-')[0]
    additionalKnowledge_askids = []
    additionalKnowledge_ids = []
    index = 0
    for record_id in pool.dataWIDs:
        if record_id[2].split('-')[0] == sourceid:
            if record_id[2] != idlabel[2]:
                additionalKnowledge_ids.append(record_id[2])
                additionalKnowledge_askids.append(index)
                lb = oracle.label(X[index])
                if lb == 1.0 : 
                    display("Label from additional knowledge N-->1 setting cannot be 1.")
                    #sys.exit(0)
                pool.update(index, lb)
                labels.append(lb)
        index += 1

# TODO use cross validation instead here?
def findBestModel(models, pool, pool_weights = []):
    bestF1 = 0.0
    bestModel = None
    
    poolx, pooly = pool.format_sklearn()
    for model in models:
        f1 = f1_score(pooly, model.predict(poolx))
        if f1 >= bestF1: 
            bestF1 = f1
            bestModel = model
    return bestModel
