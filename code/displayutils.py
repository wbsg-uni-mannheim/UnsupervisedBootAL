# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from statsmodels import robust
import sys
import libact


def showConfusionMatrix(labels, data, model):
    cm = confusion_matrix(labels, model.predict(data))
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['negative', 'positive']); ax.yaxis.set_ticklabels(['negative', 'positive']);
    plt.show()

# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
    
def printbold(string):
    print(Markdown())

def printFeaturesAsList(features):
    for c in sorted(features):
        print c

def get_model_importances(model,classifierName=None):
    if isinstance(model, libact.base.interfaces.ProbabilisticModel): 
        if model.name=='svm':
            importances= model.model.coef_.ravel()
        else: importances = model.feature_importances_()
    else: 
        if classifierName == 'logr':
            importances = model.coef_.ravel()
        elif classifierName == 'svm':
            if model.kernel != 'linear':
                display("Cannot print feature importances without a linear kernel")
                return
            else: importances = model.coef_.ravel()
        else:
            importances = model.feature_importances_
    
    return importances

def showFeatureImportances(column_names, model, classifierName):
      
    importances = get_model_importances(model, classifierName)
       
    column_names = [c.replace('<http://schema.org/Product/', '').replace('>','') for c in column_names]
    sorted_zipped = sorted(list(zip(column_names, importances)), key = lambda x: x[1], reverse=True)[:50]
    #for feature in sorted(zipped, key = lambda x: x[1]):
        #print(feature)
    plt.figure(figsize=(18,3))
    plt.title('Feature importances for classifier %s (max. top 50 features)' % classifierName)
    plt.bar(range(len(sorted_zipped)), [val[1] for val in sorted_zipped], align='center', width = 0.8)
    plt.xticks(range(len(sorted_zipped)), [val[0] for val in sorted_zipped])
    plt.xticks(rotation=90)
    plt.show() 
    
def plotResults(max_quota, train_acc_scores, test_acc_scores, test_f1_scores, title):
    query_num = np.arange(1, max_quota + 1)
    plt.plot(query_num, train_rf_uncertain, label='qs Ein Accuracy')
    plt.plot(query_num, test_rf_uncertain, label='qs Eout Accuracy')
    plt.plot(query_num, testf1_rf_uncertain, label='qs Eout F1')
    plt.xlabel('Number of Queries')
    plt.title(title)
    plt.legend(loc='upper center')
    plt.show()
    
def plotResultsMultipleRuns(max_quota, train_f1_scores, test_f1_scores, title):
    query_num = np.arange(1, max_quota + 1)
    
    # train f1
    np_train_f1_scores = np.array(train_f1_scores)
    mean_train_f1_scores = np.mean(np_train_f1_scores, axis=0)
    std_train_f1_scores = np.std(np_train_f1_scores, axis=0)
   
    # test_f1
    np_test_f1_scores = np.array(test_f1_scores)
    mean_test_f1_scores = np.mean(np_test_f1_scores, axis=0)
    std_test_f1_scores = np.std(np_test_f1_scores, axis=0)
    
    print"Final Results"
    print'Train F1 final iteration, mean % 2.3f' % mean_train_f1_scores[max_quota-1],  'σ % 2.3f'  % std_train_f1_scores[max_quota-1]

    print'Test F1 final iteration, mean % 2.3f' % mean_test_f1_scores[max_quota-1], 'σ % 2.3f'  % std_test_f1_scores[max_quota-1]
    
    #plot
    plt.errorbar(query_num, mean_train_f1_scores, yerr=std_train_f1_scores, label='Mean, std Train F1')
    plt.errorbar(query_num, mean_test_f1_scores, yerr=std_test_f1_scores, label='Mean, std Test F1')

    
    plt.xlabel('Number of Queries')
    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc