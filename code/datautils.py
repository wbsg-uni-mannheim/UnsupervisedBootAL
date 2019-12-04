from sklearn.datasets import dump_svmlight_file
import pandas as pd
import csv
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from similarityutils import *
from gensim.models import Word2Vec, KeyedVectors
from displayutils import *
import re
import sys
from dateutil.parser import parse
from datetime import datetime, tzinfo 
import pytz
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 


def getLabelledDataFromFile(fileName, rescale=False, showPlots=False):
    data = pd.read_csv(fileName, ',')

    if rescale:
        print("Rescale values")
        data.replace(-1, np.nan, inplace=True)
        for c in data.columns.drop(['source_id','target_id','pair_id','label']):
            data[c] -= data[c].min()
            data[c] /= data[c].max()
            if ('diff' in c):
                #reverse
                data[c] = 1- data[c]
                data.rename(columns={c: c.replace("diff", "sim")}, inplace=True)
        data.replace(np.nan,-1, inplace=True)
    
    #set features
    data_feature_names = list(set(data.columns.values) - {'label','source_id','target_id','pair_id',''})
    # remove the type mproperty if existing
    data_feature_names = [x for x in data_feature_names if not x.startswith('<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>')]
    data_feature_values = data[data_feature_names]
    
    print"Nan values in input labelled data: ", data.isnull().sum().sum()
    print"Replace all Nan values with -1"
    data = data.replace(np.NaN,-1)   
    
    #set label
    data_labels = data['label']
    #encode the labels
    label = LabelEncoder()
    data_labels = label.fit_transform(data_labels)
    label_names=[True, False]
    label_order=label.transform(label_names)
    
    if showPlots:
        sns.countplot(data_labels, label="count")
        plt.show()
    labeled_data = dict()
    labeled_data['feature_values'] = data_feature_values
    labeled_data['feature_names'] = data_feature_names
    labeled_data['labels'] = data_labels
    labeled_data['ids'] = data['pair_id']
    
    return labeled_data

def readData(directory):
    source = pd.read_csv('%s/source_.csv' % directory, sep='\|\|')
    print"Source file records:",len(source)
    target = pd.read_csv('%s/target_.csv'  % directory, sep='\|\|')
    print"Target file records:",len(target)
    pool = pd.read_csv('%s/pool.csv'  % directory, sep=';')
    print"Correspondences in the pool:",len(pool)
    validation = pd.read_csv('%s/validation.csv'  % directory, sep=';')
    print"Correspondences in the validation set:",len(validation)
    data = dict()
    data['source'] = source
    data['target'] = target
    data['pool'] = pool 
    data['validation'] = validation
    return data

def getTypesofData(data):
    dict_types = dict()
    #return dictionary
    for column in data:
        column_values = data[column].dropna()
        type_list=list(set(column_values.map(type).tolist()))

        if len(type_list) == 0: 
            "No type could be detected. Default (string) will be assigned."
            dict_types[column] = 'str'
        elif len(type_list) >1: 
            "More than one types could be detected. Default (string) will be assigned."
            dict_types[column] = 'str'
        else:            
            if str in type_list:   
                types_of_column = []
                length = 0 
                for value in column_values:
                    length = length + len(value.split())
                    if re.match(r'.?\d{2,4}[-\.\\]\d{2}[-\.\\]\d{2,4}.?', value):
                        types_of_column.append('date')
                avg_length = length/len(column_values)
                
                if (avg_length>6): types_of_column.append('long_str')

                if len(set(types_of_column)) == 1:                  
                    if ('date' in types_of_column): 
                        dict_types[column] = 'date'
                    elif ('long_str' in types_of_column):
                        dict_types[column] = 'long_str'
                    else : dict_types[column] = 'str'
                else: 
                    "More than one types could be detected. Default (string) will be assigned."
                    dict_types[column] = 'str'
            else: # else it must be numeric
                dict_types[column] = 'numeric'
    return dict_types

def is_date(string, fuzzy=True):
    try: 
        parse(string, fuzzy=fuzzy, default= datetime(1, 1, 1, tzinfo=pytz.UTC))        
        return True

    except ValueError:
        return False

def createFeatureVectorFile(source,target,pool,featureFile, keyfeature='subject_id', embeddings = True, predefinedTypes = dict()):    
    source_headers = source.columns.values
    target_headers = target.columns.values

    print("Get types of data")
    dict_types_source = getTypesofData(source)
    display(dict_types_source)
    dict_types_target = getTypesofData(target)
    display(dict_types_target)
    
    common_elements = list(set(source_headers) & set(target_headers) - {keyfeature})
    common_elements_types = dict()
    for common_element in common_elements:
        if (dict_types_source[common_element] is dict_types_target[common_element]):
            common_elements_types[common_element] = dict_types_source[common_element]
        else:
            if (dict_types_source[common_element]=='long_str' or dict_types_target[common_element]=='long_str'):
                print("Different data types in source and target for element %s. Will assign long string" % common_element)
                common_elements_types[common_element] = 'long_str'
            else: 
                print("Different data types in source and target for element %s. Will assign string" % common_element)
                common_elements_types[common_element] = 'str'
             
    
    #calculate tfidf vectors
    print "Calculate tfidf scores"   
    records = dict()
    records["data"] = np.concatenate((source[common_elements].values, target[common_elements].values), axis=0)
    records["ids"] = np.concatenate((source[keyfeature], target[keyfeature]), axis=0)


    tfidfvector_ids = calculateTFIDF(records)
    
    print"Create similarity based features from",len(common_elements),"common elements"
    
    similarity_metrics={
        'str':['lev', 'jaccard', 'relaxed_jaccard', 'overlap', 'cosine', 'containment'],
        'numeric':['abs_diff', 'num_equal'],
        'date':['day_diff', 'month_diff','year_diff'],
        'long_str':['cosine','lev', 'jaccard', 'relaxed_jaccard', 'overlap', 'cosine_tfidf', 'containment']
    }
    
    if not embeddings:
        similarity_metrics['str'].remove('cosine')
        similarity_metrics['long_str'].remove('cosine')

    
    features = []
   
    #fix headers
    header_row = []
    header_row.append('source_id')
    header_row.append('target_id')
    header_row.append('pair_id')
    header_row.append('label')
    header_row.append('cosine_tfidf')
    for f in common_elements:
        for sim_metric in similarity_metrics[common_elements_types[f]]:
            header_row.append(f+"_"+sim_metric)       
            
    features.append(header_row)
    word2vec=None
    if embeddings :
        print"Load pre-trained word2vec embeddings"
        filename = '../../GoogleNews-vectors-negative300.bin'
        word2vec = KeyedVectors.load_word2vec_format(filename, binary=True)
        print"Pre-trained embeddings loaded"
    

    tfidfvector_perlongfeature = dict()
    if 'long_str' in common_elements_types.values():
        for feature in common_elements_types:
            if common_elements_types[feature] == 'long_str':             
                records_feature = dict()
                records_feature["data"] = np.concatenate((source[feature].values, target[feature].values), axis=0)
                records_feature["ids"] = np.concatenate((source[keyfeature], target[keyfeature]), axis=0)
                tfidfvector_feature = calculateTFIDF(records_feature)
                tfidfvector_perlongfeature[feature] = tfidfvector_feature
    
    print_progress(0, len(pool), prefix = 'Create Features:', suffix = 'Complete')
    ps = PorterStemmer()
    for i in range(len(pool)):
        print_progress(i + 1, len(pool), prefix = 'Create Features:', suffix = 'Complete')
        features_row = []
        #metadata
        r_source_id = pool['source_id'].loc[i]
        r_target_id = pool['target_id'].loc[i]
        
        features_row.append(r_source_id)
        features_row.append(r_target_id)
        features_row.append(r_source_id+"-"+r_target_id)
        features_row.append(pool['matching'].loc[i])

        features_row.append(get_cosine_tfidf(tfidfvector_ids, r_source_id, r_target_id))
        
        for f in common_elements:
            fvalue_source = str(source.loc[source[keyfeature] == r_source_id][f].values[0])
            fvalue_target = str(target.loc[target[keyfeature] == r_target_id][f].values[0])
             
            if common_elements_types[f] is 'str' or common_elements_types[f] is 'long_str' :
                fvalue_source = re.sub('[^A-Za-z0-9]+', ' ', str(fvalue_source.lower())).strip()
                fvalue_target = re.sub('[^A-Za-z0-9]+', ' ', str(fvalue_target.lower())).strip()
            ## if long str remove stopwords and stem
            if common_elements_types[f] is 'long_str':
                cachedStopWords = stopwords.words("english")
                fvalue_source = ' '.join([word for word in fvalue_source.split() if word not in cachedStopWords])
                fvalue_target = ' '.join([word for word in fvalue_target.split() if word not in cachedStopWords])
                #stem
                fvalue_source = ' '.join([ps.stem(word) for word in fvalue_source.split()])
                fvalue_target = ' '.join([ps.stem(word) for word in fvalue_target.split()])
            
            if f in tfidfvector_perlongfeature:            
                typeSpecificSimilarities(common_elements_types[f], fvalue_source, fvalue_target, similarity_metrics, features_row, word2vec, tfidfvector_perlongfeature[f], r_source_id, r_target_id)
            else: 
                typeSpecificSimilarities(common_elements_types[f], fvalue_source, fvalue_target, similarity_metrics, features_row,word2vec)
            
        features.append(features_row)

    print('Created', len(features[0]), 'features for',len(features),'entity pairs')
    
    
    with open(featureFile, mode='w') as feature_file:
        writer = csv.writer(feature_file)
        writer.writerows(features)

    print"Feature file created"

def typeSpecificSimilarities(data_type, valuea, valueb, type_sim_map, features_row, word2vec, tfidfvector=None, r_source_id=None, r_target_id=None):
    #similarity-based features
    values_sim = []
    for sim_metric in type_sim_map[data_type]:
        if valuea == 'nan' or valueb == 'nan' or valuea == '' or valueb == '':
            features_row.append(-1.0)
            values_sim.append(-1)
        elif sim_metric=='lev':
            features_row.append(get_levenshtein_sim(valuea,valueb))
            values_sim.append(get_levenshtein_sim(valuea,valueb))
        elif sim_metric=='jaccard':
            features_row.append(get_jaccard_sim(valuea,valueb))
            values_sim.append(get_jaccard_sim(valuea,valueb))
        elif sim_metric=='relaxed_jaccard':
            features_row.append(get_relaxed_jaccard_sim(valuea,valueb))
            values_sim.append(get_relaxed_jaccard_sim(valuea,valueb))
        elif sim_metric=='overlap':
            features_row.append(get_overlap_sim(valuea,valueb))
            values_sim.append(get_overlap_sim(valuea,valueb))
        elif sim_metric=='containment':
            features_row.append(get_containment_sim(valuea,valueb))
            values_sim.append(get_containment_sim(valuea,valueb))
        elif sim_metric=='cosine':
            features_row.append(get_cosine_word2vec(valuea,valueb,word2vec))
            values_sim.append(get_cosine_word2vec(valuea,valueb,word2vec))
        elif sim_metric=='cosine_tfidf':
            features_row.append(get_cosine_tfidf(tfidfvector, r_source_id, r_target_id))
            values_sim.append(get_cosine_tfidf(tfidfvector, r_source_id, r_target_id))
        elif sim_metric=='abs_diff':
            features_row.append(get_abs_diff(valuea,valueb))
            values_sim.append(get_abs_diff(valuea,valueb))
        elif sim_metric=='num_equal':
            features_row.append(get_num_equal(valuea,valueb))
            values_sim.append(get_num_equal(valuea,valueb))
        elif sim_metric=='day_diff':
            features_row.append(get_day_diff(valuea,valueb))  
            values_sim.append(get_day_diff(valuea,valueb))
        elif sim_metric=='month_diff':
            features_row.append(get_month_diff(valuea,valueb))
            values_sim.append(get_month_diff(valuea,valueb))
        elif sim_metric=='year_diff':
            features_row.append(get_year_diff(valuea,valueb))
            values_sim.append(get_year_diff(valuea,valueb))      
        else: print("Unknown similarity metric %s" % sim_metric)
        if (-1 in values_sim and len(set(values_sim))>1):
            import pdb; pdb.set_trace();
    
def writeDataAsLibSVM(X,y,fileName, query_ids_):
    dump_svmlight_file(X,y,fileName,zero_based=True,multilabel=False, query_id=query_ids_)
    
def duplicatesinTrainingData(c,y):
    alreadyAdded = False
    dupl_c = dict()
    sorted_ind_c = sorted(range(len(c)), key=lambda x: c[x]) # sort incoming list but save the indexes of sorted items
 
    for i in xrange(len(c) - 1): # loop over indexes of sorted items
        if c[sorted_ind_c[i]] == c[sorted_ind_c[i+1]]: # if two consecutive indexes point to the same value, add it to the duplicates
            if not alreadyAdded:
                dupl_c[ c[sorted_ind_c[i]] ] = [sorted_ind_c[i], sorted_ind_c[i+1]]
                alreadyAdded = True
            else:
                dupl_c[ c[sorted_ind_c[i]] ].append( sorted_ind_c[i+1] )
        else:
            alreadyAdded = False
    return dupl_c

