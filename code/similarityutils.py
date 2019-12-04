from __future__ import division
from similarity.levenshtein import Levenshtein
from similarity.jaccard import Jaccard
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import datetime
import re
from datetime import tzinfo
from dateutil.parser import parse
import pytz
from displayutils import *
from numpy import trapz
import re
from scipy.ndimage import gaussian_filter
from numpy import matlib
from copy import deepcopy

def static_threshold(similarities):
    threshold = 0.5
    for dp in range(0, len(similarities)):
        if similarities[dp] >= threshold:
            print 'Threshold defined with static method threshold (0.5): %f' % similarities[dp]
            return similarities[dp]

#code from https://dataplatform.cloud.ibm.com/analytics/notebooks/54d79c2a-f155-40ec-93ec-ed05b58afa39/view?access_token=6d8ec910cf2a1b3901c721fcb94638563cd646fe14400fecbb76cea6aaae2fb1 (accessed:12.09.2019)
def elbow_threshold(similarities, labels):
    sim_list = list(similarities)
    nPoints = len(sim_list)
    allCoord = np.vstack((range(nPoints), sim_list)).T
    
    firstPoint = allCoord[0]
    # get vector between first and last point - this is the line
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel    
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    idxOfBestPoint = np.argmax(distToLine)
    

    print "Knee of the curve is at index =",idxOfBestPoint
    print "Knee value =", similarities[idxOfBestPoint]
       
    return similarities[idxOfBestPoint],idxOfBestPoint
    
def auc_threshold(similarities):
    points = np.arange(1, len(similarities) + 1)
    plt.plot(points, similarities)
    plt.show()
    
    area = trapz(similarities, dx=len(similarities))
    half_area = float(area) / 2.0
    for dp in range(0, len(similarities)):
        area = trapz(similarities[:dp], dx=dp)
        if area >= half_area:
            print 'Threshold defined with AUC method: %f' % similarities[dp]
            return similarities[dp]        

def otsus_threshold(sorted_dataset):
    similarities = deepcopy(sorted_dataset)
    
    similarities.reverse()
    hist, _ = np.histogram(similarities, bins=len(similarities), range=(0.0, 1.0))
    hist = 1.0 * hist / np.sum(hist)
    val_max = -999
    thr = -1
    print_progress(1, len(similarities) - 1, prefix="Find Otsu's threshold:", suffix='Complete')
    for t in range(1, len(similarities) - 1):
        print_progress(t + 1, len(similarities) - 1, prefix="Find Otsu's threshold:", suffix='Complete')
        q1 = np.sum(hist[:t])
        q2 = np.sum(hist[t:])
        if q1 != 0 and q2 != 0:
            m1 = np.sum(np.array([ i for i in range(t) ]) * hist[:t]) / q1
            m2 = np.sum(np.array([ i for i in range(t, len(similarities)) ]) * hist[t:]) / q2
            val = q1 * (1 - q1) * np.power(m1 - m2, 2)
            if val_max < val:
                val_max = val
                thr = similarities[t]

    print "Threshold defined with Otsu's method: %f " % thr
    return thr

def valley_threshold(sorted_dataset):
    similarities = deepcopy(sorted_dataset)
   
    similarities.reverse()
    hist, _ = np.histogram(similarities, bins=len(similarities), range=(0.0, 1.0))
    hist = 1.0 * hist / np.sum(hist)
    val_max = -999
    thr = -1
    print_progress(1, len(similarities) - 1, prefix="Find Valley threshold:", suffix='Complete')
    float_list = [round(elem, 2) for elem in similarities]
    #normalizes by occurrences of most frequent value
    fre_occur = float_list.count(max(float_list,key=float_list.count))
    for t in range(1, len(similarities) - 1):
        print_progress(t + 1, len(similarities) - 1, prefix="Find Valley threshold:", suffix='Complete')
        q1 = np.sum(hist[:t])
        q2 = np.sum(hist[t:])
        if q1 != 0 and q2 != 0:
            m1 = np.sum(np.array([ i for i in range(t) ]) * hist[:t]) / q1
            m2 = np.sum(np.array([ i for i in range(t, len(similarities)) ]) * hist[t:]) / q2

            val = (1.0-float(float_list.count(round(similarities[t],2)))/float(fre_occur))*(q1 * (1.0 - q1) * np.power(m1 - m2, 2))
            if val_max < val:
                val_max = val
                thr = similarities[t]
    
    
    print "Threshold defined with valley threshold method: %f " % thr
    return thr
    
def get_date_type_(date_str):
    try:
        date_ = parse(date_str, fuzzy=True, default=datetime(1, 1, 1, 1, 1, tzinfo=tzoffset(None, 18000)))
        return date_
    except:
        import pdb
        pdb.set_trace()
        display('Could not parse %s' % date_str)
        return


def get_date_type(date_str):
    separator = ''
    if '.' in date_str:
        separator = '.'
    elif '\\' in date_str:
        separator = '\\'
    elif '/' in date_str:
        separator = '/'
    elif '-' in date_str:
        separator = '-'
    else:
        return None
    try:
        date_parts = [ d.strip() for d in date_str.split(separator) ]
        if re.match('\\d{4}[-\\.\\\\]\\d{1,2}[-\\.\\\\]\\d{1,2}', date_str):
            return datetime.datetime.strptime(date_str, '%Y' + separator + '%m' + separator + '%d').date()
        if re.match('\\d{1,2}[-\\.\\\\]\\d{1,2}[-\\.\\\\]\\d{4}', date_str):
            return datetime.datetime.strptime(date_str, '%d' + separator + '%m' + separator + '%Y').date()
        if re.match('\\d{2}[-\\.\\\\]\\d{1,2}[-\\.\\\\]\\d{1,2}', date_str):
            p = re.compile('\\d+')
            splitted_date = p.findall(date_str)
            if int(splitted_date[0]) < 32 and int(splitted_date[1]) < 13:
                return datetime.datetime.strptime(date_str, '%d' + separator + '%m' + separator + '%y').date()
            if int(splitted_date[0]) > 32:
                return datetime.datetime.strptime(date_str, '%y' + separator + '%m' + separator + '%d').date()
            try:
                return datetime.datetime.strptime(date_str, '%d' + separator + '%m' + separator + '%y').date()
            except:
                try:
                    return datetime.datetime.strptime(date_str, '%y' + separator + '%m' + separator + '%d').date()
                except:
                    display('Unknown pattern or invalid date: %s' % date_str)
                    return None

        else:
            return parse(date_str, fuzzy=True)
    except:
        f = open('unparseddates.txt', 'a')
        f.write(date_str + '\n')
        f.close()
        return None


def get_day_diff(date1, date2):
    if date1 == 'nan' or date2 == 'nan':
        return -1.0
    date1_ = get_date_type(date1)
    date2_ = get_date_type(date2)
    if date1_ == None or date2_ == None:
        return -1.0
    delta = date1_.day - date2_.day
    return abs(delta)


def get_month_diff(date1, date2):
    if date1 == 'nan' or date2 == 'nan':
        return -1.0
    date1_ = get_date_type(date1)
    date2_ = get_date_type(date2)
    if date1_ == None or date2_ == None:
        return -1.0
    delta = date1_.month - date2_.month
    return abs(delta)


def get_year_diff(date1, date2):
    if date1 == 'nan' or date2 == 'nan':
        return -1.0
    date1_ = get_date_type(date1)
    date2_ = get_date_type(date2)
    if date1_ == None or date2_ == None:
        return -1.0
    difference = abs(date1_.year - date2_.year)
    if len(date1) != len(date2) and difference % 100 == 0:
        difference = 0
    return difference


def get_num_equal(num1, num2):
    if num1 == 'nan' or num2 == 'nan':
        return -1.0
    try:
        num1_ = float(num1)
        num2_ = float(num2)
        if num1_ == num2_:
            return 1.0
        return 0.0
    except:
        return -1


def get_abs_diff(num1, num2):
    if num1 == 'nan' or num2 == 'nan':
        return -1.0
    try:
        num1_ = float(num1)
        num2_ = float(num2)
        return abs(num1_ - num2_)
    except:
        return -1


def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    if str1 == 'nan' or str2 == 'nan':
        return -1.0
    else:
        return float(len(c)) / float(len(a) + len(b) - len(c))


def get_relaxed_jaccard_sim(str1, str2):
    if str1 == 'nan' or str2 == 'nan':
        return -1.0
    a = set(str1.split())
    b = set(str2.split())
    c = []
    for a_ in a:
        for b_ in b:
            if get_levenshtein_sim(a_, b_) > 0.7:
                c.append(a_)

    intersection = len(c)
    min_length = min(len(a), len(b))
    if intersection > min_length:
        intersection = min_length
    return float(intersection) / float(len(a) + len(b) - intersection)


def get_containment_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    if str1 == 'nan' or str2 == 'nan':
        return -1.0
    elif len(a) == 0 or len(b) == 0:
        return -1.0
    else:
        return float(len(c)) / float(min(len(a), len(b)))


def get_levenshtein_sim(str1, str2):
    levenshtein = Levenshtein()
    if str1 == 'nan' or str2 == 'nan':
        return -1.0
    else:
        max_length = max(len(str1), len(str2))
        return 1.0 - levenshtein.distance(str1, str2) / max_length


def get_missing(str1, str2):
    if str1 == 'nan' or str2 == 'nan':
        return 1.0
    else:
        return 0.0


def get_overlap_sim(str1, str2):
    if str1 == 'nan' or str2 == 'nan':
        return -1.0
    elif str1 == str2:
        return 1.0
    else:
        return 0.0


def get_cosine_word2vec(str1, str2, model):
    if str1 == 'nan' or str2 == 'nan':
        return -1.0
    elif str1.replace(' ', '') in model.vocab and str2.replace(' ', '') in model.vocab:
        return model.similarity(str1.replace(' ', ''), str2.replace(' ', ''))
    else:
        return 0.0


def get_cosine_tfidf(tfidf_scores_ids, sourceID, targetID):
    source_index = np.where(tfidf_scores_ids['ids'] == sourceID)
    target_index = np.where(tfidf_scores_ids['ids'] == targetID)
    score = cosine_similarity(tfidf_scores_ids['scores'][source_index].todense(), tfidf_scores_ids['scores'][target_index].todense())
    return score[0][0]
    

def calculateTFIDF(records):   
    records_data = records['data']
    concat_records = []
    for row in records_data:
        if (isinstance(row,np.ndarray)): # tfidf based on  more that one features
            concat_row = ''
            for value in row:
                if not pd.isnull(value):
                    if type(value) is str:
                        if value.lower() != 'nan':
                            value = re.sub('[^A-Za-z0-9\s\t\n]+', '', str(value))
                            concat_row += ' ' + value
                    else: # tfidf based on one feature
                        value = re.sub('[^A-Za-z0-9\s\t\n]+', '', str(value))
                        concat_row += ' ' + str(value)

            concat_records.append(concat_row)
        else: 
            if pd.isnull(row):
                concat_records.append("")
            else:
                value = re.sub('[^A-Za-z0-9\s\t\n]+', '', str(row.lower()))
                concat_records.append(value)

            

    tf_idfscores = TfidfVectorizer(encoding='latin-1').fit_transform(concat_records)
    tf_idf = dict()
    tf_idf['ids'] = records['ids']
    tf_idf['scores'] = tf_idfscores
    
    return tf_idf
