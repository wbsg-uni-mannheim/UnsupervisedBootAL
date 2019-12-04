import numpy as np
from sklearn.cluster import KMeans
from IPython.display import Markdown, display
import random
import pandas as pd
from collections import Counter
from bootstrapping import*
from similarityutils import*

class BootstrappingUnsupervised(Bootstrapping) :

    def __init__(self, sample_size=0, data=None, labels=None, ids=None, bootstrap_method=None, weighting_method=None, otsu_threshold=None, domain = None):
        self.sample_size = sample_size
        self.data = data
        self.labels = labels
        self.ids = ids
        self.bootstrap_method = bootstrap_method
        self.threshold = otsu_threshold
        self.threshold_index = 0
        self.domain = domain
        self.penalty_factor = 0.01

        if self.sample_size>len(self.data):
            print ("The sample size is bigger than the dataset. All examples will be chosen for bootstrapping.")
            self.sample_size = len(self.data)
            
        self.sample= dict()
                 
        if self.bootstrap_method == 'bowtopbottom': self.bootstrapBoW(selection = 'topbottom')
        elif self.bootstrap_method == 'bowotsu': self.bootstrapBoW(selection = 'otsu')
        elif self.bootstrap_method == 'attrotsu': self.bootstrapAttribute(selection = 'otsu')
        elif self.bootstrap_method == 'bowauc': self.bootstrapBoW(selection = 'auc')
        elif self.bootstrap_method == 'attrauc': self.bootstrapAttribute(selection = 'auc')
        elif self.bootstrap_method == 'attrauc_penalty': self.bootstrapAttribute(selection = 'auc', weight='penalty')
        elif self.bootstrap_method == 'attrauc_density': self.bootstrapAttribute(selection = 'auc', weight='density')
        elif self.bootstrap_method == 'bowelbow': self.bootstrapBoW(selection = 'elbow')
        elif self.bootstrap_method == 'attrelbow': self.bootstrapAttribute(selection = 'elbow')
        elif self.bootstrap_method == 'attrelbow_penalty': self.bootstrapAttribute(selection = 'elbow', weight='penalty')
        elif self.bootstrap_method == 'attrstatic_penalty': self.bootstrapAttribute(selection = 'static', weight='penalty')
        elif self.bootstrap_method == 'attrelbow_density': self.bootstrapAttribute(selection = 'elbow', weight='density')
        elif self.bootstrap_method == 'bowstatic': self.bootstrapBoW(selection = 'static')
        elif self.bootstrap_method == 'attrstatic': self.bootstrapAttribute(selection = 'static')
        elif self.bootstrap_method == 'attrstatic_density': self.bootstrapAttribute(selection = 'static', weight='density')
        elif self.bootstrap_method == 'attrtopbottom': self.bootstrapAttribute(selection = 'topbottom')
        elif self.bootstrap_method == 'attrtopbottom_density': self.bootstrapAttribute(selection = 'topbottom', weight='density')
        elif self.bootstrap_method == 'attrtopbottom_penalty': self.bootstrapAttribute(selection = 'topbottom', weight='penalty')
        elif self.bootstrap_method == 'attrotsu_density': self.bootstrapAttribute(selection = 'otsu', weight='density')
        elif self.bootstrap_method == 'attrotsu_penalty': self.bootstrapAttribute(selection = 'otsu', weight='penalty')
        elif self.bootstrap_method == 'attrvalley_penalty': self.bootstrapAttribute(selection = 'valley', weight='penalty')
        elif self.bootstrap_method == 'attrvalley_density': self.bootstrapAttribute(selection = 'valley', weight='density')


        elif self.bootstrap_method == 'handwritten_rule' : self.selectWRule()

        else : display("Unknown boostrapping method: "+self.bootstrap_method)
        self.printLabelDistr()
     
    def getStaticThresholdElementsFromSortedDataset(self, sorted_dataset):
        # the dataset is sorted by ascending order
        display("Define static threshold (0.5) and take positive and negative elements")
        sim_scores = [sim[2] for sim in sorted_dataset] 
        
        threshold = static_threshold(sim_scores)
        self.threshold = 0.5
        self.getThrElements(sorted_dataset, threshold, sim_scores)

    
    
    def getThrElements(self, sorted_dataset, threshold_value, sim_scores):
        print("Consider everything above the threshold as positive and below as negative.")
        
        threshold_index = sim_scores.index(threshold_value)
        self.threshold = threshold_value
        top_elements_all = sorted_dataset[:threshold_index]
                                                  
        bottom_elements_all = sorted_dataset[-(len(sorted_dataset)-threshold_index):]

        self.createBootstrappedSample(top_elements_all, bottom_elements_all)

    def getElbowThrElementsFromSortedDataset(self, sorted_dataset):
         # the dataset is sorted by ascending order
        sim_scores = [sim[2] for sim in sorted_dataset] 
        labels = [sim[0] for sim in sorted_dataset] 
        threshold, index = elbow_threshold(sim_scores, labels)
        
        display("Define Elbow threshold: %f " % threshold)
        self.threshold = threshold
        self.threshold_index = index
        self.getThrElements(sorted_dataset, threshold, sim_scores)
    
    def getAUCThrElementsFromSortedDataset(self, sorted_dataset):
        # the dataset is sorted by ascending order
        sim_scores = [sim[2] for sim in sorted_dataset] 
        
        threshold = auc_threshold(sim_scores)
        self.threshold = threshold
        display("Define AUC threshold: %f " % threshold)

        self.getThrElements(sorted_dataset, threshold, sim_scores)
        
        
    def getOtsuThrElementsFromSortedDataset(self, sorted_dataset):
        # the dataset is sorted by ascending order
        sim_scores = [sim[2] for sim in sorted_dataset] 
        if self.threshold is None:
            threshold = otsus_threshold(sim_scores)
        else : threshold = self.threshold
        
        display("Define Otsu's threshold: %f" % threshold)

        self.getThrElements(sorted_dataset, threshold, sim_scores)

       
    def getValleyThrElementsFromSortedDataset(self, sorted_dataset):
        # the dataset is sorted by ascending order
        sim_scores = [sim[2] for sim in sorted_dataset] 
        if self.threshold is None:
            threshold = valley_threshold(sim_scores)
        else : threshold = self.threshold
        self.threshold = threshold
        display("Define Valley threshold: %f" % threshold)

        self.getThrElements(sorted_dataset, threshold, sim_scores)
    

    def createBootstrappedSample(self, top_elements, bottom_elements):
        boot_data = []
        boot_labels = []
        boot_ids = []
        boot_indices = [] 
        boot_scores = []
        
        correct_samples = 0
        truepositives = 0
        falsepositives = 0
        falsenegatives = 0
        for boot_element in (top_elements):
            if (boot_element[0] == 0): 
                correct_samples = correct_samples+1
            else: falsepositives = falsepositives+1
            boot_labels.append(0)
            boot_ids.append(boot_element[1])
            boot_scores.append(boot_element[2])
            boot_indices.append(boot_element[3])
            
        for boot_element in (bottom_elements):
            if (boot_element[0] == 1): 
                correct_samples = correct_samples+1
                truepositives =truepositives+1
            else: falsenegatives = falsenegatives+1
            boot_labels.append(1)
            boot_ids.append(boot_element[1])
            boot_scores.append(boot_element[2])
            boot_indices.append(boot_element[3])


        self.sample['data'] = self.data.iloc[boot_indices]
        self.sample['labels'] = boot_labels
        self.sample['ids'] = boot_ids
        self.sample['indices'] = boot_indices 
        self.sample['scores'] = boot_scores
        if (len(top_elements)+len(bottom_elements)) == 0:
            self.sample['correctness'] = 0
            precision =0
            recall =0
            self.sample['f1'] = 0
        else:
            self.sample['correctness'] = float(correct_samples)/float(len(top_elements)+len(bottom_elements))
            precision = float(truepositives)/float(truepositives+falsepositives)
            recall = float(truepositives)/float(truepositives+falsenegatives)
            self.sample['f1'] = 2.0*precision*recall/(precision+recall)
        
        
    def getTopBottomElementsFromSortedDataset(self, sorted_dataset):
        # the dataset is sorted by ascending order
        display("Select top and bottom elements")
        if (self.sample_size==0):
            top_elements = sorted_dataset[:0]
            bottom_elements = sorted_dataset[:0]
        else:
            top_elements = sorted_dataset[:(self.sample_size/2)]
            bottom_elements = sorted_dataset[-(self.sample_size/2):]
        
        self.createBootstrappedSample(top_elements, bottom_elements)

   
    def selectWRule(self):

        print("Will select samples based on handwritten rule for the %s domain" % self.domain)

        helping_index = np.arange(self.ids.size)
        ruleBasedLabels = []

        boot_data = []
        boot_labels = []
        boot_ids = []
        boot_indices = []

        correct_samples = 0
        truepositives = 0
        falsepositives = 0
        falsenegatives = 0

        randomize_index = np.arange(self.ids.size)
        random.Random(1).shuffle(randomize_index)

        for p in randomize_index:
            label = 0
            pair_data = self.data.loc[p]

            if self.domain == 'author':
                if (pair_data['cosine_tfidf']>0.9 or pair_data['label_overlap']==1.0 or (pair_data['label_relaxed_jaccard'] >0.6 and pair_data['birthdate_year_sim'] == 1.0) or (pair_data['label_relaxed_jaccard'] >0.7 and pair_data['birthdate_year_sim'] > 0.8)):                   
                    label = 1
            elif self.domain == 'abt_buy' or self.domain== 'amazon_google':
                if (pair_data['name_jaccard'] == 1.0 or (pair_data['name_relaxed_jaccard']>0.6 and pair_data['description_relaxed_jaccard'] >0.7) or pair_data['name_relaxed_jaccard']>0.9 or pair_data['name_cosine_tfidf']>0.7 or pair_data['description_cosine_tfidf']>0.7):
                    label = 1

            elif self.domain == 'citation':
                if (pair_data['title_overlap'] == 1.0 or (pair_data['title_containment']>0.8 and pair_data['authors_overlap']==1.0) or
                   (pair_data['authors_overlap'] == 1.0 and pair_data['year_num_equal']==1) or ( pair_data['frequent_part_title_containment']>0.8 and pair_data['authors_containment']>0.6)): label = 1
            
             
            if (label == 1): 
                if (self.labels[p] == 1): 
                    correct_samples = correct_samples+1
                    truepositives = truepositives+1
                else: falsepositives = falsepositives+1
                boot_labels.append(label)    
                boot_ids.append(self.ids[p])
                boot_indices.append(helping_index[p])
            else : 
                if (self.labels[p] == 0): correct_samples = correct_samples+1
                else : falsenegatives = falsenegatives+1
                boot_labels.append(label)    
                boot_ids.append(self.ids[p])
                boot_indices.append(helping_index[p])

        self.sample['data'] = self.data.iloc[boot_indices]
        self.sample['labels'] = boot_labels
        self.sample['ids'] = boot_ids
        self.sample['indices'] = boot_indices   
        self.sample['correctness'] = float(correct_samples)/float(len(boot_ids))
        precision = float(truepositives)/float(truepositives+falsepositives)
        recall = float(truepositives)/float(truepositives+falsenegatives)
        self.sample['f1'] = 2.0*precision*recall/(precision+recall)

        
       
