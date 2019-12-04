import numpy as np
from sklearn.cluster import KMeans
from IPython.display import Markdown, display
import random
import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import euclidean_distances
from sets import Set
import sys

class Bootstrapping(object):

    def __init__(self, sample_size=0, data=None, labels=None, ids=None, bootstrap_method=None, setting=None):
        self.sample_size = sample_size
        self.data = data
        self.labels = labels
        self.ids = ids
        self.bootstrap_method = bootstrap_method
        self.penalty_factor = 0.01
        self.sorted_dataset = []
        self.sample= dict()
        if self.bootstrap_method == 'bowtopbottom': self.bootstrapBoW(selection = 'topbottom')
        elif self.bootstrap_method == 'bowtop': self.bootstrapBoW(selection = 'top')
        elif self.bootstrap_method == 'bowboundary': self.bootstrapBoW(selection = 'boundary')
        elif self.bootstrap_method == 'attrtopbottom': self.bootstrapAttribute(selection = 'topbottom')
        elif self.bootstrap_method == 'attrtop': self.bootstrapAttribute(selection = 'top')
        elif self.bootstrap_method == 'attrtopbottom_density': self.bootstrapAttribute(selection = 'topbottom', weight='density')
        elif self.bootstrap_method == 'attrtopbottom_penalty': self.bootstrapAttribute(selection = 'topbottom', weight='penalty')
        elif self.bootstrap_method == 'attrtop_density': self.bootstrapAttribute(selection = 'top', weight='density')
        elif self.bootstrap_method == 'attrboundary': self.bootstrapAttribute(selection = 'boundary')
        elif self.bootstrap_method == 'attrboundary_density': self.bootstrapAttribute(selection = 'boundary', weight='density')
        elif self.bootstrap_method == 'random' : self.bootstrapRandom()
        elif self.bootstrap_method == 'randomFromClusters' : self.bootstrapRandomFromClusters()
        elif self.bootstrap_method == 'centralFromClusters' : self.bootstrapCentralElementsFromClusters()
        else : display("Unknown boostrapping method: "+self.bootstrap_method)
            
        if setting is 'n_to_one':
            self.addNegativesFromPositives()
        self.printLabelDistr()
        
        
    
    #uses cosine tfidf over concatenated feature values and selects the sample_size/2 top positive and sample_size/2 bottom pairs orders by desc sim.
    def bootstrapBoW(self, selection):
        display(Markdown("<span style='color:blue;font-size:160%'><b> Get bootstrapped pairs using BoW and cosine sim. score based on tfidf vectors. </b></span>"))
        cosine_tfidf_columns = self.data['cosine_tfidf']
        
        #rescale
        cosine_tfidf_columns = np.interp(cosine_tfidf_columns, (cosine_tfidf_columns.min(), cosine_tfidf_columns.max()), (0, +1))
        sorted_dataset = zip(self.labels, self.ids, cosine_tfidf_columns, np.arange(self.ids.size))
       
        random.Random(1).shuffle(sorted_dataset)
        sorted_dataset.sort(key = lambda t: t[2])
        self.sorted_dataset = map(lambda x:x[2],sorted_dataset)
        if selection == 'topbottom':
            self.getTopBottomElementsFromSortedDataset(sorted_dataset)
        elif selection == 'top':
            self.getTopElementsFromSortedDataset(sorted_dataset)
        elif selection == 'boundary':
            self.getElementsCloseToDecisionBoundaryFromSortedDataset(sorted_dataset)
        elif selection == 'otsu':
            self.getOtsuThrElementsFromSortedDataset(sorted_dataset)
        elif selection == 'valley':
            self.getValleyThrElementsFromSortedDataset(sorted_dataset)
        elif selection == 'auc':
            self.getAUCThrElementsFromSortedDataset(sorted_dataset)
        elif selection == 'static':
            self.getStaticThresholdElementsFromSortedDataset(sorted_dataset)
        elif selection == 'elbow':
            self.getElbowThrElementsFromSortedDataset(sorted_dataset)
            
        else: 
            display("Unknown bootstrapping selection method : "+selection)
        
    # use the heuristic of Yaser and calculate the weighted average of BoW sim (cosine) and Attribute Similarity (all other attributes for every pair of feature values). Normalize the scores and  select pairs based on the selection strategy
    def bootstrapAttribute(self, selection, weight=None):
        
        display(Markdown("<span style='color:blue;font-size:160%'><b> Get bootstrapped pairs using feature value pairwise similarity *all* weighted by the feature density (*0.5) and cosine with tfidf (*0.5). </b></span>"))
        cosine_tfidf_column = self.data['cosine_tfidf']
        other_columns  = self.data.drop(['cosine_tfidf'], axis=1)
        other_columns = other_columns.replace(-1.0,np.nan)
        if weight is None:
            other_columns_sum = other_columns.sum(axis=1, skipna=True)
            other_columns_mean = other_columns_sum/len(other_columns.columns)
        if weight is 'penalty':
            other_columns_sum = other_columns.sum(axis=1, skipna=True)
            dense_columns_per_row = other_columns.apply(lambda x: x.count(), axis=1)
            penalty = dense_columns_per_row.apply(lambda x, col_size=len(other_columns.columns), penalty=self.penalty_factor : ((col_size-x)*penalty))
            other_columns_mean = (other_columns_sum/dense_columns_per_row)-penalty
        if weight is 'density':
            column_weights = []
            for c in other_columns:
                nan_values = other_columns[c].isna().sum()
                ratio = float(nan_values)/float(len(other_columns[c]))
                column_weights.append(1.0-ratio)
            
            #print(column_weights)
            weighted_columns = other_columns*column_weights
            #print(weighted_columns.iloc[0])
            
            other_columns_sum = weighted_columns.sum(axis=1, skipna=True)
            
            #display("Jaccard_colum_sum:", jaccard_column_sum)
            other_columns_mean = other_columns_sum/len(other_columns.columns)
        
        #rescale 
        other_columns_mean = np.interp(other_columns_mean, (other_columns_mean.min(), other_columns_mean.max()), (0, +1))
        cosine_tfidf_column = np.interp(cosine_tfidf_column, (cosine_tfidf_column.min(), cosine_tfidf_column.max()), (0, +1))
        #print("Jaccard mean:",jaccard_column_mean[0])
          
        weighted_cosine = cosine_tfidf_column*0.5
        #print("Weighted cosine:", weighted_cosine[0])
        weighted_other_columns = other_columns_mean*0.5
        #print("Weighted jaccard:",weighted_jaccard[0])
        sum_weighted_similarity = weighted_other_columns+weighted_cosine
        #print("SUM weighted similarity:",sum_weighted_similarity[0])
        
        sorted_dataset = zip(self.labels, self.ids, sum_weighted_similarity, np.arange(self.ids.size))
        random.Random(1).shuffle(sorted_dataset)
        sorted_dataset.sort(key = lambda t: t[2])
        self.sorted_dataset = map(lambda x:x[2],sorted_dataset)
        if selection == 'topbottom':
            self.getTopBottomElementsFromSortedDataset(sorted_dataset)
        elif selection == 'boundary':
            self.getElementsCloseToDecisionBoundaryFromSortedDataset(sorted_dataset)
        elif selection == 'top':
            self.getTopElementsFromSortedDataset(sorted_dataset)
        elif selection == 'otsu':
            self.getOtsuThrElementsFromSortedDataset(sorted_dataset)
        elif selection == 'valley':
            self.getValleyThrElementsFromSortedDataset(sorted_dataset)
        elif selection == 'auc':
            self.getAUCThrElementsFromSortedDataset(sorted_dataset)
        elif selection == 'static':
            self.getStaticThresholdElementsFromSortedDataset(sorted_dataset)
        elif selection == 'elbow':
            self.getElbowThrElementsFromSortedDataset(sorted_dataset)
        
        else: 
            display("Unknown bootstrapping selection method!!")


    def getElementsCloseToDecisionBoundaryFromSortedDataset(self, sorted_dataset):
        
        display("Select elements close to the decision boundary")
        mindistance = 1
        index_ofmindistance = 0
        rescaled_values = []
        for element in range(len(sorted_dataset)):
            rescaled_values.append(sorted_dataset[element][2])
            distance = abs(sorted_dataset[element][2]-0.5)
            if (distance < mindistance):
                mindistance= distance
                index_ofmindistance = element
        
        
        display("Index of min distance:",index_ofmindistance)
        
        #middle_index = len(sorted_dataset)/2
        middle_index = index_ofmindistance
        middle_up_index = middle_index - self.sample_size/2
        middle_down_index = middle_index + self.sample_size/2
        
        middle_elements = sorted_dataset[middle_up_index:middle_down_index]
        middle_labels = []       
        for x in middle_elements : middle_labels.append(x[0])
       
        #add items until you find a positive one
        while not ((1 in middle_labels) and (0 in middle_labels)):
            middle_down_index += 1
            middle_elements = sorted_dataset[middle_up_index:middle_down_index]
            middle_labels = []       
            for x in middle_elements : middle_labels.append(x[0])
                   
        boot_data = []
        boot_labels = []
        boot_ids = []
        boot_indices = []

        for boot_element in (middle_elements):
            boot_labels.append(boot_element[0])
            boot_ids.append(boot_element[1])
            boot_indices.append(boot_element[3])

        self.sample['data'] = self.data.iloc[boot_indices]
        self.sample['labels'] = boot_labels
        self.sample['ids'] = boot_ids
        self.sample['indices'] = boot_indices 
     
    def getTopElementsFromSortedDataset(self, sorted_dataset):
        
        display("Select top elements. If positive add the corresponding negatives based on the n-->setting")
        
        top_elements = sorted_dataset[-self.sample_size:] #the dataset is sorted ascending
        #display("Top_elements:",top_elements)

        boot_data = []
        boot_labels = []
        boot_ids = []
        boot_indices = []

        for boot_element in top_elements:
            boot_labels.append(boot_element[0])
            boot_ids.append(boot_element[1])
            boot_indices.append(boot_element[3])

            
        self.sample['data'] = self.data.iloc[boot_indices]
        self.sample['labels'] = boot_labels
        self.sample['ids'] = boot_ids
        self.sample['indices'] = boot_indices   
    
    def addNegativesFromPositives(self):
        
        boot_labels = self.sample['labels']
        boot_data = self.sample['data']
        boot_ids = self.sample['ids']
        boot_indices = self.sample['indices']
        
        additional_elements = []
        zipped_sample = list(zip(self.sample['labels'],self.sample['ids']))
        for s in zipped_sample:
            if s[0] == 1:
                additional_elements.append(s[1])

         
        zipped_dataset = list(zip(self.labels, self.ids))
        for a in additional_elements:
            sourceid = a.split('-')[0] 
            for i in range(len(zipped_dataset)):
                if zipped_dataset[i][1].split('-')[0] == sourceid:
                    if a != zipped_dataset[i][1]:
                        if (zipped_dataset[i][0] == 1):
                            print("This should not happen in bootstrapping while adding negative knowledge.")
                        boot_labels.append(0)
                        boot_ids.append(zipped_dataset[i][1])
                        boot_indices.append(i)
                        
                        
        self.sample['data'] = self.data.iloc[boot_indices]
        self.sample['labels'] = boot_labels
        self.sample['ids'] = boot_ids
        self.sample['indices'] = boot_indices   
        
        
    def getTopBottomElementsFromSortedDataset(self, sorted_dataset):
        
        display("Select top and bottom elements")
        import pdb; pdb.set_trace();
        bottom_elements = sorted_dataset[:(self.sample_size/2)]
        #display("Top_elements:",top_elements)
        top_elements = sorted_dataset[-(self.sample_size/2):]
        #display("Bottom_elements:",bottom_elements)

        boot_data = []
        boot_labels = []
        boot_ids = []
        boot_indices = []

        for boot_element in (top_elements+bottom_elements):
            boot_labels.append(boot_element[0])
            boot_ids.append(boot_element[1])
            boot_indices.append(boot_element[3])

        self.sample['data'] = self.data.iloc[boot_indices]
        self.sample['labels'] = boot_labels
        self.sample['ids'] = boot_ids
        self.sample['indices'] = boot_indices   
    
    def bootstrapRandom(self):
        display(Markdown("<span style='color:blue;font-size:160%'><b> Get random sample for bootstrapping the AL method. Sample until sample size is reached and at least one positive example is picked. </b></span>"))
        
        sample_indices = []
        i=0
        annotatedpositive = False
        while (i<self.sample_size or  (not annotatedpositive)):
            sample_i = random.randint(0, len(self.labels)-1)
            while (sample_i in sample_indices):
                sample_i = random.randint(0, len(self.labels)-1)

            if (self.labels[sample_i] == 1): annotatedpositive = True
            sample_indices.append(sample_i)
            i += 1

        self.sample['data'] = self.data.iloc[sample_indices]
        self.sample['labels'] = list(self.labels[sample_indices])
        self.sample['ids'] = list(self.ids[sample_indices])
        self.sample['indices'] = list(sample_indices)


    def bootstrapRandomFromClusters(self):
                
        display(Markdown("<span style='color:blue;font-size:160%'><b> Get cluster-based sample by clustering with K-means with K 10. Then select randomly from the created clusters. </b></span>"))

        cluster_sample_size = (self.sample_size/10)
        kmeans = self.cluster_kmeans(10)
        sample_indices = []
        i=0
        while (i<10 or (1 not in self.labels[sample_indices] or 0 not in self.labels[sample_indices])):
            # (1) indices of all the points from X that belong to cluster i
            C_i = np.where(kmeans.labels_ == i)[0].tolist() 
            n_i = len(C_i) # number of points in cluster i

            # (2) indices of the points from X to be sampled from cluster i
            sample_i = np.random.choice(C_i, cluster_sample_size)
            sample_indices.extend(sample_i)
            i+=1
            
        self.sample['data'] = self.data.iloc[list(sample_indices)]
        self.sample['labels'] = self.labels[list(sample_indices)].tolist()
        self.sample['ids'] = self.ids[list(sample_indices)].tolist()
        self.sample['indices'] = list(sample_indices)
      
    def bootstrapCentralElementsFromClusters(self):
        display(Markdown("<span style='color:blue;font-size:160%'><b> Get cluster-based sample by clustering with K-means with K 10. Then select the most central data points fro every cluster. </b></span>"))
    
        cluster_sample_size = (self.sample_size/10)
        kmeans = self.cluster_kmeans(10)
        sample_indices = Set([])
        i=0
        while (i<10):
            # (1) indices of all the points from X that belong to cluster i
            C_i = np.where(kmeans.labels_ == i)[0].tolist() 
            cluster_center = kmeans.cluster_centers_[i]
            cluster_data = self.data.iloc[C_i]
            #get euklideian distance of every point of the cluster to the center
            distances_from_center = []
            for row in cluster_data.values:
                distance = euclidean_distances([row], [cluster_center])
                distances_from_center.extend(distance[0])

            distances_index = list(zip(C_i,distances_from_center))
            distances_index.sort(key = lambda t: t[1])
                        
            if i == 9: #if the labels are not complete take more elements from the last cluster till you have found at least one item from every class
                additional_elements = 1
                while (1 not in self.labels[list(sample_indices)] or 0 not in self.labels[list(sample_indices)]):
                    top_elements = distances_index[:(cluster_sample_size+additional_elements)]
                    additional_elements += 1
                    for te in top_elements:
                        sample_indices.add(te[0])
            else: 
                top_elements = distances_index[:cluster_sample_size]
                for te in top_elements:
                    sample_indices.add(te[0])
            
            i+=1
        
            
            
        #import pdb; pdb.set_trace()     
        self.sample['data'] = self.data.iloc[list(sample_indices)]
        self.sample['labels'] = self.labels[list(sample_indices)]
        self.sample['ids'] = self.ids[list(sample_indices)]
        self.sample['indices'] = list(sample_indices)

    def cluster_kmeans(self, n_clusters):
        #clusterdata = self.data.replace(-1.0,np.nan)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42) 
        kmeans.fit(self.data)
        clust_labels = kmeans.predict(self.data)
        unique, counts = np.unique(clust_labels, return_counts=True)
        display(dict(zip(unique, counts)))
        cluster = pd.DataFrame(clust_labels)
        #uncomment if you want to add the cluster info aas an additional feature to the dataset
        #cluster_kmeans_data.insert((self.data.shape[1]),'kmeans_cluster',clust_labels)
        return kmeans

    def printLabelDistr(self):
        unique, counts = np.unique(self.sample['labels'], return_counts=True)
        print("Class distribution in sample:")
        display(dict(zip(unique, counts)))
