import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd



# PCA with 99% variance retained
# tutorial followed https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
# how can I still interpret the learned model?
def reduceDimensions_PCA(data, threshold):
    #first scale the data
    scaler = StandardScaler()
    scaler.fit(data)
    transf_data = scaler.transform(data)
    pca = PCA(threshold)
    #WATCH OUT: if we have a split with training and test data we should only apply PCA on the training side
    data_pca = pca.fit(data)
    print"Reduced from ", data.shape, " features to",pca.n_components_, "PCA components (variance =", threshold,"). "
    
#removes features with a variance close to 0
def reduceDimensions_Variance(data,showEliminatedFeatures, varianceThreshold):
    if (showEliminatedFeatures): display(data.var()[data.var() < varianceThreshold])
    reduced_data = data.drop(data.var()[data.var() < varianceThreshold].index.values, axis=1)
    print"Reduced training data dimensions using ", varianceThreshold, " Variance Threshold: ",reduced_data.shape

    return reduced_data


def reduceDimensions_Correlation(data, showEliminatedFeatures, correlationThreshold):
   
    cor = data.corr().abs()
    #plot correlation heatmap
    ax = sns.heatmap(cor, yticklabels=False, xticklabels=False)   
    ax.set_title("Feature correlation heatmap before dim. reduction")
    plt.show()
    # Select upper triangle of correlation matrix
    upper = cor.where(np.triu(np.ones(cor.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > correlationThreshold)]
    if (showEliminatedFeatures): display(to_drop)
    reduced_data = data.drop(to_drop, axis=1)
    print"Reduced training data dimensions using ", correlationThreshold, " Correlation Threshold: ",reduced_data.shape
    
    cor_reduced = reduced_data.corr().abs()
    ax_red = sns.heatmap(cor_reduced, yticklabels=False, xticklabels=False)   
    ax_red.set_title("Feature correlation heatmap after dim. reduction")
    plt.show()
    return reduced_data