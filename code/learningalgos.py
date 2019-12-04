from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import sklearn.linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from libact.models import *
from libact.base import *
import libact
import xgboost as xgb
import numpy as np
from learning import *

class RandomForest_(libact.base.interfaces.ProbabilisticModel):

    """Random Forest Classifier

    """

    def __init__(self,  noisy = False, *args, **kwargs):
        self.model = RandomForestClassifier(*args, **kwargs)
        self.noisy = noisy
        self.name = "rf"        

    def train(self, dataset, *args, **kwargs):
        if self.noisy: 
            return self.model.fit(*(dataset.unsupervised_format_sklearn() + args), **kwargs)
        else:
            return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)
    def predict_proba(self, feature, *args, **kwargs):
        return self.model.predict_proba(feature, *args, **kwargs)
    
    def feature_importances_(self):
        return self.model.feature_importances_
    
    def get_params(self):
        return self.model.get_params
    
class XGBClassifier_(libact.base.interfaces.ProbabilisticModel):

    """Gradient Boosting Classifier

    """

    def __init__(self, noisy = False, *args, **kwargs):
        self.model = xgb.XGBClassifier(*args, **kwargs)
        self.noisy = noisy
        self.name = "xgboost"

    def train(self, dataset, *args, **kwargs):
         if self.noisy: 
            return self.model.fit(*(dataset.unsupervised_format_sklearn() + args), **kwargs)
         else:
            return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        feature_array = np.asarray(feature)
        return self.model.predict(feature_array, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)
    def predict_proba(self, feature, *args, **kwargs):
        feature_array = np.asarray(feature)
        return self.model.predict_proba(feature_array, *args, **kwargs)
    
    def feature_importances_(self):
        return self.model.feature_importances_
    
    def get_params(self):
        return self.model.get_params
    
class SVC_(libact.base.interfaces.ProbabilisticModel):

    """SVC

    """

    def __init__(self, noisy = False, *args, **kwargs):
        self.model = SVC(*args, **kwargs)
        self.noisy = noisy
        self.name = "svm"

    def train(self, dataset, *args, **kwargs):
         if self.noisy: 
            return self.model.fit(*(dataset.unsupervised_format_sklearn() + args), **kwargs)
         else:
            return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)
    def predict_proba(self, feature, *args, **kwargs):
        return self.model.predict_proba(feature, *args, **kwargs)
    
    def feature_importances_(self):
        if model.kernel != 'linear':
            display("Cannot print feature importances without a linear kernel")
            return
        return self.model.coef_.ravel()
    
    def get_params(self):
        return self.model.get_params
    
    def kernel(self):
        return self.model.kernel
    
class DecisionTree_(libact.base.interfaces.ProbabilisticModel):

    """Decision Tree Classifier

    """

    def __init__(self, noisy = False, *args, **kwargs):
        self.model = DecisionTreeClassifier(*args, **kwargs)
        self.noisy = noisy
        self.name = "dt"


    def train(self, dataset, *args, **kwargs):
        if self.noisy: 
            return self.model.fit(*(dataset.unsupervised_format_sklearn() + args), **kwargs)
        else:
            return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)
    def predict_proba(self, feature, *args, **kwargs):
        return self.model.predict_proba(feature, *args, **kwargs)
    
    def feature_importances_(self):
        return self.model.feature_importances_
    
    def get_params(self):
        return self.model.get_params
    
class LinearRegression_(libact.base.interfaces.ProbabilisticModel):
    

    def __init__(self, noisy = False, *args, **kwargs):
        self.model = sklearn.linear_model.LinearRegression(*args, **kwargs)
        self.noisy = noisy
        self.name = "linr"

    def train(self, dataset, *args, **kwargs):
        if self.noisy: 
            return self.model.fit(*(dataset.unsupervised_format_sklearn() + args), **kwargs)
        else:
            return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args), **kwargs)

    def predict_real(self, feature, *args, **kwargs):
        dvalue = self.model.decision_function(feature, *args, **kwargs)
        if len(np.shape(dvalue)) == 1:  # n_classes == 2
            return np.vstack((-dvalue, dvalue)).T
        else:
            return dvalue

    def predict_proba(self, feature, *args, **kwargs):
        return self.model.predict_proba(feature, *args, **kwargs)
    
    def feature_importances_(self):
        return self.model.coef_.ravel()
    
    def get_params(self):
        return self.model.get_params
    

class LogisticRegression_(libact.base.interfaces.ProbabilisticModel):
    

    def __init__(self, noisy= False,  *args, **kwargs):
        self.model = sklearn.linear_model.LogisticRegression(*args, **kwargs)
        self.noisy = noisy
        self.name = "logr"

    def train(self, dataset, *args, **kwargs):
        if self.noisy: 
            return self.model.fit(*(dataset.unsupervised_format_sklearn() + args), **kwargs)
        else:
            return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args), **kwargs)

    def predict_real(self, feature, *args, **kwargs):
        dvalue = self.model.decision_function(feature, *args, **kwargs)
        if len(np.shape(dvalue)) == 1:  # n_classes == 2
            return np.vstack((-dvalue, dvalue)).T
        else:
            return dvalue

    def predict_proba(self, feature, *args, **kwargs):
        return self.model.predict_proba(feature, *args, **kwargs)
    
    def feature_importances_(self):
        return self.model.coef_.ravel()
    def get_params(self):
        return self.model.get_params
    
class Knn_(libact.base.interfaces.ProbabilisticModel):
    

    def __init__(self, noisy= False,  *args, **kwargs):
        self.model = sklearn.neighbors.KNeighborsClassifier(*args, **kwargs)
        self.noisy = noisy
        self.name = "knn"

    def train(self, dataset, *args, **kwargs):
        if self.noisy: 
            return self.model.fit(*(dataset.unsupervised_format_sklearn() + args), **kwargs)
        else:
            return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args), **kwargs)

    def predict_real(self, feature, *args, **kwargs):
        dvalue = self.model.decision_function(feature, *args, **kwargs)
        if len(np.shape(dvalue)) == 1:  # n_classes == 2
            return np.vstack((-dvalue, dvalue)).T
        else:
            return dvalue

    def predict_proba(self, feature, *args, **kwargs):
        return self.model.predict_proba(feature, *args, **kwargs)
    
    def feature_importances_(self):
        return self.model.coef_.ravel()
    def get_params(self):
        return self.model.get_params

