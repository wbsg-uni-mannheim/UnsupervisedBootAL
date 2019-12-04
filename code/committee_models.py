import learningalgos as la

class CommitteeModels(object):
    
    def __init__(self, learning_algorithm, noisy=False):
        self.committee = dict() # this will be filled with the committee of models ad their cost
        self.learning_algorithm = learning_algorithm
        self.noisy = noisy       
        if self.learning_algorithm == 'dt' : self.getDecisionTreeCommittee()
        elif self.learning_algorithm == 'rf' : self.getRandomForrestCommittee()
        elif self.learning_algorithm == 'logr' : self.getLogisticRegressionCommittee()
        elif self.learning_algorithm == 'xgboost' : self.getXGBoostCommittee()
        elif self.learning_algorithm == 'svm' : self.getSVMCommittee()
                
   
    def getDecisionTreeCommittee(self):
        max_depth_list = [None, 3, 5, 10, 15]
        min_samples_leaf_list = [3, 5, 10]
        criterion_list = ['gini', 'entropy']
        models = []
        
        for max_depth_ in max_depth_list:
                for criterion_ in criterion_list:
                        for min_samples_leaf_ in min_samples_leaf_list:
                            models.append(la.DecisionTree_(random_state=1, max_depth=max_depth_,criterion=criterion_, min_samples_leaf=min_samples_leaf_, noisy=self.noisy))
                    
        self.committee['models'] = models
        
    def getRandomForrestCommittee(self):
        n_estimators_list = [10, 50, 100]
        max_depth_list = [None, 3, 5, 10, 15]
        min_samples_leaf_list = [3, 5, 10]
        models = []

        for n_estimators_ in n_estimators_list:
            for max_depth_ in max_depth_list:
                for min_samples_leaf_ in min_samples_leaf_list:
                    models.append(la.RandomForest_(random_state=1, max_depth=max_depth_, n_estimators=n_estimators_, min_samples_leaf=min_samples_leaf_, noisy=self.noisy))
                        
        self.committee['models'] = models
    
    def getLogisticRegressionCommittee(self):
        penalty_list = ['l1', 'l2']
        fit_intercept_list = [True, False]
        solver_list = ['liblinear', 'saga']
        max_iter_list = [50, 100, 150]
        models = []
        
        for penalty_ in penalty_list:
            for fit_intercept_ in fit_intercept_list:
                for solver_ in solver_list:
                    for max_iter_ in max_iter_list:
                        models.append(la.LogisticRegression_(penalty=penalty_, fit_intercept=fit_intercept_, solver=solver_, max_iter=max_iter_, noisy=self.noisy))
                        
        self.committee['models'] = models
        
    def getXGBoostCommittee(self):
        n_estimators_list = [100, 150, 200]
        learning_rate_list = [0.05, 0.1, 0.15]
        max_depth_list = [3, 5, 7]
        models = []
        
        for n_estimators_ in n_estimators_list:
            for learning_rate_ in learning_rate_list:
                for max_depth_ in max_depth_list:
                    models.append(la.XGBClassifier_(random_state=1, n_estimators=n_estimators_,learning_rate=learning_rate_,  max_depth=max_depth_, noisy=self.noisy))
                    #models.append(XGBClassifier_(random_state=1, n_estimators=n_estimators_,learning_rate=learning_rate_))
                    
        self.committee['models'] = models
        
    def getSVMCommittee(self):
        kernel_list = ['linear', 'rbf']
        gamma_list = [0.1, 1, 5, 10]
        C_list = [0.5, 1, 5, 10]
        models = []
        
        for kernel_ in kernel_list:
            for gamma_ in gamma_list:
                for C_ in C_list:
                    models.append(la.SVC_(random_state=1, kernel=kernel_, gamma=gamma_, C=C_, noisy=self.noisy))
        
        self.committee['models'] = models
