import numpy as np
import copy
import calibration
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


class JackknifePlus:
    def __init__(self, X, Y, X_test, C, black_box, random_state_training=2020,random_state_test=2021):
        self.black_box = black_box
        self.n = X.shape[0]  ## size of training data
        self.C = C           ## number of classes
        self.n_test=X_test.shape[0]               ## number of test data points
        self.black_box = black_box                ## choice of predictive model
        mu = self.black_box.fit(X,Y)              ## fit the black box model to the entire training data
         
        ## The purpose of fitting black box model on the entire test data is to construct I(X) which will be considered given sets for test data.
        ## The purpose is to provide coverage guarantees on I(X) for test data.
        
        self.predicted_test_prob = mu.predict_proba(X_test)  ## predicted probabiities for classes in test data
        self.grey_box = calibration.ProbAccum(self.predicted_test_prob) ## function provides ranks of classes, order of classes, sorted probabilities and cumulated sorted probabilities
        
        
    ## The functions below can be used to produce different kinds of given sets I(X)
    
    
    def given_sets_fixed_length(self, K):   ## given sets of fixed length K, i.e., K classes with maximum softmax for test data fitted by black box on complete training data.
        self.K = K
        self.given_sets_test = self.grey_box.order[:,:K]
    
    
    def given_sets_min_prob_sum(self,s):  ## given sets with variable set size, but summation of softmax within given sets above given threshold s, the classes with maximum predicted                                                 ## softmax are chosen
        self.given_sets_test = []
        for i_ in range(self.n_test):
            given_set_i_ = [self.grey_box.order[i_][0]]
            j=1
            while self.grey_box.prob_sort_sum[i_][j]<s:
                given_set_i_.append(self.grey_box.order[i_][j])
                j+=1
            self.given_sets_test.append(given_set_i_)
            
    def given_sets_min_prob(self,c):      ## given sets with variable set size, but classes with predicted softmax greater than threshold c.
        self.given_sets_test = []
        for i_ in range(self.n_test):
            given_set_i_ = [self.grey_box.order[i_][0]]
            j=1
            while self.grey_box.prob_sort[i_][j]>=c:
                given_set_i_.append(self.grey_box.order[i_][j])
                j+=1
            self.given_sets_test.append(given_set_i_)
            
            
    def given_sets(self,set_type,u):
        if set_type == "fixed length":
            self.given_sets_fixed_length(u)
        elif set_type == "minimum set probability":
            self.given_sets_min_prob_sum(u)
        elif set_type == "minimum class probability":
            self.given_sets_min_prob(u)
            
    ## The following function is sensitive to jackknife technique, if you work on split conforal or CV+, it will require modification.
    ## For the function i_ associates with test data points, i associates with training data points
    ## The function computes \sum_{i=1}^{n_train}  I(E(X_i,Y_i,..) < E(X_{i_}, j, ..)) for all test data points i_, for all the classes j 
    ## The (n_test, C) matrix is named score_comparison.
    
    
    def Score_comparison_function(self, X, Y, X_test, random_state_training=2020,random_state_test=2021): 
        rng = np.random.default_rng(random_state_training)
        epsilon_training = rng.uniform(low=0.0, high=1.0, size=self.n)
        rng = np.random.default_rng(random_state_test)
        epsilon_test = rng.uniform(low=0.0, high=1.0, size=self.n_test)
        
        # loop over data points in training data
        self.n_train = 0
        self.score_comparison = np.zeros((self.n_test,self.C))

        for i in range(self.n):
            if len(np.unique(np.delete(Y,i))) == self.C:
                self.n_train += 1
                # fit the black box model leaving one one
                mu_LOO = self.black_box.fit(np.delete(X,i,0),np.delete(Y,i))
                grey_box_training_i = calibration.ProbAccum(mu_LOO.predict_proba(X[i]))
                score_i = grey_box_training_i.calibrate_scores(Y[i], epsilon_training[i])
                grey_box_test_i = calibration.ProbAccum(mu_LOO.predict_proba(X_test))
                grey_box_test_i.all_scores(epsilon = epsilon_test)
                ## loop over test_data
                for i_ in range(self.n_test):
                    scores_given_test_i_ = grey_box_test_i.normalised_scores[i_]
                
                    for j in np.arange(self.C):
                        if score_i < scores_given_test_i_[j]:
                            self.score_comparison[i_][j] += 1
                    
                
    ## the function below provides miscoverage rate for each test data point, we could take an average to get a scalar value
    ## For further details, please see the related paper, 
    
    def miscoverage(self):
        self.results_lower = []
        self.results_upper = []
        for i_ in range(self.n_test):
            self.results_lower.append(max(self.score_comparison[i_][self.given_sets_test[i_]]))
            self.results_upper.append(min(self.score_comparison[i_][np.setdiff1d(np.arange(self.C),self.given_sets_test[i_])]))
        self.alpha = 1 - np.array(self.results_lower)/(self.n_train + 1)
        
    ## I have written the function below for sanity check purpose, i.e., to crosscheck if CI(alpha, x) is subset of I(X)

    def conformal_prediction_sets(self,alpha):
        prediction_sets = []
        for i_ in range(self.n_test):
            set_i_ = []
            for j in range(self.C):
                if self.score_comparison[i_][j] < (self.n+1)*(1-alpha[i_]):
                    set_i_.append(j)
            prediction_sets.append(set_i_)
        return(prediction_sets)
    
    ## This is sanity check for alpha
    

    def sanity_check(self,alpha):
        prediction_sets_alpha = self.conformal_prediction_sets(alpha)
        
        unvalid_intervals_alpha = []
        
        for i_ in range(self.n_test):
            if not set(prediction_sets_alpha[i_]).issubset(set(self.given_sets_test[i_])):
                unvalid_intervals_alpha.append(i_)
        return(unvalid_intervals_alpha)
                    
    def calibrate_alpha(self):
        prediction_sets = self.conformal_prediction_sets(self.alpha)
        
        for i_ in range(self.n_test):
            if not set(prediction_sets[i_]).issubset(set(self.given_sets_test[i_])):
                while not set(prediction_sets[i_]).issubset(set(self.given_sets_test[i_])):
                    self.alpha[i_] = self.alpha[i_] + 1/(self.n+1)
                    set_i_ = []
                    for j in range(self.C):
                        if self.score_comparison[i_][j] < (self.n+1)*(1-self.alpha[i_]):
                            set_i_.append(j)
                    prediction_sets[i_] = copy.deepcopy(set_i_)

#         self.alpha_2 = copy.deepcopy(self.alpha)
#         new_prediction_sets = []
#         for i_ in range(self.n_test):
#             new_prediction_sets.append(copy.deepcopy(prediction_sets[i_]))        
#             while self.alpha_2[i_]<=1 and len(prediction_sets[i_]) == len(new_prediction_sets[i_]):
#                 #print(alpha[i_])
#                 #print(prediction_sets[i_])
#                 #print(grey_box.given_sets_test[i_])
#                 self.alpha_2[i_] = self.alpha_2[i_] + 1/(self.n+1)
#                 set_i_ = []
#                 for j in range(self.C):
#                     if self.score_comparison[i_][j] < (self.n+1)*(1-self.alpha_2[i_]):
#                         set_i_.append(j)
#                 new_prediction_sets[i_] = copy.deepcopy(set_i_)

                
    def compilation(self,set_type,u,X, Y, X_test,Y_test, simulation = False, data_model = None):
        self.given_sets(set_type,u)
        self.miscoverage()
        #self.sanity_check()
        self.true_miscoverage = []
        ## true probabilites
        if simulation:

            p_test = data_model.compute_prob(X_test)

            ## true miscoverage

            for i_ in range(self.n_test):
                self.true_miscoverage.append(1-sum(p_test[i_][self.given_sets_test[i_]]))


        self.predicted_miscoverage_black_box = []

        for i_ in range(self.n_test):
            self.predicted_miscoverage_black_box.append(1-sum(self.predicted_test_prob[i_][self.given_sets_test[i_]]))
            
        self.true_miscoverage_indicator = 0
        
        for i_ in range(self.n_test):
            if not Y_test[i_] in self.given_sets_test[i_]:
                self.true_miscoverage_indicator += 1
        self.true_miscoverage_indicator /= self.n_test
        train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=0.33, random_state=42)

        mu_train = self.black_box.fit(train_X,train_Y)
        prob_valid = mu_train.predict_proba(valid_X)
        grey_box_valid = calibration.ProbAccum(prob_valid)
        grey_box_valid.given_sets(set_type,u)

        given_sets_valid = grey_box_valid.given_sets_test

        validation_coverage = 0
        for i in range(len(valid_Y)):
            if valid_Y[i] in given_sets_valid[i]:
                validation_coverage+=1   
                
        self.validation_coverage = validation_coverage/len(valid_Y)     

        


class CVPlus:
    def __init__(self, X, Y, X_test, C, black_box, random_state_training=2020,random_state_test=2021, n_folds = 10, random_state = 2022):
        
        self.black_box = black_box
        self.n = X.shape[0]  ## size of training data
        self.C = C           ## number of classes
        self.n_test=X_test.shape[0]               ## number of test data points
        self.black_box = black_box                ## choice of predictive model
        mu = self.black_box.fit(X,Y)              ## fit the black box model to the entire training data
        self.n_folds = n_folds
        self.cv = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)
        ## The purpose of fitting black box model on the entire test data is to construct I(X) which will be considered given sets for test data.
        ## The purpose is to provide coverage guarantees on I(X) for test data.
        
        self.predicted_test_prob = mu.predict_proba(X_test)  ## predicted probabiities for classes in test data
        self.grey_box = calibration.ProbAccum(self.predicted_test_prob) ## function provides ranks of classes, order of classes, sorted probabilities and cumulated sorted probabilities
        
        
    ## The functions below can be used to produce different kinds of given sets I(X)
    
    
    def given_sets_fixed_length(self, K):   ## given sets of fixed length K, i.e., K classes with maximum softmax for test data fitted by black box on complete training data.
        self.K = K
        self.given_sets_test = self.grey_box.order[:,:K]
    
    
    def given_sets_min_prob_sum(self,s):  ## given sets with variable set size, but summation of softmax within given sets above given threshold s, the classes with maximum predicted                                                 ## softmax are chosen
        self.given_sets_test = []
        for i_ in range(self.n_test):
            ## add the class with lastest softmax to make sure given set is non empty
            given_set_i_ = [self.grey_box.order[i_][0]]
            j=1
            while self.grey_box.prob_sort_sum[i_][j]<s:
                given_set_i_.append(self.grey_box.order[i_][j])
                j+=1
            self.given_sets_test.append(given_set_i_)
            
    def given_sets_min_prob(self,c):      ## given sets with variable set size, but classes with predicted softmax greater than threshold c.
        self.given_sets_test = []
        for i_ in range(self.n_test):
            ## add the class with lastest softmax to make sure given set is non empty
            given_set_i_ = [self.grey_box.order[i_][0]]
            j=1
            while self.grey_box.prob_sort[i_][j]>=c:
                given_set_i_.append(self.grey_box.order[i_][j])
                j+=1
            self.given_sets_test.append(given_set_i_)
            
            
    def given_sets(self,set_type,u):
        if set_type == "fixed length":
            self.given_sets_fixed_length(u)
        elif set_type == "minimum set probability":
            self.given_sets_min_prob_sum(u)
        elif set_type == "minimum class probability":
            self.given_sets_min_prob(u)
            
    ## The following function is sensitive to jackknife technique, if you work on split conforal or CV+, it will require modification.
    ## For the function i_ associates with test data points, i associates with training data points
    ## The function computes \sum_{i=1}^{n_train}  I(E(X_i,Y_i,..) < E(X_{i_}, j, ..)) for all test data points i_, for all the classes j 
    ## The (n_test, C) matrix is named score_comparison.
    
    
    def Score_comparison_function(self, X, Y, X_test, random_state_training=2020,random_state_test=2021): 
        rng = np.random.default_rng(random_state_training)
        epsilon_training = rng.uniform(low=0.0, high=1.0, size=self.n)
        rng = np.random.default_rng(random_state_test)
        epsilon_test = rng.uniform(low=0.0, high=1.0, size=self.n_test)
        self.n_train = 0
        self.score_comparison = np.zeros((self.n_test,self.C))
        
        # loop over folds
        for train_index, test_index in self.cv.split(X):
            if len(np.unique(Y[train_index])) == self.C:
                mu_LOO = self.black_box.fit(X[train_index],Y[train_index])
                grey_box_training_i = calibration.ProbAccum(mu_LOO.predict_proba(X[test_index]))
                scores_calib = grey_box_training_i.calibrate_scores(Y[test_index], epsilon_training[test_index])
                grey_box_test_i = calibration.ProbAccum(mu_LOO.predict_proba(X_test))
                grey_box_test_i.all_scores(epsilon = epsilon_test)
                ## loop over test_data
                for i_ in range(self.n_test):
                    scores_given_test_i_ = grey_box_test_i.normalised_scores[i_]
                    for score_i in scores_calib:
                        for j in np.arange(self.C):
                            if score_i < scores_given_test_i_[j]:
                                self.score_comparison[i_][j] += 1
                   
                
    ## the function below provides miscoverage rate for each test data point, we could take an average to get a scalar value
    ## For further details, please see the related paper, 
    
    def miscoverage(self):
        self.results_lower = []
        for i_ in range(self.n_test):
            self.results_lower.append(max(self.score_comparison[i_][self.given_sets_test[i_]]))
        self.alpha = 1 - np.array(self.results_lower)/(self.n + 1)
        
    ## I have written the function below for sanity check purpose, i.e., to crosscheck if CI(alpha, x) is subset of I(X)

    def conformal_prediction_sets(self,alpha):
        prediction_sets = []
        for i_ in range(self.n_test):
            set_i_ = []
            for j in range(self.C):
                if self.score_comparison[i_][j] < (self.n+1)*(1-alpha[i_]):
                    set_i_.append(j)
            prediction_sets.append(set_i_)
        return(prediction_sets)
    
    ## This is sanity check for alpha
    

    def sanity_check(self,alpha):
        prediction_sets_alpha = self.conformal_prediction_sets(alpha)
        
        unvalid_intervals_alpha = []
        
        for i_ in range(self.n_test):
            if not set(prediction_sets_alpha[i_]).issubset(set(self.given_sets_test[i_])):
                unvalid_intervals_alpha.append(i_)
        return(unvalid_intervals_alpha)
                    
    def calibrate_alpha(self):
        prediction_sets = self.conformal_prediction_sets(self.alpha)
        
        for i_ in range(self.n_test):
            if not set(prediction_sets[i_]).issubset(set(self.given_sets_test[i_])):
                while not set(prediction_sets[i_]).issubset(set(self.given_sets_test[i_])):
                    self.alpha[i_] = self.alpha[i_] + 1/(self.n+1)
                    set_i_ = []
                    for j in range(self.C):
                        if self.score_comparison[i_][j] < (self.n+1)*(1-self.alpha[i_]):
                            set_i_.append(j)
                    prediction_sets[i_] = copy.deepcopy(set_i_)

        self.alpha_2 = copy.deepcopy(self.alpha)
        new_prediction_sets = []
        for i_ in range(self.n_test):
            new_prediction_sets.append(copy.deepcopy(prediction_sets[i_]))        
            while self.alpha_2[i_]<=1 and len(prediction_sets[i_]) == len(new_prediction_sets[i_]):
                #print(alpha[i_])
                #print(prediction_sets[i_])
                #print(grey_box.given_sets_test[i_])
                self.alpha_2[i_] = self.alpha_2[i_] + 1/(self.n+1)
                set_i_ = []
                for j in range(self.C):
                    if self.score_comparison[i_][j] < (self.n+1)*(1-self.alpha_2[i_]):
                        set_i_.append(j)
                new_prediction_sets[i_] = copy.deepcopy(set_i_)

    
                
    def compilation(self,set_type,u,X, Y, X_test,Y_test, simulation = False, data_model = None):
        self.given_sets(set_type,u)
        self.miscoverage()
        #self.sanity_check()
        self.true_miscoverage = []
        ## true probabilites
        if simulation:

            p_test = data_model.compute_prob(X_test)

            ## true miscoverage

            for i_ in range(self.n_test):
                self.true_miscoverage.append(1-sum(p_test[i_][self.given_sets_test[i_]]))


        self.predicted_miscoverage_black_box = []

        for i_ in range(self.n_test):
            self.predicted_miscoverage_black_box.append(1-sum(self.predicted_test_prob[i_][self.given_sets_test[i_]]))
            
        self.true_miscoverage_indicator = 0
        
        for i_ in range(self.n_test):
            if not Y_test[i_] in self.given_sets_test[i_]:
                self.true_miscoverage_indicator += 1
        self.true_miscoverage_indicator /= self.n_test
        train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=0.33, random_state=42)

        mu_train = self.black_box.fit(train_X,train_Y)
        prob_valid = mu_train.predict_proba(valid_X)
        grey_box_valid = calibration.ProbAccum(prob_valid)
        grey_box_valid.given_sets(set_type,u)

        given_sets_valid = grey_box_valid.given_sets_test

        validation_coverage = 0
        for i in range(len(valid_Y)):
            if valid_Y[i] in given_sets_valid[i]:
                validation_coverage+=1   
                
        self.validation_coverage = validation_coverage/len(valid_Y)     

class minmaxJackknifePlus:
    def __init__(self, X, Y, X_test, C, black_box, random_state_training=2020,random_state_test=2021):
        self.black_box = black_box
        self.n = X.shape[0]  ## size of training data
        self.C = C           ## number of classes
        self.n_test=X_test.shape[0]               ## number of test data points
        self.black_box = black_box                ## choice of predictive model
        mu = self.black_box.fit(X,Y)              ## fit the black box model to the entire training data
         
        ## The purpose of fitting black box model on the entire test data is to construct I(X) which will be considered given sets for test data.
        ## The purpose is to provide coverage guarantees on I(X) for test data.
        
        self.predicted_test_prob = mu.predict_proba(X_test)  ## predicted probabiities for classes in test data
        self.grey_box = calibration.ProbAccum(self.predicted_test_prob) ## function provides ranks of classes, order of classes, sorted probabilities and cumulated sorted probabilities
        
        
    ## The functions below can be used to produce different kinds of given sets I(X)
    
    
    def given_sets_fixed_length(self, K):   ## given sets of fixed length K, i.e., K classes with maximum softmax for test data fitted by black box on complete training data.
        self.K = K
        self.given_sets_test = self.grey_box.order[:,:K]
    
    
    def given_sets_min_prob_sum(self,s):  ## given sets with variable set size, but summation of softmax within given sets above given threshold s, the classes with maximum predicted                                                 ## softmax are chosen
        self.given_sets_test = []
        for i_ in range(self.n_test):
            given_set_i_ = [self.grey_box.order[i_][0]]
            j=1
            while self.grey_box.prob_sort_sum[i_][j]<s:
                given_set_i_.append(self.grey_box.order[i_][j])
                j+=1
            self.given_sets_test.append(given_set_i_)
            
    def given_sets_min_prob(self,c):      ## given sets with variable set size, but classes with predicted softmax greater than threshold c.
        self.given_sets_test = []
        for i_ in range(self.n_test):
            given_set_i_ = [self.grey_box.order[i_][0]]
            j=1
            while self.grey_box.prob_sort[i_][j]>=c:
                given_set_i_.append(self.grey_box.order[i_][j])
                j+=1
            self.given_sets_test.append(given_set_i_)
            
            
    def given_sets(self,set_type,u):
        if set_type == "fixed length":
            self.given_sets_fixed_length(u)
        elif set_type == "minimum set probability":
            self.given_sets_min_prob_sum(u)
        elif set_type == "minimum class probability":
            self.given_sets_min_prob(u)
            
    ## The following function is sensitive to jackknife technique, if you work on split conforal or CV+, it will require modification.
    ## For the function i_ associates with test data points, i associates with training data points
    ## The function computes \sum_{i=1}^{n_train}  I(E(X_i,Y_i,..) < E(X_{i_}, j, ..)) for all test data points i_, for all the classes j 
    ## The (n_test, C) matrix is named score_comparison.
    
    
    def Score_comparison_function(self, X, Y, X_test, random_state_training=2020,random_state_test=2021): 
        rng = np.random.default_rng(random_state_training)
        epsilon_training = rng.uniform(low=0.0, high=1.0, size=self.n)
        rng = np.random.default_rng(random_state_test)
        epsilon_test = rng.uniform(low=0.0, high=1.0, size=self.n_test)
        
        # loop over data points in training data
        self.n_train = 0
        self.score_test_min = np.ones((self.n_test,self.C))
        self.score_training = []
        self.score_comparison = np.zeros((self.n_test,self.C))

        for i in range(self.n):
            if len(np.unique(np.delete(Y,i))) == self.C:
                
                self.n_train += 1
                # fit the black box model leaving one one
                mu_LOO = self.black_box.fit(np.delete(X,i,0),np.delete(Y,i))
                grey_box_training_i = calibration.ProbAccum(mu_LOO.predict_proba(X[i]))
                score_i = grey_box_training_i.calibrate_scores(Y[i], epsilon_training[i])
                self.score_training.append(score_i)
                grey_box_test_i = calibration.ProbAccum(mu_LOO.predict_proba(X_test))
                grey_box_test_i.all_scores(epsilon = epsilon_test)
                #print(self.grey_box_test_i.normalised_scores.shape)
                ## loop over test_data
                for i_ in range(self.n_test):
                    scores_given_test_i_ = grey_box_test_i.normalised_scores[i_]
               
                    for j in np.arange(self.C):
                        self.score_test_min[i_][j] = min(self.score_test_min[i_][j], scores_given_test_i_[j])
        for i_ in range(self.n_test):
            for i in range(self.n):
                for j in range(self.C):
                    if self.score_training[i] < self.score_test_min[i_][j]:
                        self.score_comparison[i_][j] += 1
                        
                    
                        
                    
                
    ## the function below provides miscoverage rate for each test data point, we could take an average to get a scalar value
    ## For further details, please see the related paper, 
    
    def miscoverage(self):
        self.results_lower = []
        self.results_upper = []
        for i_ in range(self.n_test):
            self.results_lower.append(max(self.score_comparison[i_][self.given_sets_test[i_]]))
            self.results_upper.append(min(self.score_comparison[i_][np.setdiff1d(np.arange(self.C),self.given_sets_test[i_])]))
        self.alpha = 1 - np.array(self.results_lower)/(self.n_train + 1)
        
    ## I have written the function below for sanity check purpose, i.e., to crosscheck if CI(alpha, x) is subset of I(X)

    def conformal_prediction_sets(self,alpha):
        prediction_sets = []
        for i_ in range(self.n_test):
            set_i_ = []
            for j in range(self.C):
                if self.score_comparison[i_][j] < (self.n+1)*(1-alpha[i_]):
                    set_i_.append(j)
            prediction_sets.append(set_i_)
        return(prediction_sets)
    
    ## This is sanity check for alpha
    

    def sanity_check(self,alpha):
        prediction_sets_alpha = self.conformal_prediction_sets(alpha)
        
        unvalid_intervals_alpha = []
        
        for i_ in range(self.n_test):
            if not set(prediction_sets_alpha[i_]).issubset(set(self.given_sets_test[i_])):
                unvalid_intervals_alpha.append(i_)
        return(unvalid_intervals_alpha)
                    
    def calibrate_alpha(self):
        prediction_sets = self.conformal_prediction_sets(self.alpha)
        
        for i_ in range(self.n_test):
            if not set(prediction_sets[i_]).issubset(set(self.given_sets_test[i_])):
                while not set(prediction_sets[i_]).issubset(set(self.given_sets_test[i_])):
                    self.alpha[i_] = self.alpha[i_] + 1/(self.n+1)
                    set_i_ = []
                    for j in range(self.C):
                        if self.score_comparison[i_][j] < (self.n+1)*(1-self.alpha[i_]):
                            set_i_.append(j)
                    prediction_sets[i_] = copy.deepcopy(set_i_)

        self.alpha_2 = copy.deepcopy(self.alpha)
        new_prediction_sets = []
        for i_ in range(self.n_test):
            new_prediction_sets.append(copy.deepcopy(prediction_sets[i_]))        
            while self.alpha_2[i_]<=1 and len(prediction_sets[i_]) == len(new_prediction_sets[i_]):
                #print(alpha[i_])
                #print(prediction_sets[i_])
                #print(grey_box.given_sets_test[i_])
                self.alpha_2[i_] = self.alpha_2[i_] + 1/(self.n+1)
                set_i_ = []
                for j in range(self.C):
                    if self.score_comparison[i_][j] < (self.n+1)*(1-self.alpha_2[i_]):
                        set_i_.append(j)
                new_prediction_sets[i_] = copy.deepcopy(set_i_)

                
                
    def compilation(self,set_type,u,X, Y, X_test,Y_test, simulation = False, data_model = None):
        self.given_sets(set_type,u)
        self.miscoverage()
        #self.sanity_check()
        self.true_miscoverage = []
        ## true probabilites
        if simulation:

            p_test = data_model.compute_prob(X_test)

            ## true miscoverage

            for i_ in range(self.n_test):
                self.true_miscoverage.append(1-sum(p_test[i_][self.given_sets_test[i_]]))


        self.predicted_miscoverage_black_box = []

        for i_ in range(self.n_test):
            self.predicted_miscoverage_black_box.append(1-sum(self.predicted_test_prob[i_][self.given_sets_test[i_]]))
            
        self.true_miscoverage_indicator = 0
        
        for i_ in range(self.n_test):
            if not Y_test[i_] in self.given_sets_test[i_]:
                self.true_miscoverage_indicator += 1
        self.true_miscoverage_indicator /= self.n_test
       
                


        
        train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=0.33, random_state=42)

        mu_train = self.black_box.fit(train_X,train_Y)
        prob_valid = mu_train.predict_proba(valid_X)
        grey_box_valid = calibration.ProbAccum(prob_valid)
        grey_box_valid.given_sets(set_type,u)

        given_sets_valid = grey_box_valid.given_sets_test

        validation_coverage = 0
        for i in range(len(valid_Y)):
            if valid_Y[i] in given_sets_valid[i]:
                validation_coverage+=1   
                
        self.validation_coverage = validation_coverage/len(valid_Y)     
        
        
class SplitConformal:
    def __init__(self, X, Y, X_test, C, black_box, random_state_training=2020,random_state_test=2021):
        self.black_box = black_box
        self.n = X.shape[0]  ## size of training data
        self.C = C           ## number of classes
        self.n_test=X_test.shape[0]               ## number of test data points
        self.black_box = black_box                ## choice of predictive model
        mu = self.black_box.fit(X,Y)              ## fit the black box model to the entire training data
         
        ## The purpose of fitting black box model on the entire test data is to construct I(X) which will be considered given sets for test data.
        ## The purpose is to provide coverage guarantees on I(X) for test data.
        
        self.predicted_test_prob = mu.predict_proba(X_test)  ## predicted probabiities for classes in test data
        self.grey_box = calibration.ProbAccum(self.predicted_test_prob) ## function provides ranks of classes, order of classes, sorted probabilities and cumulated sorted probabilities
        
        
    ## The functions below can be used to produce different kinds of given sets I(X)
    
    
    def given_sets_fixed_length(self, K):   ## given sets of fixed length K, i.e., K classes with maximum softmax for test data fitted by black box on complete training data.
        self.K = K
        self.given_sets_test = self.grey_box.order[:,:K]
    
    
    def given_sets_min_prob_sum(self,s):  ## given sets with variable set size, but summation of softmax within given sets above given threshold s, the classes with maximum predicted                                                 ## softmax are chosen
        self.given_sets_test = []
        for i_ in range(self.n_test):
            given_set_i_ = [self.grey_box.order[i_][0]]
            j=1
            while self.grey_box.prob_sort_sum[i_][j]<s:
                given_set_i_.append(self.grey_box.order[i_][j])
                j+=1
            self.given_sets_test.append(given_set_i_)
            
    def given_sets_min_prob(self,c):      ## given sets with variable set size, but classes with predicted softmax greater than threshold c.
        self.given_sets_test = []
        for i_ in range(self.n_test):
            given_set_i_ = [self.grey_box.order[i_][0]]
            j=1
            while self.grey_box.prob_sort[i_][j]>=c:
                given_set_i_.append(self.grey_box.order[i_][j])
                j+=1
            self.given_sets_test.append(given_set_i_)
            
            
    def given_sets(self,set_type,u):
        if set_type == "fixed length":
            self.given_sets_fixed_length(u)
        elif set_type == "minimum set probability":
            self.given_sets_min_prob_sum(u)
        elif set_type == "minimum class probability":
            self.given_sets_min_prob(u)
            
    ## The following function is sensitive to jackknife technique, if you work on split conforal or CV+, it will require modification.
    ## For the function i_ associates with test data points, i associates with training data points
    ## The function computes \sum_{i=1}^{n_train}  I(E(X_i,Y_i,..) < E(X_{i_}, j, ..)) for all test data points i_, for all the classes j 
    ## The (n_test, C) matrix is named score_comparison.
    
    
    def Score_comparison_function(self, train_X, train_Y, valid_X, valid_Y, X_test, random_state_training=2020,random_state_test=2021): 
        rng = np.random.default_rng(random_state_training)
        epsilon_valid = rng.uniform(low=0.0, high=1.0, size = len(valid_Y))
        rng = np.random.default_rng(random_state_test)
        epsilon_test = rng.uniform(low=0.0, high=1.0, size=self.n_test)
        self.n_valid = len(valid_Y)
        
        
        # fit the black box model leaving one one
        mu_train = self.black_box.fit(train_X,train_Y)
        grey_box_calib = calibration.ProbAccum(mu_train.predict_proba(valid_X))
        scores_calib = grey_box_calib.calibrate_scores(valid_Y, epsilon_valid)
        grey_box_test = calibration.ProbAccum(mu_train.predict_proba(X_test))
        grey_box_test.all_scores(epsilon = epsilon_test)
        self.score_comparison = np.zeros((self.n_test,self.C))
        #print(self.grey_box_test_i.normalised_scores.shape)
        ## loop over test_data
        for i_ in range(self.n_test):
            scores_given_test_i_ = grey_box_test.normalised_scores[i_]
            #print(scores_given_test_i_)
            for j in np.arange(self.C):
                for i in range(len(valid_Y)):
                    score_i = scores_calib[i]
                    if score_i < scores_given_test_i_[j]:
                        self.score_comparison[i_][j] += 1
            #print(self.alpha_lower[0][0])
                
    ## the function below provides miscoverage rate for each test data point, we could take an average to get a scalar value
    ## For further details, please see the related paper, 
    
    def miscoverage(self):
        self.results_lower = []
        self.results_upper = []
        for i_ in range(self.n_test):
            self.results_lower.append(max(self.score_comparison[i_][self.given_sets_test[i_]]))
            self.results_upper.append(min(self.score_comparison[i_][np.setdiff1d(np.arange(self.C),self.given_sets_test[i_])]))
        self.alpha = 1 - np.array(self.results_lower)/(self.n_valid + 1)
         
    ## I have written the function below for sanity check purpose, i.e., to crosscheck if CI(alpha, x) is subset of I(X)

    def conformal_prediction_sets(self,alpha):
        prediction_sets = []
        for i_ in range(self.n_test):
            set_i_ = []
            for j in range(self.C):
                if self.score_comparison[i_][j] < (self.n_valid+1)*(1-alpha[i_]):
                    set_i_.append(j)
            prediction_sets.append(set_i_)
        return(prediction_sets)
    
    ## This is sanity check for alpha
    

    def sanity_check(self,alpha):
        prediction_sets_alpha = self.conformal_prediction_sets(alpha)
        
        unvalid_intervals_alpha = []
        
        for i_ in range(self.n_test):
            if not set(prediction_sets_alpha[i_]).issubset(set(self.given_sets_test[i_])):
                unvalid_intervals_alpha.append(i_)
        return(unvalid_intervals_alpha)
                
    def calibrate_alpha(self):
        prediction_sets = self.conformal_prediction_sets(self.alpha)
        
        for i_ in range(self.n_test):
            if not set(prediction_sets[i_]).issubset(set(self.given_sets_test[i_])):
                while not set(prediction_sets[i_]).issubset(set(self.given_sets_test[i_])):
                    self.alpha[i_] = self.alpha[i_] + 1/(self.n_valid+1)
                    set_i_ = []
                    for j in range(self.C):
                        if self.score_comparison[i_][j] < (self.n_valid+1)*(1-self.alpha[i_]):
                            set_i_.append(j)
                    prediction_sets[i_] = copy.deepcopy(set_i_)

        self.alpha_2 = copy.deepcopy(self.alpha)
        new_prediction_sets = []
        for i_ in range(self.n_test):
            new_prediction_sets.append(copy.deepcopy(prediction_sets[i_]))        
            while self.alpha_2[i_]<=1 and len(prediction_sets[i_]) == len(new_prediction_sets[i_]):
                #print(alpha[i_])
                #print(prediction_sets[i_])
                #print(grey_box.given_sets_test[i_])
                self.alpha_2[i_] = self.alpha_2[i_] + 1/(self.n_valid+1)
                set_i_ = []
                for j in range(self.C):
                    if self.score_comparison[i_][j] < (self.n_valid+1)*(1-self.alpha_2[i_]):
                        set_i_.append(j)
                new_prediction_sets[i_] = copy.deepcopy(set_i_)
                
    def compilation(self,set_type,u,X, Y, X_test,Y_test, simulation = False, data_model = None):
        self.given_sets(set_type,u)
        self.miscoverage()
        #self.sanity_check()
        self.true_miscoverage = []
        ## true probabilites
        if simulation:

            p_test = data_model.compute_prob(X_test)

            ## true miscoverage

            for i_ in range(self.n_test):
                self.true_miscoverage.append(1-sum(p_test[i_][self.given_sets_test[i_]]))


        self.predicted_miscoverage_black_box = []

        for i_ in range(self.n_test):
            self.predicted_miscoverage_black_box.append(1-sum(self.predicted_test_prob[i_][self.given_sets_test[i_]]))

        self.true_miscoverage_indicator = 0
        
        for i_ in range(self.n_test):
            if not Y_test[i_] in self.given_sets_test[i_]:
                self.true_miscoverage_indicator += 1
        self.true_miscoverage_indicator /= self.n_test

        
        train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=0.33, random_state=42)

        mu_train = self.black_box.fit(train_X,train_Y)
        prob_valid = mu_train.predict_proba(valid_X)
        grey_box_valid = calibration.ProbAccum(prob_valid)
        grey_box_valid.given_sets(set_type,u)

        given_sets_valid = grey_box_valid.given_sets_test

        validation_coverage = 0
        for i in range(len(valid_Y)):
            if valid_Y[i] in given_sets_valid[i]:
                validation_coverage+=1   
                
        self.validation_coverage = validation_coverage/len(valid_Y)    
