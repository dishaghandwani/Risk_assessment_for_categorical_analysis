# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from sklearn import svm
from sklearn import ensemble
from sklearn import calibration
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import copy
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

class CNN:
    def __init__(self, num_classes, num_epochs=10, batch_size=64, learning_rate=0.001):
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    def build_model(self):
        # Define a simple CNN architecture
        model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )
        return model

    def fit(self, X, y):
        train_data = self.transform(X)
        train_labels = torch.tensor(y, dtype=torch.long)
        train_dataset = TensorDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

        print("Finished Training")

    def predict(self, X):
        test_data = self.transform(X)
        test_data = test_data.unsqueeze(1).to(self.device)
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(test_data)
        _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def predict_proba(self, X):
        test_data = self.transform(X)
        test_data = test_data.unsqueeze(1).to(self.device)
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(test_data)
        softmax = nn.Softmax(dim=1)
        probabilities = softmax(outputs)
        return probabilities.cpu().numpy()
    
class Oracle:
    def __init__(self, model):
        self.model = model
    
    def fit(self,X,y):
        return self

    def predict(self, X):
        return self.model.sample(X)        

    def predict_proba(self, X):
        if(len(X.shape)==1):
            X = X.reshape((1,X.shape[0]))
        prob = self.model.compute_prob(X)
        prob = np.clip(prob, 1e-6, 1.0)
        prob = prob / prob.sum(axis=1)[:,None]
        return prob


class SimpleLogisticRegressionModel:
    def __init__(self, penalty='l2', C=1.0, random_state=2020):
        self.model = LogisticRegression(penalty=penalty, C=C, random_state=random_state, max_iter=5000)
        
    def fit(self, X, y):
        self.model_fit = self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model_fit.predict(X)

    def predict_proba(self, X):
        if len(X.shape) == 1:
            X = X.reshape((1, X.shape[0]))
        prob = self.model_fit.predict_proba(X)
        return prob


class SimpleNaiveBayesClassifier:
    def __init__(self):
        self.model = MultinomialNB()
        
    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    
class SVC:
    def __init__(self, calibrate=False,
                 kernel = 'linear',
                 C = 1,
                 clip_proba_factor = 0.1,
                 random_state = 2020):
        self.model = svm.SVC(kernel = kernel,
                             C = C,
                             probability = True,
                             random_state = random_state)
        self.calibrate = calibrate
        self.num_classes = 0
        self.factor = clip_proba_factor
        
    def fit(self, X, y):
        self.num_classes = len(np.unique(y)) 
        self.model_fit = self.model.fit(X, y)
        if self.calibrate:
            self.calibrated = calibration.CalibratedClassifierCV(self.model_fit,
                                                                 method='sigmoid',
                                                                 cv=10)
        else:
            self.calibrated = None
        return copy.deepcopy(self)

    def predict(self, X):
        return self.model_fit.predict(X)

    def predict_proba(self, X):        
        if(len(X.shape)==1):
            X = X.reshape((1,X.shape[0]))
        if self.calibrated is None:
            prob = self.model_fit.predict_proba(X)
        else:
            prob = self.calibrated.predict_proba(X)
        prob = np.clip(prob, self.factor/self.num_classes, 1.0)
        prob = prob / prob.sum(axis=1)[:,None]
        return prob

class RFC:
    def __init__(self, calibrate=False,
                 n_estimators = 1000,
                 criterion="gini", 
                 max_depth=None,
                 max_features="auto",
                 min_samples_leaf=1,
                 clip_proba_factor=0.1,
                 random_state = 2020):
        
        self.model = ensemble.RandomForestClassifier(n_estimators=n_estimators,
                                                     criterion=criterion,
                                                     max_depth=max_depth,
                                                     max_features=max_features,
                                                     min_samples_leaf=min_samples_leaf,
                                                     random_state = random_state)
        self.calibrate = calibrate
        self.num_classes = 0
        self.factor = clip_proba_factor
        
    def fit(self, X, y):
        self.num_classes = len(np.unique(y)) 
        self.model_fit = self.model.fit(X, y)
        if self.calibrate:
            self.calibrated = calibration.CalibratedClassifierCV(self.model_fit,
                                                                 method='sigmoid',
                                                                 cv=10)
        else:
            self.calibrated = None
        return copy.deepcopy(self)

    def predict(self, X):
        return self.model_fit.predict(X)

    def predict_proba(self, X):        
        if(len(X.shape)==1):
            X = X.reshape((1,X.shape[0]))
        if self.calibrated is None:
            prob = self.model_fit.predict_proba(X)
        else:
            prob = self.calibrated.predict_proba(X)
        prob = np.clip(prob, self.factor/self.num_classes, 1.0)
        prob = prob / prob.sum(axis=1)[:,None]
        return prob

class NNet:
    def __init__(self, calibrate=False,
                 hidden_layer_sizes = 64,
                 batch_size = 128,
                 learning_rate_init = 0.01,
                 max_iter = 20,
                 clip_proba_factor = 0.1,
                 random_state = 2020):
        
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                                   batch_size=batch_size,
                                   learning_rate_init=learning_rate_init,
                                   max_iter=max_iter,
                                   random_state=random_state)
        self.calibrate = calibrate
        self.num_classes = 0
        self.factor = clip_proba_factor
        
    def fit(self, X, y):
        self.num_classes = len(np.unique(y)) 
        self.model_fit = self.model.fit(X, y)
        if self.calibrate:
            self.calibrated = calibration.CalibratedClassifierCV(self.model_fit,
                                                                 method='sigmoid',
                                                                 cv=10)
        else:
            self.calibrated = None
        return copy.deepcopy(self)

    def predict(self, X):
        return self.model_fit.predict(X)

    def predict_proba(self, X):        
        if(len(X.shape)==1):
            X = X.reshape((1,X.shape[0]))
        if self.calibrated is None:
            prob = self.model_fit.predict_proba(X)
        else:
            prob = self.calibrated.predict_proba(X)
        prob = np.clip(prob, self.factor/self.num_classes, 1.0)
        prob = prob / prob.sum(axis=1)[:,None]
        return prob


class KNNClassifier:
    def __init__(self, calibrate=False, n_neighbors=3, clip_proba_factor=0.1, random_state=2020):
        self.n_neighbors = n_neighbors
        self.calibrate = calibrate
        self.clip_proba_factor = clip_proba_factor
        self.random_state = random_state

        # Create the K-NN classifier
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

        # Create a calibrated classifier if calibration is enabled
        if self.calibrate:
            self.calibrated = CalibratedClassifierCV(self.model, method='sigmoid', cv=10)
        else:
            self.calibrated = None

    def fit(self, X, y):
        # Fit the K-NN model to the data
        self.model.fit(X, y)

        # Fit the calibrated model if calibration is enabled
        if self.calibrate:
            self.calibrated.fit(X, y)

        return copy.deepcopy(self)

    def predict(self, X):
        # Predict class labels using the K-NN model
        return self.model.predict(X)

    def predict_proba(self, X):
        # Predict class probabilities using the K-NN model
        prob = self.model.predict_proba(X)

        # Clip probabilities to ensure they don't fall below a certain threshold
        prob = np.clip(prob, self.clip_proba_factor / self.n_neighbors, 1.0)

        # Normalize probabilities to sum to 1 for each input sample
        prob = prob / prob.sum(axis=1)[:, None]

        return prob




