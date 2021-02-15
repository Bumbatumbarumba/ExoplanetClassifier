# -*- coding: utf-8 -*-
import utils
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing



class LogReg:
    def __init__(self):
        self.lr = LogisticRegression(max_iter=1000, class_weight='balanced')#attempt to balance the classes
        self.scaler = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.y_hat = None
        self.x = None
        self.y = None
        
        
    '''
    ppd is a preprocess_data instance
    '''
    def fit(self, ppd):
        self.x, self.y = utils.load_dataset(ppd.all_data)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, 
                                                                                self.y, 
                                                                                test_size=0.2, 
                                                                                random_state=0)
        self.x_train, self.x_test = utils.fill_na_mean(ppd.target_columns, self.x_train, self.x_test)
        
        self.scaler = preprocessing.StandardScaler().fit(self.x_train)
        scaled_x_train = self.scaler.transform(self.x_train)
        self.lr.fit(scaled_x_train, self.y_train)


    '''
    use the test set to make predictions on the model
    '''
    def predict(self):
        scaled_x_test = self.scaler.transform(self.x_test)
        self.y_hat = self.lr.predict(scaled_x_test)
    
    
    '''
    save predictions and true labels for comparisons
    '''
    def save_txt(self, exp_name):
        pred_loc = exp_name + "lr_labels_pred.txt"
        true_loc = exp_name + "lr_labels_true.txt"
        
        np.savetxt(pred_loc, self.y_hat)
        np.savetxt(true_loc, self.y_test)
        print('... saved predicted labels to ' + pred_loc)
        print('... saved real labels to ' + true_loc)
        