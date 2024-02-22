#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 00:01:29 2021

@author: guillermogarcia
"""
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from scipy.spatial import distance
from collections import defaultdict
from pylab import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import joblib
from os import path

########## ANALYSE ##########

def describe_data(training_data,num_bool = False):
    #Values of object features
    for col in training_data:
        if training_data[col].dtypes == 'object':
            print(col,training_data[col].unique())
    
    #Values of non object features
    if num_bool == True:
        for col in training_data:
            if training_data[col].dtypes != 'object':
                print(col,training_data[col].unique())    
            
    #Hist object features        
    for col in training_data:
        if training_data[col].dtypes == 'object':
            training_data[col].value_counts().plot(kind='bar')
            plt.title(col)
            plt.show()
    #Hist non object features
    for col in training_data:
        if training_data[col].dtypes != 'object':
            training_data[col].plot.hist(bins=30, alpha=0.5)
            plt.title(col)
            plt.show()
    return


######## MISSING VALUES #######

def create_nan_class(df,columns):
    '''
    Create a new class with name 'nan' in the columns given
    '''
    
    for col in columns:
        df[col] = df[col].apply(lambda x: x if not pd.isnull(x) else 'nan')
        
    return df


######### CATEGORICAL TO NUMERICAL ########

def bincat2num(df,columns):
    '''
    Get dummies for binary classes.
    With drop first so that the binary features has only 0 and 1's
    '''
    return pd.get_dummies(df, columns = columns, drop_first=True)

def multicat2num(df,columns):
    '''
    Get dummies for multi class features.
    With no drop first so that we can check if all categorical features are important
    '''
    return pd.get_dummies(df, columns = columns)



######### PREDICT REST OF MISSING VALUES ########

def train_missing_values(df,features_predict,features_train):
    '''
    Train a model to predict missing values.
    Imput missing values in train set.
    '''
    for feat_predict in features_predict:
        
        print(f'Computing feature {feat_predict}')
        #Where are the missing values?
        idx = np.where((df[feat_predict].isnull()))
        df_nulls = df.loc[idx]
        df_without_nulls = df[df[feat_predict].notnull()]
        
        
        #Train with no-missing values 
        y_train = df_without_nulls[feat_predict]
        X_train = df_without_nulls[features_train]
        
        #Predict the missing values
        X_test = df_nulls[features_train]
        
        
        #Regression
        if df[feat_predict].dtype != 'object':
            model = xgb.XGBRegressor()
        #Classification
        else: model = xgb.XGBClassifier()
            
        model.fit(X_train, y_train)
        model.save_model(f"models_missing_values/model_sklearn_{feat_predict}.json")
        y_pred = model.predict(X_test)  

        #y_pred = np.round(y_pred)
        df_nulls[feat_predict] = y_pred
        
        df = pd.concat([df_nulls, df_without_nulls], axis=0)
        df = df.sort_index()
        #print(df.isnull().sum())
        
    print('Done!')
    return df


####### SAME FEATURE UTILS ########

def hamming_dist(x,y):
    return distance.hamming(x, y) * len(x)


########### CATEGORICAL TO NUMERICAL ########

def target_count_per_feature(df_train,feature,target_class = 'classlabel_yes.'):

    '''
    Compute how many labels of the target are in each class of the categorical variable
    '''
    #Percentage labels in all total examples
    val_count_total = df_train[target_class].value_counts()
    per_total = (val_count_total[0] /(val_count_total[0] + val_count_total[1]))*100
    print('Percentage total of label 0:',per_total)
    
    for var in df_train[feature].unique():
        val_count = df_train[df_train[feature] == var][target_class].value_counts()
        
        print('----------------------')
        
        #If only one variable we cannot compute relation of the 2 labels
        if len(val_count) == 2:
            #Percentage per class
            per = (val_count[0]  /(val_count[0] + val_count[1]))*100
            desviation = per - per_total
            print('Percentage of label 0 in category: ',per)
            print('Desviation from mean of dataset:',desviation)
            
        print('Counts target labels of ',var,'--->',val_count)
        
        

def make_binary_features_from_categorical(df_train,feature,feat_classes):
    '''
    Select the category class that I want to encode as binary variables
    '''
    # Compute binary features
    df_tmp = multicat2num(df = df_train,columns = feature)
    
    # Add the ones that we want
    for new_feat in feat_classes:
        df_train[new_feat] = df_tmp[new_feat]
      
    return df_train
    


def encode_as_ordered(df_train,feature):
    '''
    Encode categorical features as integers ordered according to 
    the number of ocurrences
    '''
    dict_order = defaultdict()
    counts = df_train[feature].value_counts()
    
    for i,var in enumerate(counts.keys()):
        dict_order[var] = i
    
    
    df_train[feature] = df_train[feature].map(dict_order)
    
    return df_train


############# FEATURE ENGINEERING IN CONTINUOUS VARIABLES ############

def represent_boxplot(df_train,columns):
    
    for col in columns:
        plt.subplots(figsize = (10,5))
        bp_dict = boxplot(df_train[col], vert=False)
        print(col)
        for line in bp_dict['medians']:

            # get position data for median line
            x, y = line.get_xydata()[1] # top of median line
            # overlay median value
            text(x, y, '%.1f' % x,
                 horizontalalignment='center') # draw above, centered

        for line in bp_dict['boxes']:
            x, y = line.get_xydata()[0] # bottom of left line
            text(x,y, '%.1f' % x,
                 horizontalalignment='center', # centered
                 verticalalignment='top')      # below
            x, y = line.get_xydata()[3] # bottom of right line
            text(x,y, '%.1f' % x,
                 horizontalalignment='center', # centered
                     verticalalignment='top')      # below

        show()
    return boxplot


def cap_outliers(df_train,dict_features):
    '''
    Cap outliers from one value ownwards.
    '''
    for key, val in dict_features.items():
        df_train[key] = df_train[key].apply(lambda x: val if x >= val else x)
    
    return df_train


class Feature_transform:
    def __init__(self,df_train,columns):
        self.df_train = df_train
        self.columns = columns
        self.scaler = StandardScaler()
        self.min_max = MinMaxScaler()
        
    def numpy2pandas(self,df,df_transformed):
        for enum, col in enumerate(self.columns):
            df[col] = df_transformed[:,enum]
        return df
    
    def min_max_fit(self):
        self.min_max.fit(self.df_train[self.columns])
        joblib.dump(self.min_max, 'models_feat_trans/min_max.gz')
        return
    
    def min_max_transform(self,df):
        if path.exists('models_feat_trans/min_max.gz'):
            min_max_loaded = joblib.load('models_feat_trans/min_max.gz')
            df_transformed = min_max_loaded.transform(df[self.columns])
        else: print('not fitted model!')
        return self.numpy2pandas(df,df_transformed)       
    
    
    def standarize_fit(self):
        self.scaler.fit(self.df_train[self.columns])
        joblib.dump(self.scaler, 'models_feat_trans/scaler.gz')
        return
    
    def standatize_transform(self,df):
        if path.exists(f'models_feat_trans/scaler.gz'):
            scaler_loaded = joblib.load('models_feat_trans/scaler.gz')
            df_transformed = scaler_loaded.transform(df[self.columns])
        else: print('not fitted model!')
        
        return self.numpy2pandas(df,df_transformed)







