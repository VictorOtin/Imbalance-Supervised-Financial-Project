#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 15:41:08 2021

@author: guillermogarcia
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection._split import KFold
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm1
import matplotlib.pylab as plt


#code snippet 6.2 implementation of an ensembke MDI method
def featImpMDI(fit, featNames):
    #feat importance based on IS mean impurity reduction
    df0 = {i:tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = featNames
    df0 = df0.replace(0, np.nan) #because max_features=1
    imp = pd.concat({'mean':df0.mean(), 'std':df0.std()*df0.shape[0]**-.5}, axis=1) #CLT
    imp /= imp['mean'].sum()
    return imp
    
#code snippet 6.3 implementation of MDA - MDI with cross validation
def featImpMDA(clf, X, y, n_splits=10):
    #feat importance based on OOS score reduction
    cvGen = KFold(n_splits=n_splits)
    scr0, scr1 = pd.Series(dtype='float64'), pd.DataFrame(columns=X.columns)
    for i, (train, test) in enumerate(cvGen.split(X=X)):
        x0, y0 = X.iloc[train, :], y.iloc[train]
        x1, y1 = X.iloc[test,:], y.iloc[test]
        fit = clf.fit(X=x0, y=y0) # the fit occures
        prob= fit.predict_proba(x1) #prediction before shuffles
        scr0.loc[i]=-log_loss(y1, prob, labels=clf.classes_)
        for j in X.columns:
            X1_ = x1.copy(deep=True)
            np.random.shuffle(X1_[j].values) #shuffle one columns
            prob = fit.predict_proba(X1_) #prediction after shuffle
            scr1.loc[i,j] = -log_loss(y1, prob, labels=clf.classes_)
    imp=(-1*scr1).add(scr0, axis=0)
    imp = imp/(-1*scr1)
    imp=pd.concat({'mean':imp.mean(), 'std':imp.std()*imp.shape[0]**-.5}, axis=1) #CLT
    return imp

#code snippet 6.2 implementation of an ensembke MDI method
def featImpMDI(fit, featNames):
    #feat importance based on IS mean impurity reduction
    df0 = {i:tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = featNames
    df0 = df0.replace(0, np.nan) #because max_features=1
    imp = pd.concat({'mean':df0.mean(), 'std':df0.std()*df0.shape[0]**-.5}, axis=1) #CLT
    imp /= imp['mean'].sum()
    return imp
    
#code snippet 6.3 implementation of MDA - MDI with cross validation
def featImpMDA(clf, X, y, n_splits=10):
    #feat importance based on OOS score reduction
    cvGen = KFold(n_splits=n_splits)
    scr0, scr1 = pd.Series(dtype='float64'), pd.DataFrame(columns=X.columns)
    for i, (train, test) in enumerate(cvGen.split(X=X)):
        x0, y0 = X.iloc[train, :], y.iloc[train]
        x1, y1 = X.iloc[test,:], y.iloc[test]
        fit = clf.fit(X=x0, y=y0) # the fit occures
        prob= fit.predict_proba(x1) #prediction before shuffles
        scr0.loc[i]=-log_loss(y1, prob, labels=clf.classes_)
        for j in X.columns:
            X1_ = x1.copy(deep=True)
            np.random.shuffle(X1_[j].values) #shuffle one columns
            prob = fit.predict_proba(X1_) #prediction after shuffle
            scr1.loc[i,j] = -log_loss(y1, prob, labels=clf.classes_)
    imp=(-1*scr1).add(scr0, axis=0)
    imp = imp/(-1*scr1)
    imp=pd.concat({'mean':imp.mean(), 'std':imp.std()*imp.shape[0]**-.5}, axis=1) #CLT
    return imp

#code snippet 6.4 - clustered MDI
def groupMeanStd(df0, clstrs):
    out = pd.DataFrame(columns=['mean', 'std'])
    for i, j in clstrs.items():
        df1 = df0[j].sum(axis=1)
        out.loc['C_'+str(i), 'mean'] = df1.mean()
        out.loc['C_'+str(i), 'std'] = df1.std() * df1.shape[0]**-.5
    return out

def featImpMDI_Clustered(fit, featNames, clstrs):
    df0 = {i:tree.feature_importances_ for i, tree in enumerate(fit.estimators_)}
    df0 = pd.DataFrame.from_dict(df0, orient='index')
    df0.columns = featNames
    df0 = df0.replace(0, np.nan) #because max_features=1
    imp = groupMeanStd(df0, clstrs)
    imp /= imp['mean'].sum()
    return imp
    
#code snippet 6.5 - clustered MDA    
def featImpMDA_Clustered(clf, X, y, clstrs, n_splits=10):
    cvGen = KFold(n_splits=n_splits)
    scr0, scr1 = pd.Series(dtype='float64'), pd.DataFrame(columns=clstrs.keys())
    for i, (train, test) in enumerate(cvGen.split(X=X)):
        X0, y0, = X.iloc[train,:], y.iloc[train] 
        X1, y1 = X.iloc[test, :], y.iloc[test]
        fit = clf.fit(X=X0, y=y0)
        prob=fit.predict_proba(X1)
        scr0.loc[i] = -log_loss(y1, prob, labels=clf.classes_)
        for j in scr1.columns:
            X1_=X1.copy(deep=True)
            for k in clstrs[j]:
                np.random.shuffle(X1_[k].values) # shuffle clusters
            prob=fit.predict_proba(X1_)
            scr1.loc[i,j]=-log_loss(y1, prob, labels=clf.classes_)
        imp=(-1*scr1).add(scr0,axis=0)
        imp = imp/(-1*scr1)
        imp = pd.concat({'mean':imp.mean(), 'std':imp.std()*imp.shape[0]**-.5}, axis=1)
        imp.index=['C_'+str(i) for i in imp.index]
    return imp


def normal_feat_imp(X,y):
    
    clf = DecisionTreeClassifier(criterion='entropy', 
                                 max_features=1, 
                                 class_weight='balanced', 
                                 min_weight_fraction_leaf=0)
                                 
    clf = BaggingClassifier(base_estimator=clf, 
                          n_estimators=1000, 
                          max_features=1., 
                          max_samples=1., 
                          oob_score=False)
    fit = clf.fit(X,y)
    imp = featImpMDI(fit, featNames=X.columns)

    #print the graph Example 6.2 Example of MDI results
    imp.sort_values('mean', inplace=True)
    plt.figure(figsize=(10, imp.shape[0] / 5))
    imp['mean'].plot(kind='barh', color='b', alpha=0.25, xerr=imp['std'], error_kw={'ecolor': 'r'})
    plt.title('MDI results')
    plt.show()


    clf = DecisionTreeClassifier(criterion='entropy', 
                                 max_features=1, 
                                 class_weight='balanced', 
                                 min_weight_fraction_leaf=0)

    clf = BaggingClassifier(base_estimator=clf, 
                          n_estimators=1000, 
                          max_features=1., 
                          max_samples=1., 
                          oob_score=False)
    fit = clf.fit(X,y)
    imp = featImpMDA(clf, X, y, 10)

    imp.sort_values('mean', inplace=True)
    plt.figure(figsize=(10, imp.shape[0] / 5))
    imp['mean'].plot(kind='barh', color='b', alpha=0.25, xerr=imp['std'], error_kw={'ecolor': 'r'})
    plt.title('MDA results')
    plt.show()



def clustered_MDI_MDA(X,y,clstrs):
    
    
    clf = DecisionTreeClassifier(criterion='entropy', 
                                     max_features=1, 
                                     class_weight='balanced', 
                                     min_weight_fraction_leaf=0)

    clf = BaggingClassifier(base_estimator=clf, 
                          n_estimators=1000, 
                          max_features=1., 
                          max_samples=1., 
                          oob_score=False)
    fit = clf.fit(X,y)
    imp = featImpMDI_Clustered(fit, X.columns, clstrs)

    imp.sort_values('mean', inplace=True)
    plt.figure(figsize=(10, 5))
    imp['mean'].plot(kind='barh', color='b', alpha=0.25, xerr=imp['std'], error_kw={'ecolor': 'r'})
    plt.title('   Clustered MDI')
    plt.show()

    #code snippet 6.8 - calling the functions for clustered MDA
    clf = DecisionTreeClassifier(criterion='entropy', 
                                  max_features=1, 
                                  class_weight='balanced', 
                                  min_weight_fraction_leaf=0)

    clf = BaggingClassifier(base_estimator=clf, 
                          n_estimators=1000, 
                          max_features=1., 
                          max_samples=1., 
                          oob_score=False)
    fit = clf.fit(X,y)
    imp = featImpMDA_Clustered(clf, X, y, clstrs, 10)

    imp.sort_values('mean', inplace=True)
    plt.figure(figsize=(10, 5))
    imp['mean'].plot(kind='barh', color='b', alpha=0.25, xerr=imp['std'], error_kw={'ecolor': 'r'})
    plt.title('Clustered MDA')
    plt.show()


def feature_importance_run(X,y,clstrs):
    normal_feat_imp(X,y)
    clustered_MDI_MDA(X,y,clstrs = clstrs)
    
