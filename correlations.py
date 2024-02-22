#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 23:03:33 2021

@author: guillermogarcia
"""
from feature_clustering.feature_cluster import *

def correlations_variables(df_new,continuous_var,discrete_var):
    
    target_class = 'classlabel_yes.'
    # All variables linear and also non-linear relationship
    print('ALL VARIABLES NON-LINEAR')
    X = df_new.drop(columns = [target_class])
    
    q = X.shape[0] / X.shape[1]
    print(X.shape)
    print('q',q)
    clusters_non_lin  = get_feature_clusters(X = X,q = q,dependence_metric = 'information_variation',distance_metric = 'squared_angular')
    
    # Continuous variables
    df_continuous = df_new[continuous_var]
    print('CONTINUOUS VARIABLES NON-LINEAR')
    continuous_non_lin  = get_feature_clusters(X = df_continuous,q = q, dependence_metric = 'information_variation',distance_metric = 'squared_angular')
    print('CONTINUOUS VARIABLES LINEAR')
    continuous_lin  = get_feature_clusters(X = df_continuous,q = q, dependence_metric = 'linear',distance_metric = 'squared_angular')
    
    # Discrete variables non linear relationship
    df_discrete = df_new[discrete_var]
    print('DISCRETE VARIABLES NON-LINEAR')
    discrete_non_lin  = get_feature_clusters(X = df_discrete, q = q, dependence_metric = 'information_variation',distance_metric = 'squared_angular')
    
    return clusters_non_lin,continuous_non_lin, continuous_lin, discrete_non_lin
