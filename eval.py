#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 08:53:35 2021

@author: guillermogarcia
"""

from sklearn.calibration import calibration_curve
from matplotlib import pyplot
from sklearn.metrics import classification_report, plot_roc_curve,roc_auc_score, f1_score

from data import *

def predict_missing_values(df,features_predict,features_train):
    
    for feat_predict in features_predict:
        print(f'Computing feature {feat_predict}')
        
        #Where are the missing values?
        idx = np.where((df[feat_predict].isnull()))
        df_nulls = df.loc[idx]
        df_without_nulls = df[df[feat_predict].notnull()]
        
        #Predict the missing values
        X_test = df_nulls[features_train]
        
        #Regression
        if df[feat_predict].dtype != 'object':
            model = xgb.XGBRegressor()
        #Classification
        else: model = xgb.XGBClassifier()
            
        model.load_model(f"models_missing_values/model_sklearn_{feat_predict}.json")
        y_pred = model.predict(X_test) 
        
        df_nulls[feat_predict] = y_pred
        
        df = pd.concat([df_nulls, df_without_nulls], axis=0)
        df = df.sort_index()
        
    print('Done!')

    return df


def preprocessing_test_set(df_test):
    target_class = 'classlabel'
    
    ## Binary features
    df_test = create_nan_class(df = df_test, columns = ['v95','v99','v32','v33','v85'])
    df_test = bincat2num(df = df_test, columns = ['v68','v70','v28','v84','v44',target_class])
    df_test = multicat2num(df = df_test, columns = ['v95'])
    target_class = 'classlabel_yes.'
    #Predict missing
    features_predict = ['v12','v7','v55','v42']
    features_train = ['v68_t','v50','v70_t','v20','v24','v97','v28_t','v53','v9','v44_t','v95_f','v95_nan','v95_t']
    df_test = predict_missing_values(df_test,features_predict,features_train)
    #v32
    df_test = df_test.drop(columns = ['v32'])
    #v7
    df_test = bincat2num(df_test,columns = ['v7'])
    #v85
    df_tmp_85 = multicat2num(df = df_test,columns = ['v85'])
    df_test['v85_p'] = df_tmp_85[['v85_p']]
    df_test = df_test.drop(columns = ['v85'])
    #v33 v99
    feat_classes = ['v33_ff','v33_bb','v33_v','v33_h','v99_ff']   
    df_test = make_binary_features_from_categorical(df_train = df_test ,feature = ['v33','v99'],feat_classes =  feat_classes)
    df_test = df_test.drop(columns = ['v33'])
    #v99
    df_test = encode_as_ordered(df_train = df_test, feature = 'v99')
    #v76
    df_test = bincat2num(df_test,columns = ['v76'])
    
    ## Continuous features
    #Outliers
    dict_features = {'v50':12,
                     'v24':16,
                     'v20':0.0015,
                     'v55':700,
                     'v42':6000000,
                     'v53':2000
                    }

    df_test = cap_outliers(df_train = df_test, dict_features = dict_features)
    
    feat_trans = Feature_transform(df_train = df_test, columns = ['v50','v55','v20','v24','v42','v53','v12','v97','v99'])
    df_test = feat_trans.min_max_transform(df = df_test)
    
    return df_test


def evaluate_model(rf_baseline,df_test_transformed,threshold = 0.5):

    target_class = 'classlabel_yes.'
    X_test = df_test_transformed.copy()
    X_test = df_test_transformed.drop(columns = [target_class])
    y_test = df_test_transformed[target_class]


    probs = rf_baseline.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)

    #Plot calibration
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    # plot calibrated reliability
    pyplot.plot(prob_pred, prob_true, marker='.')
    pyplot.show()

    #Classification report
    y_pred = rf_baseline.predict(X_test)

    print(classification_report(y_test, y_pred))

    print()
    print('---------- Choose threshold -----------')
    print()

    y_pred_ = probs > threshold
    print(classification_report(y_test, y_pred_))
    
    
    ## ROC-Curve
    
    plot_roc_curve(rf_baseline, X_test, y_test)  
    plt.show() 
    
    
    print('ROC-AUC score:', roc_auc_score(y_test, probs))
    print('F1.Score:',f1_score(y_test, y_pred, average='macro'))


