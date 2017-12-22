# - * - coding: utf8 - * - -
'''
@ Author : Tinkle G
Creation Time: 2017/11/9
'''

import pandas as pd
import numpy as np

def __predict(df,clf,onehot_feats,features):
    test_num = df[df['TARGET'] == -1].index
    train_num = df[df['TARGET'] != -1].index

    # One-hot Encoder
    df1 = pd.get_dummies(df[onehot_feats], columns=onehot_feats)
    df1 = pd.concat([df1, df[features]], axis=1)
    #print df1.head()

    train_x = df1.ix[train_num]
    print (train_x.shape)
    train_y = df.ix[train_num, 'TARGET']
    test_x = df1.ix[test_num]
    print (test_x.shape)
    print ('Train Target Mean: % s' % (float(np.sum(train_y))/len(train_y)))

    #clf = LogisticRegression(C=3, class_weight={0: 0.2, 1: 0.8})
    clf.fit(train_x, train_y)

    y_ = list(clf.predict_proba(test_x))

    EID = []
    FORTARGET = []
    PROB = []
    for i in range(len(test_num)):
        idx = test_num[i]
        EID.append(df['EID'][idx])
        if y_[i][1] >= 0.5:
            FORTARGET += [1]
        else:
            FORTARGET += [0]
        PROB += [round(y_[i][1], 4)]
    ret = pd.DataFrame({'EID': EID, 'FORTARGET': FORTARGET, 'PROB': PROB})
    print ('Predict Target Mean: % s' % (float(np.sum(ret['FORTARGET']))/len(ret)))

    return ret

def __crossvalidation(df,clf,onehot_feats,features,pca):
    from sklearn.metrics import f1_score
    from sklearn.metrics import roc_auc_score
    from sklearn.cross_validation import train_test_split
    from sklearn.decomposition import PCA

    y = df.index
    train_num, test_num = train_test_split(y, test_size=0.3, random_state=1)
    df1 = pd.get_dummies(df[onehot_feats], columns=onehot_feats)
    df1 = pd.concat([df1, df[features]], axis=1)

    train_x = df1.ix[train_num]
    if pca:
        pca.fit(train_x)
        train_x = pca.transform(train_x)

    print (train_x.shape)
    train_y = df.ix[train_num, 'TARGET']
    test_x = df1.ix[test_num]
    if pca:
        pca.fit(test_x)
        test_x = pca.transform(test_x)
    print (test_x.shape)

    print ('Train Target Mean: % s' % (float(np.sum(train_y))/len(train_y)))

    test_y = df.ix[test_num, 'TARGET']

    print ('Test Target Mean: % s' % (float(np.sum(test_y))/len(test_y)))

    clf.fit(train_x, train_y)

    y_pred = clf.predict(test_x)
    f1score = f1_score(test_y, y_pred)  # , average='binary')
    a= roc_auc_score(test_y, y_pred)
    print (int(a*1000)*100+f1score*100)

