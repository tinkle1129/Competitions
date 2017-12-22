# - * - coding: utf8 - * - -
'''
@ Author : Tinkle G
Creation Time: 2017/11/9
'''
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def chi_evalutation(data,target,k):
    return SelectKBest(chi2, k=k).fit_transform(data, target)