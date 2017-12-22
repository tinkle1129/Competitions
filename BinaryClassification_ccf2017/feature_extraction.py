# - * - coding: utf8 - * - -
'''
@ Author : TinkleG  & lcp
Creation Time: 2017/11/15
'''
#############import module#################
import pandas as pd
import numpy as np


############## Path ###################
os_path  ='data/'
train_path = os_path+'train.csv'
entbase_path = os_path+'1entbase.csv'
alter_path = os_path+'2alter.csv'
branch_path = os_path+'3branch.csv'
invest_path = os_path+'4invest.csv'
right_path = os_path+'5right.csv'
project_path = os_path+'6project.csv'
lawsuit_path = os_path+'7lawsuit.csv'
breakfaith_path = os_path+'8breakfaith.csv'
recuit_path = os_path+'9recruit.csv'
eva_path = os_path+'evaluation_public.csv'

#entbase_path = pd.read_csv(entbase_path) # table1
#alter_data = pd.read_csv(alter_path) #
#branch_data = pd.read_csv(branch_path)
#invest_data = pd.read_csv(invest_path)
#right_data = pd.read_csv(right_path)
#project_data = pd.read_csv(project_path)
#lawsuit_data = pd.read_csv(lawsuit_path)
#breakfaith_data = pd.read_csv(breakfaith_path)
#recuit_data = pd.read_csv(recuit_path)

# 处理企业基本数据 From table 1
def deal_enterbase():
    df_all = pd.read_csv(entbase_path).drop_duplicates()
    # 缺失值处理
    df_all['ZCZB'] = df_all['ZCZB'].fillna(0)
    # df_all['TARGET'] = df_all['TARGET'].fillna(-1)
    df_all['MPNUM'] = df_all['MPNUM'].fillna(0)
    df_all['INUM'] = df_all['INUM'].fillna(0)
    df_all['FINZB'] = df_all['FINZB'].fillna(0)
    df_all['FSTINUM'] = df_all['FSTINUM'].fillna(0)
    df_all['TZINUM'] = df_all['TZINUM'].fillna(0)
    hy_one_hot = pd.get_dummies(df_all['HY'], prefix='hy_')
    hy_num = df_all.groupby('HY', as_index=False)['EID'].count()
    hy_num.columns = ['HY', 'HY_NUM']
    hy_zc_max = df_all.groupby('HY', as_index=False)['ZCZB'].max()
    hy_zc_max.columns = ['HY', 'hy_zc_max']
    hy_zc_min = df_all.groupby('HY', as_index=False)['ZCZB'].min()
    hy_zc_min.columns = ['HY', 'hy_zc_min']
    hy_zc_mean = df_all.groupby('HY', as_index=False)['ZCZB'].mean()
    hy_zc_mean.columns = ['HY', 'hy_zc_mean']

    etype_num = df_all.groupby('ETYPE', as_index=False)['EID'].count()
    etype_num.columns = ['ETYPE', 'etype_num']
    etype_zc_mean = df_all.groupby('ETYPE', as_index=False)['ZCZB'].mean()
    etype_zc_mean.columns = ['ETYPE', 'etype_zc_mean']
    etype_zc_max = df_all.groupby('ETYPE', as_index=False)['ZCZB'].max()
    etype_zc_max.columns = ['ETYPE', 'etype_zc_max']
    etype_zc_min = df_all.groupby('ETYPE', as_index=False)['ZCZB'].min()
    etype_zc_min.columns = ['ETYPE', 'etype_zc_min']

    test_df = pd.DataFrame({})

    def f(x):
        import math
        if x > 1:
            return math.log(x)
        else:
            return 0

    test_df['ZCZB_20'] = pd.cut(df_all['ZCZB'].map(f).fillna(0).tolist(), 20,
                                labels=np.arange(20))  # cut对结果进行编码，便于之后做one-hot
    test_df['ZCZB_20_1'] = df_all['ZCZB']
    test_df['RGYEAR_14'] = pd.cut(df_all['RGYEAR'].fillna(0).tolist(), 14, labels=np.arange(14))
    test_df['HY'] = df_all['HY']

    test_df['EID'] = df_all['EID']
    # test_df['TARGET'] = df_all['TARGET'].fillna(-1)
    test_df['ETYPE'] = df_all['ETYPE']
    # test_df['MPNUM'] = min_max_scaler.fit_transform(df_all['MPNUM'])
    test_df['INUM'] = pd.cut(df_all['INUM'].map(f).fillna(0).tolist(), 10, labels=np.arange(10))
    test_df['FINZB'] = df_all['FINZB']  # min_max_scaler.fit_transform(df_all['FINZB'].map(f))
    test_df['FSTINUM'] = df_all['FSTINUM']  # preprocessing.normalize(df_all['FSTINUM'])[0]
    test_df['TZINUM'] = df_all['TZINUM']  # preprocessing.normalize(df_all['TZINUM'])[0]
    test_df['ON_YEAR'] = 2017 - df_all['RGYEAR']
    test_df = pd.merge(test_df, hy_num, how='left', on='HY')
    test_df = pd.merge(test_df, hy_zc_mean, how='left', on='HY')
    test_df = pd.merge(test_df, hy_zc_max, how='left', on='HY')
    test_df = pd.merge(test_df, hy_zc_min, how='left', on='HY')
    # test_df = pd.merge(test_df, etype_num,how='left',on='ETYPE')
    # test_df = pd.merge(test_df, etype_zc_mean,how='left',on='ETYPE')
    # test_df = pd.merge(test_df, etype_zc_max,how='left',on='ETYPE')
    # test_df = pd.merge(test_df, etype_zc_min, how='left',on='ETYPE')
    # test_df = pd.concat([test_df, hy_one_hot],axis=1)
    # del test_df['HY']
    HY_FEATURE = []
    for i in range(len(test_df)):
        if test_df.HY[i] not in [6, 42]:
            HY_FEATURE.append(0)
        else:
            HY_FEATURE.append(1)
    test_df['HY_FEATURE'] = HY_FEATURE

    def ff(x):
        return int(x / 5)

    test_df['ON_YEAR_feat'] = test_df.ON_YEAR.map(ff)
    return test_df

#处理变更数据 From Table 2
def deal_alter():
    df2 = pd.read_csv(alter_path).drop_duplicates()
    def regulation(x):
        import re
        try:
            return float(re.findall('[0-9\.]+',str(x))[0])
        except:
            return 0.0
    df2 = df2.fillna(0)
    df2['ALTAF']=df2['ALTAF'].map(regulation)
    df2['ALTBE']=df2['ALTBE'].map(regulation)
    altnum = df2.groupby('EID', as_index=False)['ALTERNO'].count()
    altnum.columns = ['EID', 'ALTNUM']

    altdatenum = df2[['EID','ALTDATE']].drop_duplicates().groupby('EID', as_index=False)['ALTDATE'].count()
    altdatenum.columns = ['EID', 'ALTDATENUM']

    df_right = pd.get_dummies(df2['ALTERNO'])
    features = df_right.columns
    df_right = pd.concat([df_right, df2[['EID']]], axis=1)
    alterno = df_right.groupby('EID', as_index=False)[features].sum()

    af05 = (df2['ALTAF'][df2.ALTERNO == '05']-df2['ALTBE'][df2.ALTERNO == '05']).tolist()
    af05eid = df2['EID'][df2.ALTERNO == '05'].tolist()
    sub05 = pd.DataFrame({'EID':af05eid,'af05':af05})
    rno_05_sub = sub05.groupby('EID', as_index=False)['af05'].mean()

    af27 = (df2['ALTAF'][df2.ALTERNO == '27']-df2['ALTBE'][df2.ALTERNO == '27']).tolist()
    af27eid = df2['EID'][df2.ALTERNO == '27'].tolist()
    sub27 = pd.DataFrame({'EID':af27eid,'af27':af27})
    rno_27_sub = sub27.groupby('EID', as_index=False)['af27'].mean()

    data = altnum
    data = pd.merge(data,altdatenum,on='EID',how='left')
    data = pd.merge(data,alterno,on='EID',how='left')
    data = pd.merge(data,rno_05_sub,on='EID',how='left')
    data = pd.merge(data,rno_27_sub,on='EID',how='left')

    data['ALTRATE'] = data.ALTNUM/data.ALTDATENUM

    from datetime import datetime
    def striptime(x):
        current = datetime.strptime('2017-09', "%Y-%m")
        return (current - datetime.strptime(x, "%Y-%m")).days
    df2['altdata'] = df2.ALTDATE.map(striptime)

    alter_max = df2.groupby('EID', as_index=False)['altdata'].max()
    alter_max.columns = ['EID', 'ALTER_MAX']

    alter_min = df2.groupby('EID', as_index=False)['altdata'].min()
    alter_min.columns = ['EID', 'ALTER_MIN']

    data = pd.merge(data,alter_max,on='EID',how='left')
    data = pd.merge(data,alter_min,on='EID',how='left')

    data['alter_duration']=data.ALTER_MAX-data.ALTER_MIN
    data['alter_mean_duration'] = data['alter_duration']/data.ALTNUM
    data = data.fillna(0)
    return data

#处理分支数据 From Table 3
def deal_branch():
    branch_data = pd.read_csv(branch_path).drop_duplicates()
    entbase_data = pd.read_csv(entbase_path).drop_duplicates()
    branch_data['last_year'] = branch_data['B_ENDYEAR'].fillna(2017)
    branch_data = branch_data.sort_values(by=['EID','B_REYEAR'])
    branch_data = pd.merge(branch_data, entbase_data.ix[:,['EID','RGYEAR']], how='left', on='EID')
    pre_year = [branch_data.ix[branch_data.index[0], 'RGYEAR']]
    for i in range(1,branch_data.shape[0]):
        if branch_data.ix[branch_data.index[i], 'EID'] == branch_data.ix[branch_data.index[i - 1], 'EID']:
            pre_year.append(branch_data.ix[branch_data.index[i-1], 'B_REYEAR'])
        else:
            pre_year.append(branch_data.ix[branch_data.index[i], 'RGYEAR'])
    branch_data['PRE_YEAR'] = pre_year
    branch_data['DURATION'] = branch_data['B_REYEAR'] - branch_data['PRE_YEAR']
    branch_data['ON_TIME'] = branch_data['last_year'] - branch_data['B_REYEAR']
    branch_start_num = branch_data.groupby('EID', as_index=False)['B_REYEAR'].count()
    branch_end_num = branch_data.groupby('EID', as_index=False)['B_ENDYEAR'].count()
    branch_home_num = branch_data.groupby('EID', as_index=False)['IFHOME'].sum()
    branch_mean_dur = branch_data.groupby('EID',as_index=False)['DURATION'].mean()
    branch_on_time = branch_data.groupby('EID',as_index=False)['ON_TIME'].mean()
    branch_first_time = branch_data.groupby('EID',as_index=False)['DURATION'].first()
    branch_last_time = branch_data.groupby('EID', as_index=False)['DURATION'].last()
    data = branch_start_num
    data = pd.merge(data, branch_end_num, how='left', on='EID')
    data = pd.merge(data, branch_home_num, how='left', on='EID')
    data = pd.merge(data, branch_mean_dur, how='left', on='EID')
    data = pd.merge(data, branch_on_time, how='left', on='EID')
    data = pd.merge(data, branch_first_time, how='left', on='EID' )
    data = pd.merge(data, branch_last_time, how='left', on='EID')
    data.columns = ['EID','branch_start_num','branch_end_num','home_num','start_duration','on_time','first_branch','last_branch']
    data['branch_on_num'] = data['branch_start_num'] - data['branch_end_num']
    data['branch_ratio'] = data['branch_end_num'].map(float)/ data['branch_start_num']

    def f(x):
        return float(int(x * 10)) / 10

    data['branch_ratio_feat'] = data['branch_ratio'].map(f)

    return data


#处理投资数据 From Table 4
def deal_invest():
    invest_data = pd.read_csv(invest_path).drop_duplicates()
    invest_data['INVEST'] = 1
    invest_num = invest_data.groupby('EID',as_index=False)['INVEST'].count()
    invest_num.columns = ['EID', 'invest_num']
    invest_mean_btbl = invest_data.groupby('EID',as_index= False)['BTBL'].mean()
    invest_mean_btbl.columns = ['EID','BTBL_mean']
    invest_btbl_max = invest_data.groupby('EID', as_index=False)['BTBL'].max()
    invest_btbl_max.columns = ['EID','BTBL_max']
    invest_btbl_min = invest_data.groupby('EID', as_index=False)['BTBL'].min()
    invest_btbl_min.columns = ['EID', 'BTBL_min']
    invest_end_num = invest_data.groupby('EID', as_index=False)['BTENDYEAR'].count()
    invest_end_num.columns = ['EID','BT_end_nums']
    bteid = invest_data.groupby('BTEID',as_index= False)['EID'].count()
    bteid.columns = ['EID','BTNUM']
    home_num = invest_data.groupby('EID',as_index=False)['IFHOME'].sum()
    home_num.columns = ['EID','T_home_num']
    invest_data['BTENDYEAR'] = invest_data['BTENDYEAR'].fillna(2017)
    invest_data['T_on_time'] = invest_data['BTENDYEAR'] - invest_data['BTYEAR']
    bt_ontime = invest_data.groupby('EID',as_index=False)['T_on_time'].mean()
    bt_ontime.columns = ['EID','T_time_mean']
    bt_ontime_min = invest_data.groupby('EID',as_index=False)['T_on_time'].min()
    bt_ontime_min.columns = ['EID', 'T_time_min']
    bt_ontime_max = invest_data.groupby('EID', as_index= False)['T_on_time'].max()
    bt_ontime_max.columns = ['EID', 'T_time_max']
    data = invest_num
    data = pd.merge(data, invest_mean_btbl, how='left', on='EID')
    data = pd.merge(data, invest_btbl_max, how= 'left', on='EID')
    data = pd.merge(data, invest_btbl_min, how='left', on='EID')
    data = pd.merge(data, invest_end_num, how='left', on= 'EID')
    data = pd.merge(data, home_num, how= 'left',on='EID')
    data = pd.merge(data, bt_ontime, how='left', on='EID')
    data = pd.merge(data, bt_ontime_max,how='left',on='EID')
    data = pd.merge(data, bt_ontime_min, how='left',on='EID')
    data = pd.merge(data, bteid, how='left',on='EID')
    data['BTNUM'] = data['BTNUM'].fillna(0)
    data['NOTHOME'] = data['invest_num'] - data['T_home_num']
    # data['BT1RATE'] = data['T_home_num'] / data['invest_num']
    # data['BT0RATE'] = data['NOTHOME'] / data['invest_num']
    data['on_branch'] = data['invest_num'] - data['BT_end_nums']
    data['invest_ratio'] = data['BT_end_nums'].map(float)/data['invest_num']
    def f(x):
        return float(int(x * 5))
    data['invest_ratio_feat']=data.invest_ratio.map(f)
    return data


# 处理权利部分数据 From Table 5
def deal_right():
    entbase_data = pd.read_csv(entbase_path).drop_duplicates()
    entbase_data = entbase_data.ix[:, ['EID', 'RGYEAR']]
    right_data = pd.read_csv(right_path).drop_duplicates()
    right_data = pd.merge(right_data, entbase_data, how='left', on='EID')
    df_right = pd.get_dummies(right_data['RIGHTTYPE'])
    right_data = pd.concat([right_data, df_right], axis=1)
    right_data = right_data.sort_values(by=['EID', 'ASKDATE'])
    right_num_type = right_data.groupby('EID', as_index=False)[11, 12, 20, 30, 40, 50, 60].sum()
    right_num_type.columns = ['EID', 'right_11', 'right_12', 'right_20', 'right_30', 'right_40', 'right_50', 'right_60']
    right_num = right_data.groupby('EID', as_index=False)['TYPECODE'].count()
    right_num.columns = ['EID', 'ASKNUM']
    fb_right_num = right_data.groupby('EID', as_index=False)['FBDATE'].count()
    fb_right_num.columns = ['EID', 'FBNUM']
    right_first_time = right_data.groupby('EID', as_index=False)['ASKDATE'].first()
    right_first_time.columns = ['EID', 'right_first_time']
    right_last_time = right_data.groupby('EID', as_index=False)['ASKDATE'].last()
    right_last_time.columns = ['EID', 'right_last_time']
    data = right_num_type
    data = pd.merge(data, right_num, how='left', on='EID')
    data = pd.merge(data, fb_right_num, how='left', on='EID')
    data = pd.merge(data, right_first_time, how='left', on='EID')
    data = pd.merge(data, right_last_time, how='left', on='EID')
    data = pd.merge(data, entbase_data, how='left', on='EID')
    data['first_ask_time'] = data['right_first_time'].str.slice(0, 4).astype('int')
    data['first_ask_duration'] = data['RGYEAR'] - data['first_ask_time']
    del data['first_ask_time']
    data['right_first_time'] = pd.to_datetime(data['right_first_time'])
    data['right_last_time'] = pd.to_datetime(data['right_last_time'])
    duration = []
    for i in range(data.shape[0]):
        duration.append((data.ix[i, 'right_last_time'] - data.ix[i, 'right_first_time']).days)
    data['right_duration'] = duration
    data['right_mean duration'] = data['right_duration'] / data['ASKNUM']
    data['fb_rate'] = data['FBNUM'] / data['ASKNUM']
    del data['right_first_time']
    del data['right_last_time']
    data['setup_time'] = 2017 - data['RGYEAR']
    data['ask_freq'] = data['ASKNUM'] / data['setup_time']
    data['fb_freq'] = data['FBNUM'] / data['setup_time']
    del data['RGYEAR']
    del data['setup_time']

    def f(x):
        if x == -1 or x == 0:
            return x
        trans = int(x) / 10 + 1
        if trans > 10:
            return 10
        return trans

    data['right_11_feat'] = data.right_11.map(f)
    data['right_12_feat'] = data.right_12.map(f)
    data['right_20_feat'] = data.right_20.map(f)
    data['right_30_feat'] = data.right_30.map(f)
    data['right_40_feat'] = data.right_40.map(f)
    data['right_50_feat'] = data.right_50.map(f)
    data['right_60_feat'] = data.right_60.map(f)

    def ff(x):
        return float(int(x * 20)) / 10

    data['fb_rate_feat'] = data.fb_rate.map(ff)
    return data


# 处理项目数据 From table 6
def deal_project():
    df6 = pd.read_csv(project_path).drop_duplicates()
    from datetime import datetime
    def striptime(x):
        current = datetime.strptime('2017-09', "%Y-%m")
        return (current - datetime.strptime(x, "%Y-%m")).days

    # PROJECT_NUM
    project_num = df6.groupby('EID', as_index=False)['TYPECODE'].count()
    project_num.columns = ['EID', 'PROJECT_NUM']

    project_1_num = df6[['EID', 'TYPECODE']][df6.IFHOME == 1].groupby('EID', as_index=False)['TYPECODE'].count()
    project_1_num.columns = ['EID', 'PROJECT_1_NUM']

    project_0_num = df6[['EID', 'TYPECODE']][df6.IFHOME == 0].groupby('EID', as_index=False)['TYPECODE'].count()
    project_0_num.columns = ['EID', 'PROJECT_0_NUM']

    date_num = df6[['EID', 'DJDATE']].drop_duplicates().groupby('EID', as_index=False)['DJDATE'].count()
    date_num.columns = ['EID', 'DATE_NUM']

    date_1_num = df6[['EID', 'DJDATE']][df6.IFHOME == 1].drop_duplicates().groupby('EID', as_index=False)[
        'DJDATE'].count()
    date_1_num.columns = ['EID', 'DATE_1_NUM']

    date_0_num = df6[['EID', 'DJDATE']][df6.IFHOME == 0].drop_duplicates().groupby('EID', as_index=False)[
        'DJDATE'].count()
    date_0_num.columns = ['EID', 'DATE_0_NUM']

    df6['djdate'] = df6.DJDATE.map(striptime)

    datemax = df6.groupby('EID', as_index=False)['djdate'].max()
    datemax.columns = ['EID', 'DATE_MAX']

    datemin = df6.groupby('EID', as_index=False)['djdate'].min()
    datemin.columns = ['EID', 'DATE_MIN']

    data = project_num
    data = pd.merge(data, project_1_num, on='EID', how='left')
    data = pd.merge(data, project_0_num, on='EID', how='left')
    data = pd.merge(data, date_num, on='EID', how='left')
    data = pd.merge(data, date_1_num, on='EID', how='left')
    data = pd.merge(data, date_0_num, on='EID', how='left')
    data = pd.merge(data, datemax, on='EID', how='left')
    data = pd.merge(data, datemin, on='EID', how='left')
    data = data.fillna(0)
    data['PROJECT_1_RATE'] = data.PROJECT_1_NUM.map(float) / data.PROJECT_NUM
    data['PROJECT_0_RATE'] = data.PROJECT_0_NUM.map(float) / data.PROJECT_NUM
    data['DATE_1_RATE'] = data.DATE_1_NUM.map(float) / data.DATE_NUM
    data['DATE_0_RATE'] = data.DATE_0_NUM.map(float) / data.DATE_NUM
    data['DATE_DELTA'] = data.DATE_MAX - data.DATE_MIN

    def f(x):
        return int(x) / 10

    data['PROJECT_NUM_feat'] = data.PROJECT_NUM.map(f)
    data['PROJECT_1_NUM_feat'] = data.PROJECT_1_NUM.map(f)
    data['PROJECT_0_NUM_feat'] = data.PROJECT_0_NUM.map(f)
    data['DATE_NUM_feat'] = data.DATE_NUM.map(f)
    data['DATE_1_NUM_feat'] = data.DATE_1_NUM.map(f)
    data['DATE_0_NUM_feat'] = data.DATE_0_NUM.map(f)

    def f(x):
        return int(x) / 100

    data['DATE_MIN_feat'] = data.DATE_DELTA.map(f)
    data['DATE_MAX_feat'] = data.DATE_MAX.map(f)
    data['DATE_MIN_feat'] = data.DATE_MIN.map(f)

    return data

#处理被执行数据 From Table 7
def deal_lawsuit():
    lawsuit_data = pd.read_csv(lawsuit_path).drop_duplicates()
    lawsuit_data = lawsuit_data.sort_values(by=['EID', 'LAWDATE'])
    print(lawsuit_data.isnull().any())
    case_num = lawsuit_data.groupby('EID',as_index=False)['TYPECODE'].count()
    case_num.columns = ['EID','case_num']
    case_first_time = lawsuit_data.groupby('EID',as_index=False)['LAWDATE'].first()
    case_first_time.columns = ['EID','case_first_time']
    case_last_time = lawsuit_data.groupby('EID',as_index=False)['LAWDATE'].last()
    case_last_time.columns = ['EID','case_last_time']
    no_label_case = lawsuit_data[lawsuit_data['LAWAMOUNT'] == 0]
    no_label_case = no_label_case.groupby('EID',as_index=False)['TYPECODE'].count()
    no_label_case.columns = ['EID','no_label_num']
    label_case = lawsuit_data[lawsuit_data['LAWAMOUNT'] != 0]
    label_case_num = label_case.groupby('EID', as_index = False)['TYPECODE'].count()
    label_case_num.columns = ['EID','label_case_num']
    label_case_mean = label_case.groupby('EID',as_index=False)['LAWAMOUNT'].mean()
    label_case_mean.columns = ['EID','mean_lawamount']
    label_case_max = label_case.groupby('EID', as_index=False)['LAWAMOUNT'].max()
    label_case_max.columns = ['EID', 'max_lawamount']
    label_case_min = label_case.groupby('EID', as_index=False)['LAWAMOUNT'].min()
    label_case_min.columns = ['EID', 'min_lawamount']
    data = case_num
    data = pd.merge(data, case_first_time,how='left', on='EID')
    data = pd.merge(data, case_last_time, how='left', on='EID')
    data = pd.merge(data, no_label_case, how = 'left',on='EID')
    data['no_label_num'] = data['no_label_num'].fillna(0)
    data = pd.merge(data, label_case_mean, how='left',on='EID')
    data = pd.merge(data, label_case_max, how='left',on='EID')
    data = pd.merge(data, label_case_min, how='left',on='EID')
    data = pd.merge(data, label_case_num, how='left', on='EID')
    data = data.fillna(0)
    data['case_first_time'] = pd.to_datetime(data['case_first_time'])
    data['case_last_time'] = pd.to_datetime(data['case_last_time'])
    duration = []
    for i in range(data.shape[0]):
        duration.append((data.ix[i,'case_last_time'] - data.ix[i,'case_first_time']).days)
    data['case_duration'] = duration
    data['case_mean_duration'] = data['case_duration'] / data['case_num']
    del data['case_last_time']
    del data['case_first_time']
    def f(x):
        if x==0 or x==-1:
            return x
        return int(x*10)+1
    data['case_rate']=data.no_label_num.map(float)/data.case_num
    data['case_rate_feat']=data.case_rate.map(f)
    return data

#处理被执行数据 From Table 7
def deal_alter():
    df2 = pd.read_csv(alter_path).drop_duplicates()
    def regulation(x):
        import re
        try:
            return float(re.findall('[0-9\.]+',str(x))[0])
        except:
            return 0.0
    df2 = df2.fillna(0)
    df2['ALTAF']=df2['ALTAF'].map(regulation)
    df2['ALTBE']=df2['ALTBE'].map(regulation)
    altnum = df2.groupby('EID', as_index=False)['ALTERNO'].count()
    altnum.columns = ['EID', 'ALTNUM']

    altdatenum = df2[['EID','ALTDATE']].drop_duplicates().groupby('EID', as_index=False)['ALTDATE'].count()
    altdatenum.columns = ['EID', 'ALTDATENUM']

    df_right = pd.get_dummies(df2['ALTERNO'])
    features = df_right.columns
    df_right = pd.concat([df_right, df2[['EID']]], axis=1)
    alterno = df_right.groupby('EID', as_index=False)[features].sum()

    af05 = (df2['ALTAF'][df2.ALTERNO == '05']-df2['ALTBE'][df2.ALTERNO == '05']).tolist()
    af05eid = df2['EID'][df2.ALTERNO == '05'].tolist()
    sub05 = pd.DataFrame({'EID':af05eid,'af05':af05})
    rno_05_sub = sub05.groupby('EID', as_index=False)['af05'].mean()

    af27 = (df2['ALTAF'][df2.ALTERNO == '27']-df2['ALTBE'][df2.ALTERNO == '27']).tolist()
    af27eid = df2['EID'][df2.ALTERNO == '27'].tolist()
    sub27 = pd.DataFrame({'EID':af27eid,'af27':af27})
    rno_27_sub = sub27.groupby('EID', as_index=False)['af27'].mean()

    data = altnum
    data = pd.merge(data,altdatenum,on='EID',how='left')
    data = pd.merge(data,alterno,on='EID',how='left')
    data = pd.merge(data,rno_05_sub,on='EID',how='left')
    data = pd.merge(data,rno_27_sub,on='EID',how='left')

    data['ALTRATE'] = data.ALTNUM/data.ALTDATENUM

    from datetime import datetime
    def striptime(x):
        current = datetime.strptime('2017-09', "%Y-%m")
        return (current - datetime.strptime(x, "%Y-%m")).days
    df2['altdata'] = df2.ALTDATE.map(striptime)

    alter_max = df2.groupby('EID', as_index=False)['altdata'].max()
    alter_max.columns = ['EID', 'ALTER_MAX']

    alter_min = df2.groupby('EID', as_index=False)['altdata'].min()
    alter_min.columns = ['EID', 'ALTER_MIN']

    data = pd.merge(data,alter_max,on='EID',how='left')
    data = pd.merge(data,alter_min,on='EID',how='left')

    data['alter_duration']=data.ALTER_MAX-data.ALTER_MIN
    data['alter_mean_duration'] = data['alter_duration']/data.ALTNUM
    data = data.fillna(0)
    return data

#处理被执行数据 From Table 8
def deal_breakfaith():
    from datetime import datetime
    def striptime(x):
        current = datetime.strptime('2017/09/01', "%Y/%m/%d")
        return (current - datetime.strptime(x, "%Y/%m/%d")).days

    df8 = pd.read_csv(breakfaith_path).drop_duplicates()
    print df8.dtypes

    breakfaith_num = df8.groupby('EID', as_index=False)['TYPECODE'].count()
    breakfaith_num.columns = ['EID', 'BREAKFAITH_NUM']

    breakfaith_end_num = df8[['EID', 'TYPECODE']][df8.SXENDDATE.notnull()].groupby('EID', as_index=False)[
        'TYPECODE'].count()
    breakfaith_end_num.columns = ['EID', 'BREAKFAITH_END_NUM']

    breakfaith_on_num = df8[['EID', 'TYPECODE']][df8.SXENDDATE.isnull()].groupby('EID', as_index=False)[
        'TYPECODE'].count()
    breakfaith_on_num.columns = ['EID', 'BREAKFAITH_NOTEND_NUM']

    df8['days'] = df8.FBDATE.map(striptime)
    breakfaith_max = df8.groupby('EID', as_index=False)['days'].max()
    breakfaith_max.columns = ['EID', 'BREAKFAITH_MAX']

    breakfaith_min = df8.groupby('EID', as_index=False)['days'].min()
    breakfaith_min.columns = ['EID', 'BREAKFAITH_MIN']

    tmp = df8[['EID', 'FBDATE', 'SXENDDATE']][df8.SXENDDATE.notnull()]
    tmp['FBDATE'] = tmp.FBDATE.map(striptime)
    tmp['SXENDDATE'] = tmp.SXENDDATE.map(striptime)
    tmp['delta_time'] = tmp.FBDATE - tmp.SXENDDATE

    BREAKFAITH_END_MAX = tmp.groupby('EID', as_index=False)['delta_time'].max()
    BREAKFAITH_END_MAX.columns = ['EID', 'BREAKFAITH_END_MAX']

    BREAKFAITH_END_MIN = tmp.groupby('EID', as_index=False)['delta_time'].min()
    BREAKFAITH_END_MIN.columns = ['EID', 'BREAKFAITH_END_MIN']

    BREAKFAITH_END_MEAN = tmp.groupby('EID', as_index=False)['delta_time'].mean()
    BREAKFAITH_END_MEAN.columns = ['EID', 'BREAKFAITH_END_MEAN']

    data8 = breakfaith_num
    data8 = pd.merge(data8, breakfaith_end_num, on='EID', how='left')
    data8 = pd.merge(data8, breakfaith_on_num, on='EID', how='left')
    data8 = pd.merge(data8, breakfaith_max, on='EID', how='left')
    data8 = pd.merge(data8, breakfaith_min, on='EID', how='left')
    data8 = pd.merge(data8, BREAKFAITH_END_MAX, on='EID', how='left')
    data8 = pd.merge(data8, BREAKFAITH_END_MIN, on='EID', how='left')
    data8 = pd.merge(data8, BREAKFAITH_END_MEAN, on='EID', how='left')

    data8 = data8.fillna(0)
    data8['BREAKFAITH_DELTA'] = data8.BREAKFAITH_MAX - data8.BREAKFAITH_MIN
    data8['BREAKFAITH_NOTEND_RATE'] = data8.BREAKFAITH_NOTEND_NUM.map(float) / data8.BREAKFAITH_NUM
    data8['BREAKFAITH_NUM_feat'] = 1

    def f(x):
        if x == 0:
            return 0
        return 1

    data8['BREAKFAITH_NOTEND_RATE_feat'] = data8.BREAKFAITH_NOTEND_RATE.map(f)
    data8 = data8.fillna(0)
    return data8

#处理被执行数据 From Table 9

def deal_recuit():
    df_recr = pd.read_csv(recuit_path).drop_duplicates()
    print(df_recr.head())
    df_recr['RECRNUM'] = df_recr['RECRNUM'].fillna(0)

    dict_recr_tot_num = {}
    dict_recr_max_num = {}
    dict_recr_min_num = {}
    dict_recr_max_code_num = {}
    dict_recrtime = {}

    dict_recr_cnt = {}
    dict_recr_time_record = {}

    dict = {}
    for i in range(len(df_recr)):
        key = df_recr['EID'][i]
        recr_num = df_recr['RECRNUM'][i]
        recr_code = int(df_recr['WZCODE'][i][-1])
        date = [int(v) for v in df_recr['RECDATE'][i].split("-")]
        recr_last_time = date[1] if date[0] == 2015 else 8 + 12 - date[1]
        dict.setdefault(key, []).append((recr_last_time, recr_code, recr_num))

    for key in dict.keys():
        lis = sorted(dict[key], key=lambda x: x[0])
        n = len(lis)
        last_time = -1
        recr_num = 0
        recr_code_num = 0
        for i in range(n + 1):
            if i == n or lis[i][0] != last_time:
                if recr_num > 0:
                    if key not in dict_recr_tot_num:
                        dict_recr_tot_num[key] = recr_num
                    else:
                        dict_recr_tot_num[key] += recr_num
                    if key not in dict_recr_max_num:
                        dict_recr_max_num[key] = recr_num
                    else:
                        dict_recr_max_num[key] = max(dict_recr_max_num[key], recr_num)
                    if key not in dict_recr_min_num:
                        dict_recr_min_num[key] = recr_num
                    else:
                        dict_recr_min_num[key] = min(dict_recr_min_num[key], recr_num)

                if key not in dict_recr_max_code_num:
                    dict_recr_max_code_num[key] = recr_code_num
                else:
                    dict_recr_max_code_num[key] = max(dict_recr_max_code_num[key], recr_code_num)
                if i == n:
                    break
                recr_num = lis[i][2]
                recr_code_num = 1
                last_time = lis[i][0]
            else:
                recr_num += lis[i][2]
                recr_code_num += 1

        last_time = -1
        recr_cnt = 0
        for i in range(n + 1):
            if i == n:
                if key not in dict_recr_cnt:
                    dict_recr_cnt[key] = recr_cnt
                break
            if lis[i][0] != last_time:
                recr_cnt += 1
                last_time = lis[i][0]
                dict_recr_time_record.setdefault(key, []).append(last_time)

        if recr_cnt < 2:
            dict_recr_time_record[key] = 0
        else:
            arr = dict_recr_time_record[key]
            sum = 0
            for i in range(len(arr) - 1):
                sum += arr[i + 1] - arr[i]
            dict_recr_time_record[key] = sum * 1.0 / (recr_cnt - 1)

        '''
        print(lis)
        print(dict_recr_max_num)
        print(dict_recr_min_num)
        print(dict_recr_max_code_num)
        print(dict_recr_tot_num)

        print("************************")

        print(dict_recr_cnt)
        print(dict_recr_time_record)
        '''

    df_recr_max_num = pd.DataFrame(list(dict_recr_max_num.items()), columns=['EID', 'RECR_MAX_NUM'])
    df_recr_min_num = pd.DataFrame(list(dict_recr_min_num.items()), columns=['EID', 'RECR_MIN_NUM'])
    df_recr_tot_num = pd.DataFrame(list(dict_recr_tot_num.items()), columns=['EID', 'RECR_TOT_NUM'])
    df_recr_max_code_num = pd.DataFrame(list(dict_recr_max_code_num.items()), columns=['EID', 'RECR_MAX_CODE_NUM'])
    df_recr_cnt = pd.DataFrame(list(dict_recr_cnt.items()), columns=['EID', 'RECR_CNT'])
    df_recr_ave_time = pd.DataFrame(list(dict_recr_time_record.items()), columns=['EID', 'RECR_AVE_TIME'])

    df_recr = df_recr_max_num.merge(df_recr_min_num, on='EID', how='outer')
    df_recr = df_recr.merge(df_recr_tot_num, on='EID', how='outer')
    df_recr = df_recr.merge(df_recr_max_code_num, on='EID', how='outer')
    df_recr = df_recr.merge(df_recr_cnt, on='EID', how='outer')
    df_recr = df_recr.merge(df_recr_ave_time, on='EID', how='outer')

    df_recr = df_recr.fillna(-1)
    df_recr['EID']=df_recr.EID.map(int)

    return df_recr


# 9
print (9)
#recuit_data = deal_recuit()

# table1
print (1)
entbase_data = deal_enterbase()

# 2
print (2)
alter_data = deal_alter()

# 3
print (3)
branch_data = deal_branch()

# 4
print (4)
invest_data = deal_invest()

# 5
print (5)
right_data = deal_right()

# 6
print (6)
project_data = deal_project()

# 7
#print (7)
#lawsuit_data = deal_lawsuit()

# 8
#print (8)
#breakfaith_data = deal_breakfaith()




data = entbase_data
data = pd.merge(data,branch_data,on='EID',how='left')
data = pd.merge(data,alter_data,on='EID',how='left')
data = pd.merge(data,branch_data,on='EID',how='left')
data = pd.merge(data,invest_data,on='EID',how='left')
data = pd.merge(data,right_data,on='EID',how='left')
data = pd.merge(data,project_data,on='EID',how='left')
#data = pd.merge(data,lawsuit_data,on='EID',how='left')
#data = pd.merge(data,breakfaith_data,on='EID',how='left')
#data = pd.merge(data,recuit_data,on='EID',how='left')

train_data = pd.read_csv(train_path)
eva_data = pd.read_csv(eva_path)

train_ = pd.merge(train_data,data,on='EID',how='left')
train_ = train_.fillna(-1)

test_ = pd.merge(eva_data,data,on='EID',how='left')
test_ = test_.fillna(-1)

train_.to_csv(os_path+'train_feature_1_9.csv',index=None)
test_.to_csv(os_path+'test_feature_1_9.csv',index=None)