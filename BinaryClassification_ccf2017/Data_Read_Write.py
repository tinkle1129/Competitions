# - * - coding: utf8 - * - -
'''
@ Author : Tinkle G
Creation Time: 2017/11/9
'''
import pandas as pd
write_path = 'result/'
os_path = 'data/'

def __submission(df=None):
    '''

    import random
    EID = []
    FORTARGET = []
    PROB = []
    for i in range(len(df)):
        EID.append(df['EID'][i])
        rand = random.randint(0, 1)
        FORTARGET += [rand]
        if rand == 0:
            PROB += [round(random.uniform(0, 0.5), 4)]
        else:
            PROB += [round(random.uniform(0.5, 1), 4)]
    new_df = pd.DataFrame({'EID': EID, 'FORTARGET': FORTARGET, 'PROB': PROB})
    new_df.to_csv(write_path+'submission.csv', sep=',', index=None)
    '''
    df.to_csv(write_path+'submission.csv', sep=',', index=None)







