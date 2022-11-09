import logging
import time

import pandas as pd
from pandas import DataFrame


def merge(feature_df:DataFrame,sample_ids_df:DataFrame):
    return pd.merge(left=feature_df,right=sample_ids_df,left_on='user_id',right_on='user_id',how='inner')


def concat(pos_df:DataFrame,neg_df:DataFrame):
    '''
    按行拼接
    :param pos_df:
    :param neg_df:
    :return:
    '''
    label_df = pd.concat([pos_df, neg_df])
    return label_df


def concat_user_actions(datas:DataFrame,miss_value=0):
    '''
    按照user_id对用户的action list进行拼接,
    以user_id为行名，action作为列名，对频次重塑成一行
    :param datas: （user_id，action,fre)
    :param miss_value：缺失值填充，默认用0进行填充
    :return:
    '''
    st = time.time()
    concat_dats = datas.pivot_table(index='user_id',columns='action',values='fre').fillna(miss_value)
    et = time.time()
    logging.info('concat_user_actions time cost:{} second'.format(et-st))
    return concat_dats.reset_index()


def insert(pos_df:DataFrame,label):
    '''
    在指定列进行插入值
    :param pos_df:
    :param label:
    :return:
    '''
    pos_df.insert(loc=0,column='label',value=label)

