import logging
import time

from pandas import DataFrame, Series
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def get_sample_rate(datas:DataFrame,pos_neg_rate=1/3):
    '''

    :param datas:
    :param pos_neg_rate: 正负样本比，默认为1/3
    :return:
    '''
    label_couts:dict = datas['label'].value_counts().to_dict()
    pos_count = label_couts.get(1)
    neg_count = label_couts.get(0)
    neg_ = pos_count /pos_neg_rate
    neg_rate = neg_/neg_count
    if neg_rate>1:neg_rate=1
    return neg_rate


def neg_sampleing(datas:DataFrame,neg_rate):
    '''
    对负样本进行抽样
    :param datas: pd.DataFrame
    :param neg_rate: 负样本抽样率
    :return:
    '''
    if neg_rate>=1:
        return datas
    neg_samples = datas[datas['label']==0].sample(frac=neg_rate)
    pos_samples = datas[datas['label']==1]
    return pd.concat([neg_samples,pos_samples])



def value_count(datas:DataFrame):
    '''
    统计每个事件的覆盖人群数量
    :param datas:
    :return:
    '''
    cols = datas.columns.to_list()
    if 'label' in cols:
        cols.remove('label')
    if 'user_id' in cols:
        cols.remove('user_id')
    sample_count = len(datas)
    action_cls_dict = {}
    f = open('./action_cls.csv',mode='w',encoding='utf-8')

    for c in cols:
        # {0.0: 122993, 1.0: 1883, 2.0: 249, 3.0: 32, 4.0: 6}
        action_cls = {}
        for fre,count in datas[c].value_counts().to_dict().items():
            percent = count/sample_count
            f.write(','.join([str(c),str(fre),str(count),str(percent)]) + "\n")
            f.flush()
            action_cls[fre] = percent
        action_cls_dict[c] = action_cls

    return action_cls_dict


def merge_action_feas(datas:DataFrame,threshold=0.95,combine_col_name='others',drop_cols=[]):
    '''
    对覆盖率少的特征，进行合并为一个特征
    :return:
    '''

    cols = datas.columns.tolist()
    if "label" in cols:
        cols.remove("label")
    if "user_id" in cols:
        cols.remove("user_id")
    if len(drop_cols) == 0:
        actions_cls_dict = value_count(datas[cols])
        # 当len(drop_cols)表示是训练阶段，否则为预测阶段
        for col, col_dict in actions_cls_dict.items():
            cover_rate = col_dict.get(0)
            if cover_rate is not None and cover_rate>=threshold:
                drop_cols.append(col)

    logging.info("total low cover cols len:{}".format(len(drop_cols)))
    logging.info("merge_action_feas raw data shape:{}".format(datas.shape))
    datas[combine_col_name] = datas[drop_cols].apply(lambda x:x.sum(),axis=1)
    return datas.drop(drop_cols,axis=1),drop_cols


def cate_feature_name(col_val,col_name):
    return col_name + "@" + col_val


def get_cate_feature_map(datas:DataFrame,cate_featurs:list):
    cate_feature_map = {}
    for col_name in cate_featurs:
        for idx,col_val in enumerate(set(datas[col_name].values)):
            cate_feature_val = cate_feature_name(col_val,col_name)
            cate_feature_map[cate_feature_val] = idx + 1
        # 添加missing值，在预测阶段可能存在有些类别特征不在训练中出现
        cate_feature_val = cate_feature_name("missing",col_name)
        cate_feature_map[cate_feature_val] = 0

    return cate_feature_map


def get_feature_idx(fea_val,fea_group,fea_map:dict):
    fea = cate_feature_name(fea_val,fea_group)
    if fea not in fea_map.keys():
        fea = cate_feature_name("missing",fea_group)
    return fea_map.get(fea)


def cate_feature_encode(datas:DataFrame,cate_features:list,feature_map:dict):
    def encode(x:Series,feture_dict):
        return x.apply(get_feature_idx,args=(x.name,feture_dict))
    datas[cate_features] = datas[cate_features].apply(encode,args=feature_map,axis=0)



def drop_low_cover_features(datas:DataFrame,threshold):
    '''
    剔除覆盖率低的事件
    :param datas:
    :param threshold:
    :return:
    '''
    cols = datas.columns.tolist()
    if "label" in cols:
        cols.remove("label")
    if "user_id" in cols:
        cols.remove("user_id")
    actions_cls_dict = value_count(datas[cols])
    drop_cols = []
    for col, col_dict in actions_cls_dict.items():
        cover_rate = col_dict.get(0)
        if cover_rate is not None and cover_rate>=threshold:
            drop_cols.append(col)
    logging.info("total drop cols len:{}".format(len(drop_cols)))
    return datas.drop(drop_cols,axis=1)


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


def split_dataset(feature_df:DataFrame,test_size:float):
    '''
    训练集，测试集拆分
    按照labels分布进行切分，保证切分后训练集和测试集，正负样本分布一致
    :param feature_df:
    :param test_size:
    :return:
    '''
    labels = feature_df['label']
    cols = feature_df.columns.to_list()
    cols.remove('label')
    features = feature_df[cols]
    train_x,test_x,train_y,test_y = train_test_split(features,labels,test_size=test_size,stratify=labels)
    return train_x,test_x,train_y,test_y


def log_smooth(datas:DataFrame,scale_cols:list):
    datas[scale_cols].apply(lambda e: np.log2(1 + e), axis=0)


def fill_missvalue(datas:DataFrame,fill_value=0):
    '''
    缺失值处理
    :param datas:
    :param fill_value:
    :return:
    '''
    return datas.fillna(fill_value)


def drop_duplicates(datas:DataFrame):
    '''
    删除重复值
    :param datas:
    :return:
    '''
    return datas.drop_duplicates()


def exception_val_fill(datas:DataFrame,max_val = 100,min_val=0):
    '''
    对异常值进行填充
    :param datas:
    :param max_val:
    :param min_val:
    :return:
    '''
    def replace(x):
        r = x
        if x>max_val:
            r = max_val
        if x<= min_val:
            r = min_val
        return r
    col_names:list = datas.columns.to_list()
    logging.info(col_names)
    if "user_id" in col_names:
        col_names.remove("user_id")
    if "label" in col_names:
        col_names.remove("label")
    st = time.time()
    for col in col_names:
        col_max = datas[col].max()
        col_min = datas[col].min()
        if col_max<= max_val and col_min>=min_val:continue
        datas[col] = datas[col].apply(replace)
    et = time.time()
    logging.info("exception_val_fill time spend:{} seconds".format(et-st))
    return datas
