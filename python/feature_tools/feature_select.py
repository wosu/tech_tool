import os
from collections import OrderedDict

from pandas import DataFrame, Series
import numpy as np
import pandas as pd
from pandas.core.groupby import SeriesGroupBy
from python.pytorch.utils import df_util


def multi_fea_tgi(datas:DataFrame,label:str,fea_cols:list,explode=False,explde_column=""):

    cols = fea_cols.copy()
    cols.append(label)
    tmp_datas:DataFrame = datas[cols].copy()

    total_count = len(tmp_datas)
    # 统计正负样本数量
    label_count_dict:dict = tmp_datas[label].value_counts().to_dict()
    if explode:
        tmp_datas = df_util.dataframe_explode(tmp_datas, explde_column)
    sample_count = sum(label_count_dict.values())
    field_tgi_dict = {}
    for c in fea_cols:
        # 统计特征的覆盖数量
        fea_count_dict = tmp_datas[c].value_counts().to_dict()
        fea_cover_rate_dict = {k:v/sample_count for k,v in fea_count_dict.items()}
        tgi_dict = {}
        # 统计特征在label中的覆盖数量
        for label_val,label_count in label_count_dict.items():
            label_fea_count_dict = tmp_datas[(tmp_datas[label] == label_val)][c].value_counts()
            label_fea_tgi_dict = {}
            for label_fea, ct in label_fea_count_dict.items():
                label_fea_rate = ct/label_count
                fea_tgi = (label_fea_rate*100)/fea_cover_rate_dict.get(label_fea,1/sample_count)
                label_fea_tgi_dict[label_fea] = fea_tgi
            tgi_dict[label_val] = label_fea_tgi_dict
        field_tgi_dict[c] = tgi_dict
    return field_tgi_dict


def save_field_tgi(tgi_dict,file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    for field, field_tgi_dict in tgi_dict.items():
        tgi_df = pd.DataFrame(columns=['label', 'feature', 'tgi'])

        for label, label_tgi_dict in field_tgi_dict.items():
            label_tgis = []
            #print(label, label_tgi_dict)
            for fea, fea_tgi in label_tgi_dict.items():
                label_tgis.append([label, fea, fea_tgi])
            label_tgi_df = pd.DataFrame(label_tgis, columns=['label', 'feature', 'tgi'])
            #print(label_tgi_df)
            if tgi_df.empty:
                tgi_df = label_tgi_df.copy()
            else:
                tgi_df = pd.merge(left=tgi_df, right=label_tgi_df, left_on='feature', right_on='feature')
        tgi_cols = list(filter(lambda x: 'tgi' in str(x), tgi_df.columns.to_list()))
        tgi_df["tgi_rate"] = tgi_df[tgi_cols].max(axis=1)/tgi_df[tgi_cols].min(axis=1)
        tgi_df.to_excel('{}/{}_tgi.xlsx'.format(file_path,field), sheet_name=field, encoding='utf-8',engine='xlsxwriter')
        print(tgi_df)
        del tgi_df


def label_entropy(labels:Series):
    label_ct_dict:dict = labels.value_counts().to_dict()
    # print("label_entropy label_ct_dict,",label_ct_dict)
    sum_ct = sum(label_ct_dict.values())
    return sum([-(ct/sum_ct)*np.log(ct/sum_ct) for label,ct in label_ct_dict.items()])


def condi_entropy(datas:DataFrame,label_name:str,fea_cols:list,explode=False,explode_columns=""):
    cols = fea_cols.copy()
    cols.append(label_name)
    tmp_datas = datas[cols].copy()
    label_ct_dict:dict = tmp_datas[label_name].value_counts().to_dict()
    # print("condi_entropy label_ct_dict,",label_ct_dict)

    labels = label_ct_dict.keys()
    # print('datas_bf',tmp_datas)
    if explode:
        tmp_datas = df_util.dataframe_explode(tmp_datas, explode_columns)
        # print('datas_af', tmp_datas)

    fea_group = tmp_datas.groupby(label_name)
    field_fea_mi_dict = {}
    # 遍历label,计算当为label_i时，计算每个特征值的信息熵
    for col in fea_cols:
        # {(0, 2): 1, (1, 2): 3, (1, 5): 1}
        fea_label_ct_dict:dict = fea_group[col].value_counts().to_dict()
        # print('fea_label_ct_dict:',fea_label_ct_dict)
        fea_ct_dict = {}
        for k,v in fea_label_ct_dict.items():
            fea_i = k[1]
            ct_tmp = v
            fea_ct_dict[fea_i] = fea_ct_dict.get(fea_i,0) + ct_tmp

        feas = fea_ct_dict.keys()
        # 遍历所有特征，计算特征的信息熵
        fea_mi_dict = {}
        for fea in feas:
            fea_ct = fea_ct_dict.get(fea)
            fea_i_mi = 0
            for label in labels:
                # 计算特征i在label_i中的占比，计算label_i在特征i中的占比
                label_i_ct = label_ct_dict.get(label)
                fea_i_label_ct = fea_label_ct_dict.get((label,fea),1)
                fea_i_mi += -(fea_i_label_ct/label_i_ct) * np.log(fea_i_label_ct/fea_ct)
                # print("---",col,label,fea,fea_i_label_ct,fea_i_mi,(fea_i_label_ct/label_i_ct),fea_i_label_ct/fea_ct ,np.log(fea_i_label_ct/fea_ct))

            fea_mi_dict[fea] = fea_i_mi

        field_fea_mi_dict[col] = fea_mi_dict

    del tmp_datas
    return field_fea_mi_dict


def multi_feature_mutula_information(datas:DataFrame,label:str,fea_cols:list,explode=False,explde_column=""):
    tmp_datas = datas.copy()
    label_enp = label_entropy(datas[label])
    # 计算每个特征的条件熵
    field_fea_mi_dict = condi_entropy(datas,label,fea_cols,explode,explde_column)
    mi_dict = {}
    for field,fea_entropy_dict in field_fea_mi_dict.items():
        fea_entropy_dict = {fea:(label_enp-enp) for fea,enp in fea_entropy_dict.items()}
        mi_dict[field] = fea_entropy_dict
    del tmp_datas
    return mi_dict


def save_multi_feature_mutula_information(mi_dict,file_path):
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    for field, field_mi_dict in mi_dict.items():
        fea_mis= []
        for fea, fea_mi in field_mi_dict.items():
            fea_mis.append([fea,fea_mi])
        field_df = pd.DataFrame(fea_mis,columns=['feature', 'mi'])
        field_df.to_excel('{}/{}_mi.xlsx'.format(file_path,field), sheet_name=field, encoding='utf-8')
        print(field_df)
        del fea_mis
        del field_df


def binary_mutula_information(datas:DataFrame,label:str,fea_cols:list):
    cols = fea_cols.copy()
    cols.append(label)
    tmp_datas:DataFrame = datas[cols]
    total_count = len(tmp_datas)
    tmp_datas2 = tmp_datas.groupby(label)
    label_count_dict = tmp_datas[label].value_counts().to_dict()
    p_label_0 = label_count_dict.get(0)/total_count
    p_label_1 = label_count_dict.get(1)/total_count
    mi_dict = OrderedDict()
    for c in fea_cols:
        col_cls_dict:dict = tmp_datas[c].value_counts().to_dict()
        col_label_cls_dict = tmp_datas2[c].value_counts().to_dict()
        col_mi = 0
        for col_lable,uv in col_label_cls_dict.items():
            label = col_lable[0]
            fre = col_lable[1]
            p_x = col_cls_dict.get(fre)/total_count
            p_xy = uv/total_count
            if label == 0:
                p_y = p_label_0
            else:
                p_y = p_label_1
            col_mi += p_xy * np.log(p_xy/(p_x*p_y))
        mi_dict[c] = col_mi
    return mi_dict


def cacl_feature_cover_rate(X:DataFrame,label_name:str,explode_columns:list,file_path):
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    data_count = X.shape[0]
    label_ct_dict:dict = X[label_name].value_counts().to_dict()
    labels = set(label_ct_dict.keys())
    for col in explode_columns:
        col_df = df_util.dataframe_explode(X[[label_name, col]].copy(), col)
        col_df = col_df.groupby(label_name)
        fea_label_ct_dict = col_df[col].value_counts().to_dict()
        fea_ct_dict = {}
        for k, v in fea_label_ct_dict.items():
            fea_i = k[1]
            ct_tmp = v
            fea_ct_dict[fea_i] = fea_ct_dict.get(fea_i, 0) + ct_tmp
        col_fea_rate_list = []
        fea_cols_list = ['feature', 'feature_count', 'sample_count', 'fea_cover_rate']
        fea_cols_list.extend([l+"_fea_cover_rate" for l in labels])

        for fea, ct in fea_ct_dict.items():
            fea_cover_rate = ct/data_count
            fea_cover_list = [fea,ct,data_count,fea_cover_rate]
            for label in labels:
                label_i_ct = label_ct_dict.get(label)
                fea_i_label_ct = fea_label_ct_dict.get((label, fea), 1)
                fea_label_cover_rate = fea_i_label_ct / label_i_ct
                fea_cover_list.append(fea_label_cover_rate)

            col_fea_rate_list.append(fea_cover_list)
        rate_df = pd.DataFrame(col_fea_rate_list,columns=fea_cols_list)
        print(rate_df)
        rate_df.to_excel('{}/{}_cover_rate.xlsx'.format(file_path,col), sheet_name=col, encoding='utf-8')


def mRMR():
    '''
    最大相关最小冗余算法
    :return:
    '''
    pass


def three_sigma(datas:DataFrame):
    mean = datas.mean()
    std = datas.std()
    lower,upper = mean-3*std,mean+3*std
    for col in datas.columns.tolist():
        excp = datas[(datas[col]>upper[col]) | (datas[col]<lower[col])]
        excp = excp[col].sort_values()
        print('\n========',col)
        print(excp)


def tgi(datas:DataFrame,label:str,fea_cols:list):

    cols = fea_cols.copy()
    cols.append(label)
    tmp_datas:DataFrame = datas[cols]

    total_count = len(tmp_datas)
    # 统计正负样本数量
    label_count_dict = tmp_datas[label].value_counts().to_dict()
    tgi_dict = OrderedDict()
    for c in fea_cols:
        # 统计特征在正样本中覆盖的数量
        fea_pos_count = tmp_datas[(tmp_datas['label']==1) & (tmp_datas[c]>0)][c].count()
        pos_count = label_count_dict.get(1)
        # 计算特征在正样本中的占比
        fea_pos_rate = fea_pos_count/pos_count

        # 统计特征在负样本中覆盖的数量
        fea_neg_count = tmp_datas[(tmp_datas['label']==0) & (tmp_datas[c]>0)][c].count()
        neg_count = label_count_dict.get(0)
        # 计算特征在负样本中的占比
        fea_neg_rate = fea_neg_count/neg_count

        # 计算特征在所有样本中的占比
        fea_rate = (fea_neg_count+fea_pos_count)/(pos_count+neg_count)

        # 计算特征与正样本的tgi 计算特征与负样本的tgi
        pos_tgi = fea_pos_rate*100/fea_rate
        neg_tgi = fea_neg_rate*100/fea_rate
        tgi_dict[c] = [pos_tgi,neg_tgi]
    return tgi_dict


def binary_mutula_information(datas:DataFrame,label:str,fea_cols:list):
    cols = fea_cols.copy()
    cols.append(label)
    tmp_datas:DataFrame = datas[cols]
    total_count = len(tmp_datas)
    tmp_datas2 = tmp_datas.groupby(label)
    label_count_dict = tmp_datas[label].value_counts().to_dict()
    p_label_0 = label_count_dict.get(0)/total_count
    p_label_1 = label_count_dict.get(1)/total_count
    mi_dict = OrderedDict()
    for c in fea_cols:
        col_cls_dict:dict = tmp_datas[c].value_counts().to_dict()
        col_label_cls_dict = tmp_datas2[c].value_counts().to_dict()
        col_mi = 0
        for col_lable,uv in col_label_cls_dict.items():
            label = col_lable[0]
            fre = col_lable[1]
            p_x = col_cls_dict.get(fre)/total_count
            p_xy = uv/total_count
            if label == 0:
                p_y = p_label_0
            else:
                p_y = p_label_1
            col_mi += p_xy * np.log(p_xy/(p_x*p_y))
        mi_dict[c] = col_mi
    return mi_dict



if __name__ == "__main__":
    a = [
        [1,2,3],
        [0,2,4],
        [1,5,6],
        [1,2,3],
        [1, 2, 3]
    ]
    df = pd.DataFrame(a,columns=['label','fea1','fea2'])
    print(df)
    label_se:Series = df['label']
    label_se.value_counts()
    print(label_se.value_counts().to_dict())
    print(label_entropy(label_se))
    df_group = df.groupby('label')
    fea_ser:SeriesGroupBy = df_group['fea1']
    print(fea_ser.value_counts().to_dict())
    print(set(i[1] for i in fea_ser.value_counts().to_dict().keys()))

    enp_dict = condi_entropy(df,'label',['fea1','fea2'])
    print(enp_dict)
    mi_dict = multi_feature_mutula_information(df,'label',['fea1','fea2'])
    print(mi_dict)