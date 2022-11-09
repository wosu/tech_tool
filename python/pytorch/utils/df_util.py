# !/usr/bin/env python
# -*- coding:utf-8 -*-
# create on 18/4/13
import os
import sys

import pandas as pd


def dataframe_explode(dataframe, fieldname):
    temp_fieldname = fieldname + '_made_tuple_'
    dataframe[temp_fieldname] = dataframe[fieldname].apply(tuple)
    list_of_dataframes = []
    for values in dataframe[temp_fieldname].unique().tolist():
        tmp_df = pd.DataFrame({
            temp_fieldname: [values] * len(values),
            fieldname: list(values), })
        list_of_dataframes.append(tmp_df)
    dataframe = dataframe[list(set(dataframe.columns) - set([fieldname]))].merge(pd.concat(list_of_dataframes), how='left', on=temp_fieldname)
    del dataframe[temp_fieldname]
    return dataframe



if __name__ =="__main__":
    # df = pd.DataFrame({'listcol':[[1,2,3,8],[4,5,6]], "aa": [222,333]})
    # print(df)
    # print(df["listcol"].apply(tuple))
    # df = dataframe_explode(df, "listcol")
    # print(df)

    test_data = [["a,b", "c", "d,e"], ["hhg,8j", "dks,dss", "0om,dss"]]
    data = pd.DataFrame(test_data)
    data.columns = ["col1","col2","col3"]
    data[["col1","col2","col3"]] = data[["col1","col2","col3"]].apply(lambda x:x.str.split(","))
    print(data)
    data = dataframe_explode(data,'col1')
    print(data)

