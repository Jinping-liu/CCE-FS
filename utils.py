# -*- encoding: utf-8 -*-
"""
@File    :   utils.py    
@Contact :   1053522308@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/4/26 18:45   wuxiaoqiang      1.0         None
"""
import os.path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch


class ConvertToSKLearnDataSet:
    def __init__(self, path: str):
        self.path = os.path.split(path)[0]

        if path.endswith(".csv"):
            self.df = pd.read_csv(path)
        elif path.endswith(".xlsx"):
            self.df = pd.read_excel(path)
        else:
            raise Exception("file type error")

    def convert_data_set(self):
        data_set = Bunch()

        data_set.data = self._get_feature()
        data_set.target = self._get_target()
        data_set.DESCR = self._get_descr()
        data_set.feature_names = self._get_feature_names()
        data_set.target_names = self._get_target_names()

        return data_set

    def _get_feature(self):
        """
        获取数据集特征值
        :return:
        """
        data_feature = self.df.iloc[:, 1:]
        data_np = np.array(data_feature)
        return data_np

    def _get_target(self):
        """
        获取数据集目标值
        :return:
        """
        data_target = self.df.iloc[:, 0]
        data_np = np.array(data_target)
        return data_np

    def _get_descr(self):
        """
        获取数据集描述
        :return:
        """
        text = "样本数量：{}；" \
               "特征数量：{}；目标值数量：{}；无缺失数据" \
               "".format(self.df.index.size, self.df.columns.size, 1)
        return text

    def _get_feature_names(self):
        """
        获取特征名字
        :return:
        """
        fnames = list(self.df.columns.values)
        fnames.pop(0)
        return fnames

    def _get_target_names(self):
        """
        获取目标值名称
        :return:
        """
        tnames = list(self.df.columns.values)[0]
        return tnames


def load_dataset(path: str):
    obj = ConvertToSKLearnDataSet(path)
    return obj.convert_data_set()


def normalized(train, test):
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    return scaler, train, test
