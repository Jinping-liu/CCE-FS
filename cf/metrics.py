# -*- encoding: utf-8 -*-
import json
import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.spatial.distance import cdist
from scipy.stats import median_abs_deviation
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor as LOF

from cf.model.baseline import BlackBox


class MetricsOfCounterfactuals:
    def __init__(self, pred_model_path, raw_dataset_path, predict_samples_path, cf_path, mapping_path, transformer_path,
                 target_class: int = 1, dataset_name: str = "adult", target_name: str = "income"):
        self.dataset_name = dataset_name

        self.pred_model = None
        self.raw_dataset = None
        self.target_name = target_name

        self.predict_samples = None
        self.nor_predict_samples = None

        self.cf_list = None
        self.nor_cf_list = None

        self.transformer = None
        self.mapping = None
        self.target_class = target_class

        self._is_init = False
        if not self._is_init:
            self.load_resource(pred_model_path, raw_dataset_path, predict_samples_path, cf_path, mapping_path,
                               transformer_path)
            self._is_init = True

    def load_resource(self, pred_model_path, raw_dataset_path, predict_samples_path, cf_path, mapping_path,
                      transformer_path):#加载预测模型，原始数据集，预测样本，反事实样本，特征映射和转换器
        """
        加载模型 读取原始数据和CF数据 加载列转换器
        :return:
        """
        model = tf.keras.models.load_model(pred_model_path)
        self.pred_model = BlackBox(model)

        if raw_dataset_path.endswith(".xlsx"):
            self.raw_dataset = pd.read_excel(raw_dataset_path)
        else:
            self.raw_dataset = pd.read_csv(raw_dataset_path)

        self.predict_samples = pd.read_csv(predict_samples_path)
        self.cf_list = pd.read_csv(cf_path)

        self.transformer = joblib.load(transformer_path)

        with open(mapping_path, "r") as f:
            self.mapping = json.loads(f.read())

        # _values = self.cf_list.values.copy()
        if self.dataset_name == "german_credit":
            self.cf_list['credits_this_bank'] = self.cf_list['credits_this_bank'].astype(str)
            # self.cf_list['people_under_maintenance'] = self.cf_list['people_under_maintenance'].astype(str)

            self.predict_samples['credits_this_bank'] = self.predict_samples['credits_this_bank'].astype(str)
            # self.predict_samples['people_under_maintenance'] = self.predict_samples['people_under_maintenance'].astype(str)

        self.nor_cf_list = self.transformer.transform(self.cf_list)

        # _values = self.predict_samples.values.copy()
        # _values = _values.astype("float64")
        self.nor_predict_samples = self.transformer.transform(self.predict_samples)

        self.__split_raw_dataset()

    def get_feature_index(self):
        cont = []
        cat = []
        for index, name in enumerate(self.mapping['feature_names']):
            if name in self.mapping['continuous_feature_names']:
                cont.append(index)
            else:
                cat.append(index)
        return cont, cat

    def __split_raw_dataset(self):#分割为训练集和测试集，标准化处理
        target = self.raw_dataset[self.target_name]

        raw_train_dataset, raw_test_dataset, raw_train_target, raw_test_target = train_test_split(self.raw_dataset,
                                                                                                  target,
                                                                                                  test_size=0.2,
                                                                                                  stratify=target,
                                                                                                  random_state=17)
        y = raw_train_dataset[self.target_name]

        if self.dataset_name == "german_credit":
            raw_train_dataset['credits_this_bank'] = raw_train_dataset['credits_this_bank'].astype(str)
            # raw_train_dataset['people_under_maintenance'] = raw_train_dataset['people_under_maintenance'].astype(str)

        nor = self.transformer.transform(raw_train_dataset)
        X = pd.DataFrame(nor, columns=self.mapping['one_hot_train_sequence'], index=raw_train_dataset.index)
        X[self.target_name] = y

        self.raw_train_dataset = raw_train_dataset
        self.raw_train_target = raw_train_target

        self.nor_X = X

    def __compute_validity(self):
        """
        default k = 1
        :return:
        """
        y_prediction = self.pred_model.predict(self.nor_cf_list)
        prediction_count = sum(y_prediction == self.target_class)
        return prediction_count / self.nor_cf_list.shape[0]

    def __compute_sparsity(self):
        """
        :return:
        """
        if self.cf_list.shape != self.predict_samples.shape:
            raise Exception("cf_list.shape != predict_samples.shape")
        # 均值的计算方法
        cont_index, cat_index = self.get_feature_index()

        diff = self.cf_list.values != self.predict_samples.values

        cont_count = np.count_nonzero(diff[:, cont_index], axis=1)
        cat_count = np.count_nonzero(diff[:, cat_index], axis=1)

        # Dice的计算方式
        k = 1
        num, d = self.cf_list.shape
        sparsity_cont_list = np.zeros((num,))
        sparsity_cat_list = np.zeros((num,))
        cont_num = len(self.mapping['continuous_feature_names'])
        cat_num = d - cont_num
        for i in range(num):
            _count = cont_count[i]
            sparsity_cont_list[i] = 1 - (_count / (k * cont_num))

            _count = cat_count[i]
            sparsity_cat_list[i] = 1 - (_count / (k * cat_num))

        sparsity_cont_mean = sparsity_cont_list.mean()
        sparsity_cat_mean = sparsity_cat_list.mean()
        return sparsity_cont_mean, sparsity_cat_mean

    def __compute_lof_score(self):
        data = self.nor_X[self.mapping['one_hot_train_sequence']].values
        lof = LOF(n_neighbors=20, novelty=True, n_jobs=-1)
        lof.fit(data)

        lof_prediction = lof.predict(self.nor_cf_list)
        normal_count = (lof_prediction == 1).sum()
        normal_percentage = normal_count / self.nor_cf_list.shape[0]
        return normal_percentage

    def __compute_cont_proximity(self):
        """
        距离度量使用原始的数据，而不是经过标准化后的
        :return:
        """
        cont_index, _ = self.get_feature_index()

        # 欧式距离
        cont_distance = np.linalg.norm(self.cf_list.values[:, cont_index] - self.predict_samples.values[:, cont_index])
        return cont_distance / self.cf_list.shape[0]

    def __compute_cat_proximity(self):
        """
        等价于稀疏度中分类特征的稀疏度了
        :return:
        """
        pass

    def __compute_sparsity_degree(self):
        """
        计算本文提出的评价指标
        :return:
        """
        pass

    def __get_mads_from_training_data(self):
        """Computes Median Absolute Deviation of features."""
        mads = {}
        for feature in self.mapping.get('continuous_feature_names'):
            mads[feature] = np.median(
                abs(self.raw_train_dataset[feature].values - np.median(self.raw_train_dataset[feature].values)))
        return mads

    def __compute_mad_distance(self):
        cont_features = self.mapping.get('continuous_feature_names')
        mads = self.__get_mads_from_training_data()
        if self.dataset_name == "adult":
            mads['hours_per_week'] = 4.0

        distances = np.zeros(self.cf_list.shape[0])  # 创建一个与行数相同的数组以保存结果

        for row in range(self.cf_list.shape[0]):
            row_cf = self.cf_list.loc[row, cont_features]
            row_predict = self.predict_samples.loc[row, cont_features]
            row_mads = np.array([mads.get(i) for i in cont_features])

            dist = np.sum(np.abs(row_cf - row_predict) / row_mads) / len(cont_features)
            distances[row] = dist

        return - np.sum(distances) / self.cf_list.shape[0]

    def __compute_other_dataset_lof_score(self, dataset_path):#计算另一个数据集的LOF得分
        data = self.nor_X[self.mapping['one_hot_train_sequence']].values
        lof = LOF(n_neighbors=20, novelty=True, n_jobs=-1)
        lof.fit(data)

        dataset = pd.read_csv(dataset_path)

        type_dict = self.raw_dataset.dtypes.to_dict()
        for k, v in type_dict.items():
            if k != self.target_name:
                dataset[k] = dataset[k].astype(v)

        nor = self.transformer.transform(dataset)
        lof_prediction = lof.predict(nor)
        normal_count = (lof_prediction == 1).sum()
        normal_percentage = normal_count / nor.shape[0]
        return normal_percentage

    def compute_metrics(self, compare_dataset=None):
        validity = self.__compute_validity()
        sparsity_cont, sparsity_cat = self.__compute_sparsity()
        lof_scores = self.__compute_lof_score()
        cont_proximity = self.__compute_cont_proximity()
        cont_proximity_mad = self.__compute_mad_distance()

        if compare_dataset:
            compare_dataset_lof_score = self.__compute_other_dataset_lof_score(compare_dataset)
            return {
                "validity": validity,
                "sparsity_cont": sparsity_cont,
                "sparsity_cat": sparsity_cat,
                "lof_scores": lof_scores,
                "cont_proximity": cont_proximity,
                "cont_proximity_mad": cont_proximity_mad,
                "compare_dataset_lof_score": compare_dataset_lof_score
            }

        else:
            return {
                "validity": validity,
                "sparsity_cont": sparsity_cont,
                "sparsity_cat": sparsity_cat,
                "lof_scores": lof_scores,
                "cont_proximity": cont_proximity,
                "cont_proximity_mad": cont_proximity_mad
            }


class CompareWithCEGMFB(MetricsOfCounterfactuals):
    def __init__(self, pred_model_path, raw_dataset_path, predict_samples_path, cf_path, mapping_path, transformer_path,
                 target_class: int = 1, dataset_name: str = "adult",
                 target_name: str = "income"):
        super().__init__(pred_model_path, raw_dataset_path, predict_samples_path, cf_path, mapping_path,
                         transformer_path, target_class, dataset_name, target_name)
        self.nor_mads = self.get_mads_from_training_data(standardized=True)

    def comparewithCEGMFB(self):
        return self.compute_metrics()

    def compute_metrics(self):
        # return self.evaluate_counterfactuals()
        return self.evaluate_counterfactuals_by_scale()

    def evaluate_counterfactuals(self):
        """
        Evaluate counterfactual instances based on validity, proximity, sparsity, and distance.

        :param counterfactual_instances_file: str, CSV file path for counterfactual instances
        :return: dict, evaluation metrics including validity, proximity, sparsity, and distance
        """
        metrics = {
            'validity': 0,
            'cont-proximity': 0.0,
            'cat-proximity': 0.0,
            'sparsity': 0,
            'distance': 0.0
        }

        original_instances = self.predict_samples.values
        counterfactual_instances = self.cf_list.values

        n_instances = original_instances.shape[0]

        contIndex, catIndex = self.get_feature_index()
        mads = [10, 4]  # 直接计算出来[10,3]，但是为了与Dice保持一致，使用了4
        for orig, cf in zip(original_instances, counterfactual_instances):
            # 计算连续特征的接近度
            finalVal = 0
            for index, i in enumerate(contIndex):
                val = abs(orig[i] - cf[i])
                mad = mads[index]
                finalVal += val / mad

            metrics['cont-proximity'] += finalVal / len(contIndex)

            # 计算分类特征的接近度
            finalCount = 0
            for i in catIndex:
                if orig[i] != cf[i]:
                    finalCount += 1

            metrics['cat-proximity'] += finalCount / len(catIndex)

            # 稀疏性
            finSparsity = 0
            for i in contIndex:
                if orig[i] != cf[i]:
                    finSparsity += 1

            metrics['sparsity'] += finSparsity

            # 距离（使用L1范数）
            # 连续特征的L1距离
            metrics['distance'] += sum([abs(orig[i] - cf[i]) for i in contIndex])

        # 计算有效性
        metrics['validity'] = 1

        # 平均指标
        metrics['cat-proximity'] /= n_instances
        metrics['cont-proximity'] /= n_instances
        metrics['sparsity'] /= n_instances
        metrics['distance'] /= n_instances

        return metrics

    def get_mads_from_training_data(self, standardized=False):
        """Computes Median Absolute Deviation of features."""
        mads = {}
        for feature in self.mapping.get('continuous_feature_names'):
            if standardized:
                feature_index = self.mapping['one_hot_index'][feature]
                train_values = self.nor_predict_samples[:, feature_index]
            else:
                pass

            median = np.median(train_values)
            mad = np.median(np.abs(train_values - median))
            mads[feature] = mad
        return mads

    def evaluate_counterfactuals_by_scale(self):
        """
        Evaluate counterfactual instances based on validity, proximity, sparsity, and distance.

        :param counterfactual_instances_file: str, CSV file path for counterfactual instances
        :return: dict, evaluation metrics including validity, proximity, sparsity, and distance
        """
        metrics = {
            'validity': 0,
            'cont-proximity': 0.0,
            'cat-proximity': 0.0,
            'sparsity': 0,
            'distance': 0.0
        }

        original_instances = self.nor_predict_samples
        counterfactual_instances = self.nor_cf_list

        n_instances = original_instances.shape[0]

        contNames = self.mapping['continuous_feature_names']
        for orig, cf in zip(original_instances, counterfactual_instances):
            # 计算连续特征的接近度
            finalVal = 0

            for i in contNames:
                index = self.mapping['one_hot_index'][i]
                val = abs(orig[index] - cf[index])
                mad = self.nor_mads[i]
                finalVal += val / mad

            metrics['cont-proximity'] += finalVal / len(contNames)

            # 计算分类特征的接近度
            finalCount = 0
            for i in range(2, len(orig)):
                if orig[i] != cf[i]:
                    finalCount += 1

            metrics['cat-proximity'] += (finalCount / 27)/2

            # 稀疏性
            finSparsity = 0
            for i in range(len(orig)):
                if orig[i] != cf[i]:
                    finSparsity += 1

            metrics['sparsity'] += finSparsity

            # 距离（使用L1范数）
            # 连续特征的L1距离
            metrics['distance'] += sum([abs(orig[i] - cf[i]) for i in range(2)])

        # 计算有效性
        metrics['validity'] = 1

        # 平均指标
        metrics['cat-proximity'] /= n_instances
        metrics['cont-proximity'] /= n_instances
        metrics['sparsity'] /= n_instances
        metrics['distance'] /= n_instances

        return metrics


if __name__ == '__main__':
    adult_params = {
        "pred_model_path": "../cf/feature_select/dataset/adult/pred_model/adult_DNN.h5",
        "raw_dataset_path": "../cf/feature_select/dataset/adult/row_adult/adult.csv",
        "predict_samples_path": "../results/adult/adult_NICE/predict_samples.csv",
        "cf_path": "../results/adult/adult_NICE/counterfactual_samples.csv",
        "transformer_path": "../cf/feature_select/dataset/adult/.pkl/adult_pipeline.pkl",
        "mapping_path": "../dataset/adult/mapping.json"
    }

    statlog_params = {
        "pred_model_path": r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\statlog\pred_model\statlog_mic_DNN.h5",
        "raw_dataset_path": r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\statlog\row_statlog\statlog_mic.csv",
        "predict_samples_path": r"E:\yan\XAI\py\SFE-CF\results\statlog_mic\predict_samples_version_1_100_IWFM_count(20,20).csv",
        "cf_path": r"E:\yan\XAI\py\SFE-CF\results\statlog_mic\counterfactual_samples_version_1_100_IWFM_count(20,20).csv",
        "transformer_path": r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\statlog\.pkl\statlog_mic_pipeline.pkl",
        "mapping_path": r"E:\yan\XAI\py\SFE-CF\dataset\statlog\mapping_mic.json",
        "dataset_name": "german_credit",
        "target_name": "default"
    }

    compas_params = {
        "pred_model_path": r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\compas\pred_model\compas_DNN.h5",
        "raw_dataset_path": r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\compas\row_compas\compass.csv",
        "predict_samples_path": r"E:\yan\XAI\py\SFE-CF\results\compas\NICE\predict_samples.csv",
        "cf_path": r"E:\yan\XAI\py\SFE-CF\results\compas\NICE\counterfactual_samples.csv",
        "transformer_path": r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\compas\.pkl\compas_pipeline.pkl",
        "mapping_path": r"E:\yan\XAI\py\SFE-CF\dataset\compas\mapping.json",
        "dataset_name": "compas",
        "target_name": "two_year_recid",
        "target_class": 0
    }

    diabetes_params = {
        "pred_model_path": r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\diabetes\pred_model\diabetes_random_forest_DNN.h5",
        "raw_dataset_path": r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\diabetes\row_diabetes\diabetes_random_forest.csv",
        "predict_samples_path": r"E:\yan\XAI\py\SFE-CF\results\diabetes_random_forest\predict_samples_version_revised_1_100_IWFM_count.csv",
        "cf_path": r"E:\yan\XAI\py\SFE-CF\results\diabetes_random_forest\counterfactual_samples_version_revised_1_100_IWFM_count.csv",
        "transformer_path": r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\diabetes\.pkl\diabetes_pipeline_random_forest.pkl",
        "mapping_path": r"E:\yan\XAI\py\SFE-CF\dataset\diabetes\mapping_random_forest.json",
        "dataset_name": "diabetes",
        "target_name": "Outcome",
        "target_class": 0
    }

    breast_cancer_params = {
        "pred_model_path": "../dataset/breast_data/breast_cancer_DNN.h5",
        "raw_dataset_path": "../dataset/breast_data/fix_breast_cancer.xlsx",
        "predict_samples_path": "../dataset/breast_data/predict_samples_version_1_100_IWFM_count.csv",
        "cf_path": "../dataset/breast_data/counterfactual_samples_version_1_100_IWFM_count.csv",
        "transformer_path": "../dataset/breast_data/breast_cancer_pipeline.pkl",
        "mapping_path": "../dataset/breast_data/mapping.json",
        "dataset_name": "breast_cancer",
        "target_name": "Class",
        "target_class": 0
    }

    lending_params = {
        "pred_model_path": "../dataset/lending/lending_DNN.h5",
        "raw_dataset_path": "../dataset/lending/lending.csv",
        "predict_samples_path": "../dataset/lending/predict_samples_version_count测试.csv",
        "cf_path": "../dataset/lending/counterfactual_samples_version_count测试.csv",
        "transformer_path": "../dataset/lending/lending_pipeline.pkl",
        "mapping_path": "../dataset/lending/mapping.json",
        "dataset_name": "lending",
        "target_name": "loan_status",
        "target_class": 1
    }

    # predict_paths = [f'../dataset/diabetes/predict_samples_version_4_100_IWFM_pop{i}.csv' for i in
    #                  [10, 50, 100, 200, 300]]
    # counterfactuals_paths = [f'../dataset/diabetes/counterfactual_samples_version_4_100_IWFM_pop{i}.csv' for i in
    #                          [10, 50, 100, 200, 300]]
    # vals = []
    # for p, c in zip(predict_paths, counterfactuals_paths):
    #     diabetes_params['predict_samples_path'] = p
    #     diabetes_params['cf_path'] = c
    #     mc = MetricsOfCounterfactuals(**diabetes_params)
    #     vals.append(mc.compute_metrics())
    # print(vals)

    mc = MetricsOfCounterfactuals(**statlog_params)
    print(mc.compute_metrics())

    mc = CompareWithCEGMFB(**statlog_params)
    print(mc.comparewithCEGMFB())
