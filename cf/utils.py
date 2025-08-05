# -*- encoding: utf-8 -*-
import copy
import json
import random
from functools import lru_cache
from typing import Union

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


def get_train_dataset_quantiles(continuous_feature_names, normalized_train_df, q=0.1):#计算训练数据集中连续特征的分位数和中位数绝对偏差
    quantiles = {}
    mads = {}
    for feature in continuous_feature_names:
        quantiles[feature] = np.quantile(
            abs(list(set(normalized_train_df[feature].tolist())) - np.median(
                list(set(normalized_train_df[feature].tolist())))), q)
        mads[feature] = np.median(
            abs(normalized_train_df[feature].values - np.median(normalized_train_df[feature].values)))
    return quantiles, mads


def find_best_solution_with_post_fix(resX, x0, pred_model, scaler, target, k: int = 1, **kwargs):
    resX_copy = copy.deepcopy(resX)
    x0_copy = copy.deepcopy(x0).astype("float64")

    scaler.single_transform(resX_copy.values, operate="num")
    scaler.single_transform(x0_copy.values, operate="num")

    y_prediction = pred_model.predict(resX_copy.values)
    print("before", y_prediction)

    features_not_to_vary = kwargs.get("features_not_to_vary", None)
    continuous_feature_names = kwargs.get("continuous_feature_names")
    if features_not_to_vary:
        continuous_feature_names = [i for i in continuous_feature_names if i not in features_not_to_vary]
    normalized_train_df = kwargs.get("normalized_train_df")

    normalized_quantiles, normalized_mads = get_train_dataset_quantiles(continuous_feature_names, normalized_train_df)
    for feature in normalized_quantiles:
        normalized_quantiles[feature] = min(normalized_quantiles[feature], normalized_mads[feature])
    features_sorted = sorted(normalized_quantiles.items(), key=lambda kv: kv[1], reverse=True)
    for ix in range(len(features_sorted)):
        features_sorted[ix] = features_sorted[ix][0]

    for cf_index in range(resX.shape[0]):
        current_cf = resX_copy.iloc[[cf_index]].values
        count = 0
        for feature in features_sorted:
            current_pred = pred_model.predict(current_cf)
            if current_pred[0] != 1:
                break

            feature_index = 0 if feature == "age" else 1
            diff = x0_copy[feature][0] - current_cf[0][feature_index]
            old_diff = diff

            if (abs(diff) <= normalized_quantiles[feature]):
                change = scaler.precisions[feature]
                while (abs(diff) > 10e-4) & (np.sign(diff * old_diff) > 0) & (current_pred[0] == 1) & (count < 10000):
                    old_val = current_cf[0][feature_index]

                    current_cf[0][feature_index] += np.sign(diff) * change
                    current_pred = pred_model.predict(current_cf)

                    # 类别发生了变化，恢复到上一个备份值
                    if current_pred[0] == 0:
                        current_cf[0][feature_index] = old_val
                        resX_copy.iloc[[cf_index]] = current_cf
                        print(f"{feature} - 做出了优化")
                        break
                    else:
                        old_diff = diff
                        diff = x0_copy[feature][0] - current_cf[0][feature_index]
                        resX_copy.iloc[[cf_index]] = current_cf

                    count += 1

    y_prediction = pred_model.predict(resX_copy.values)
    print("after", y_prediction)

    filtered_idx = np.where(y_prediction == target)[0]
    filtered_nor_arr = resX_copy.loc[filtered_idx].reset_index(drop=True)

    distance = cdist(x0_copy.values, filtered_nor_arr.values, 'euclidean')[0]
    if k == 1:
        min_index = np.argmin(distance)
        return filtered_nor_arr.iloc[[min_index]].reset_index(drop=True)
    else:
        sorted_arr = np.argsort(distance)
        return filtered_nor_arr.iloc[sorted_arr[:k]].reset_index(drop=True)


def find_best_solution_with_post_fix_original_dataset_back(resX, x0, pred_model, scaler, target, k: int = 1, **kwargs):
    resX_copy = copy.deepcopy(resX)
    x0_copy = copy.deepcopy(x0).astype("float64")

    if kwargs.get("optimal", None):
        features_not_to_vary = kwargs.get("features_not_to_vary", None)
        continuous_feature_names = kwargs.get("continuous_feature_names")
        if features_not_to_vary:
            continuous_feature_names = [i for i in continuous_feature_names if i not in features_not_to_vary]

        raw_train_df = kwargs.get("normalized_train_df")

        normalized_quantiles, normalized_mads = get_train_dataset_quantiles(continuous_feature_names, raw_train_df)
        for feature in normalized_quantiles:
            normalized_quantiles[feature] = min(normalized_quantiles[feature], normalized_mads[feature])
        features_sorted = sorted(normalized_quantiles.items(), key=lambda kv: kv[1], reverse=True)
        for ix in range(len(features_sorted)):
            features_sorted[ix] = features_sorted[ix][0]

        one_hot_index = kwargs.get("ohe_index", None)
        opposite_target_class = 0 if target else 1

        for cf_index in range(resX.shape[0]):
            current_cf = resX_copy.iloc[[cf_index]].values
            count = 0
            for feature in features_sorted:
                _current_cf = scaler.single_transform(current_cf, operate="num", is_return=True)
                current_pred = pred_model.predict(_current_cf)
                if current_pred[0] != target:
                    break

                if not one_hot_index:
                    # 兼容Adult-Income
                    feature_index = 0 if feature == "age" else 1
                else:
                    feature_index = one_hot_index.get(feature)

                diff = x0_copy[feature][0] - current_cf[0][feature_index]
                old_diff = diff

                if (abs(diff) <= normalized_quantiles[feature]):
                    change = 1
                    while (abs(diff) > 10e-4) & (np.sign(diff * old_diff) > 0) & (current_pred[0] == target) & (
                            count < 100000):
                        old_val = current_cf[0][feature_index]
                        current_cf[0][feature_index] += np.sign(diff) * change

                        _current_cf = scaler.single_transform(current_cf, operate="num", is_return=True)
                        current_pred = pred_model.predict(_current_cf)

                        if current_pred[0] == opposite_target_class:
                            current_cf[0][feature_index] = old_val
                            resX_copy.iloc[[cf_index]] = current_cf
                            break
                        else:
                            old_diff = diff
                            diff = x0_copy[feature][0] - current_cf[0][feature_index]
                            resX_copy.iloc[[cf_index]] = current_cf

                        count += 1

    scaler.single_transform(resX_copy.values, operate="num")
    scaler.single_transform(x0_copy.values, operate="num")

    y_prediction = pred_model.predict(resX_copy.values)

    filtered_idx = np.where(y_prediction == target)[0]
    filtered_nor_arr = resX_copy.loc[filtered_idx].reset_index(drop=True)

    distance = cdist(x0_copy.values, filtered_nor_arr.values, 'euclidean')[0]
    if k == 1:
        min_index = np.argmin(distance)
        return filtered_nor_arr.iloc[[min_index]].reset_index(drop=True)
    else:
        sorted_arr = np.argsort(distance)
        return filtered_nor_arr.iloc[sorted_arr[:k]].reset_index(drop=True)


def find_best_solution_with_post_fix_original_dataset(resX, x0, pred_model, scaler, target, k: int = 1, **kwargs):
    resX_copy = copy.deepcopy(resX)
    x0_copy = copy.deepcopy(x0).astype("float64")

    magnification = kwargs.get('magnification', 20)

    if kwargs.get("optimal", None):
        features_not_to_vary = kwargs.get("features_not_to_vary", None)
        continuous_feature_names = kwargs.get("continuous_feature_names")
        if features_not_to_vary:
            continuous_feature_names = [i for i in continuous_feature_names if i not in features_not_to_vary]

        raw_train_df = kwargs.get("normalized_train_df")

        normalized_quantiles, normalized_mads = get_train_dataset_quantiles(continuous_feature_names, raw_train_df)
        for feature in normalized_quantiles:
            normalized_quantiles[feature] = min(normalized_quantiles[feature], normalized_mads[feature])
        features_sorted = sorted(normalized_quantiles.items(), key=lambda kv: kv[1], reverse=True)
        for ix in range(len(features_sorted)):
            features_sorted[ix] = features_sorted[ix][0]

        one_hot_index = kwargs.get("ohe_index", None)
        opposite_target_class = 0 if target else 1

        for cf_index in range(resX.shape[0]):
            current_cf = resX_copy.iloc[[cf_index]].values
            count = 0
            for feature in features_sorted:
                _current_cf = scaler.single_transform(current_cf, operate="num", is_return=True)
                current_pred = pred_model.predict(_current_cf)
                if current_pred[0] != target:
                    break

                if not one_hot_index:
                    feature_index = 0 if feature == "age" else 1
                else:
                    feature_index = one_hot_index.get(feature)

                diff = x0_copy[feature][0] - current_cf[0][feature_index]
                old_diff = diff

                # todo 在Adult-Income数据集中不需要放大30 该参数可以自由变化
                if abs(diff) <= magnification*normalized_quantiles[feature] and diff != 0:
                    while abs(diff) > 10e-4 and np.sign(diff * old_diff) > 0 and current_pred[
                        0] == target and count < 10000:
                        if abs(diff) <= 1:
                            old_val = current_cf[0][feature_index]
                            current_cf[0][feature_index] = x0_copy[feature][0]
                            _current_cf = scaler.single_transform(current_cf, operate="num", is_return=True)
                            current_pred = pred_model.predict(_current_cf)

                            if current_pred[0] == opposite_target_class:
                                print(current_cf[0][feature_index], old_val, '---')
                                current_cf[0][feature_index] = old_val
                            resX_copy.iloc[[cf_index]] = current_cf
                            break

                        old_val = current_cf[0][feature_index]
                        # 动态调整步长，保持步长最小为1，且考虑diff的正负
                        change = max(1, abs(int(diff / 2))) * np.sign(diff)
                        current_cf[0][feature_index] += change

                        _current_cf = scaler.single_transform(current_cf, operate="num", is_return=True)
                        current_pred = pred_model.predict(_current_cf)

                        if current_pred[0] == opposite_target_class:
                            current_cf[0][feature_index] = old_val
                            resX_copy.iloc[[cf_index]] = current_cf
                            break
                        else:
                            old_diff = diff
                            diff = x0_copy[feature][0] - current_cf[0][feature_index]
                            resX_copy.iloc[[cf_index]] = current_cf

                        count += 1

    scaler.single_transform(resX_copy.values, operate="num")
    scaler.single_transform(x0_copy.values, operate="num")

    y_prediction = pred_model.predict(resX_copy.values)

    filtered_idx = np.where(y_prediction == target)[0]
    filtered_nor_arr = resX_copy.loc[filtered_idx].reset_index(drop=True)

    distance = cdist(x0_copy.values, filtered_nor_arr.values, 'euclidean')[0]
    if k == 1:
        min_index = np.argmin(distance)
        return filtered_nor_arr.iloc[[min_index]].reset_index(drop=True)
    else:
        sorted_arr = np.argsort(distance)
        return filtered_nor_arr.iloc[sorted_arr[:k]].reset_index(drop=True)

# Assuming `get_train_dataset_quantiles` is a utility function you have defined elsewhere
