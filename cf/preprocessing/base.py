# -*- encoding: utf-8 -*-
import numpy as np
import joblib
import pandas as pd


def build_feature_dict(dataset, target_name: str):
    """
    build datastruct for pandas datasource
    :param dataset:
    :param target_name:
    :return:
    """
    mapping = dict()
    feature_names = dataset.columns.tolist()
    target_index = feature_names.index(target_name)
    feature_names.remove(target_name)

    feature_dtypes = dict()
    for k, v in dataset.dtypes.to_dict().items():
        feature_dtypes[k] = v.name

    mapping['feature_dtypes'] = feature_dtypes

    mapping["target_name"] = target_name
    mapping["target_index"] = target_index

    mapping["feature_names"] = feature_names

    # Adult-Income数据集
    if target_name == "income":
        mapping["features_to_vary"] = list(set(feature_names))
        mapping["features_not_to_vary"] = []

    # german credit等其他数据集
    else:
        mapping['features_to_vary'] = list(set(feature_names))
        mapping["features_not_to_vary"] = []

    df_object_col = [col for col in dataset.columns if dataset[col].dtype.name == 'object' and col != target_name]
    df_int_col = [col for col in dataset.columns if dataset[col].dtype.name != 'object' and col != target_name]

    mapping["continuous_feature_names"] = df_int_col
    mapping["category_feature_names"] = df_object_col

    mapping["continuous_feature_index"] = [i for i, name in enumerate(feature_names) if name in df_int_col]
    mapping["category_feature_index"] = [i for i, name in enumerate(feature_names) if name in df_object_col]

    range_value = dict()
    for name in df_object_col:
        range_value[name] = list(dataset[name].unique())
    for i, name in zip(mapping["continuous_feature_index"], df_int_col):
        range_value[name] = [dataset[name].min(), dataset[name].max()]

    mapping["feature_range"] = range_value
    mapping["classes"] = np.unique(dataset[target_name]).tolist()
    return mapping
