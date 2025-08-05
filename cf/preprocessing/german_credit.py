# -*- encoding: utf-8 -*-
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from cf.preprocessing.base import build_feature_dict


def default_dump(obj):
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def PrepareGermanCredit(dataset_path):
    # Reading data from a csv file
    df = pd.read_csv(dataset_path)

    # todo 该数据集Dice处理时 将结果反转了 1是好 2是坏 经过处理后1是坏 0是好 现进行重新转换
    df['credits_this_bank'] = df['credits_this_bank'].astype(str)
    df['people_under_maintenance'] = df['people_under_maintenance'].astype(str)

    # df['default'] = df['default'].replace({0: 1, 1: 0})
    # cols = list(df.columns)
    # cols.append(cols.pop(0))
    # df = df[cols]
    # df.to_csv("../cf/feature_select/dataset/Statlog/german_credit_cast.csv", index=False)

    german_dtypes = df.columns.to_series().groupby(df.dtypes).groups
    german_dtypes = {k.name: v.tolist() for k, v in german_dtypes.items()}

    continuous_features = german_dtypes['int64']
    continuous_features = [feat for feat in continuous_features if feat != 'default']

    # Recognizing inputs
    target_feature = 'default'

    mapping = build_feature_dict(df, target_feature)

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(sparse=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, mapping["continuous_feature_names"]),
            ('cat', categorical_transformer, mapping["category_feature_names"]),
        ]
    )

    target = df[target_feature]
    raw_train_dataset, raw_test_dataset, raw_train_target, raw_test_target = train_test_split(df,
                                                                                              target,
                                                                                              test_size=0.2,
                                                                                              stratify=target,
                                                                                              random_state=17)

    # 对数据进行预处理
    _ = preprocessor.fit_transform(raw_train_dataset.drop(target_feature, axis=1))
    joblib.dump(preprocessor, r'/cf/feature_select/dataset/statlog\.pkl\statlog_pipeline.pkl')

    processed_df = preprocessor.transform(df.drop(target_feature, axis=1))

    # 获取编码后的离散特征列名
    cat_feature_names = list(
        preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
            mapping["category_feature_names"]))

    feature_names = mapping["continuous_feature_names"] + cat_feature_names
    mapping["one_hot_train_sequence"] = feature_names
    mapping["one_hot_test_sequence"] = feature_names

    X = pd.DataFrame(processed_df, columns=feature_names)
    X[target_feature] = target
    X.to_csv(r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\Statlog\processed_statlog.csv", index=False)

    n_var = len(feature_names)
    xl = [0] * n_var
    xu = [0] * n_var

    feature_range = mapping["feature_range"]
    for index, name in enumerate(feature_names):
        if name in continuous_features:
            xl[index] = int(min(feature_range[name]))
            xu[index] = int(max(feature_range[name]))
        else:
            xl[index] = 0
            xu[index] = 1

    mapping["xl"] = xl
    mapping["xu"] = xu

    one_hot_index = {name: index for index, name in enumerate(feature_names)}
    mapping["one_hot_index"] = one_hot_index

    with open(r"E:\yan\XAI\py\SFE-CF\dataset\statlog\mapping.json", "w") as f:
        f.write(json.dumps(mapping, default=default_dump))

    print("success")


if __name__ == '__main__':
    PrepareGermanCredit(r"/cf/feature_select/dataset/statlog/row_statlog/german_credit.csv")
