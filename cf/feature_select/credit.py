# -*- encoding: utf-8 -*-
import json

import joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

from cf.preprocessing.base import build_feature_dict


def PrepareCreditCardDefault(dataset_path):
    ## Reading data from a csv file
    df = pd.read_csv(dataset_path, delimiter=',')

    continuous_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5',
                           'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    discrete_features = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    ## Recognizing inputs
    target_feature = 'default payment next month'

    df[discrete_features] = df[discrete_features].astype(str)

    df = df.replace(
        {"SEX": {"1": "Male", "2": "Female"}}
    )
    df = df.replace(
        {"EDUCATION": {"0": "Other/Unknown", "1": "Graduate", "2": "University", "3": "High_School",
                       "4": "Other/Unknown", "5": "Other/Unknown", "6": "Other/Unknown"}}
    )

    df = df.replace(
        {"MARRIAGE": {"0": "others", "1": "married", "2": "single", "3": "others"}}
    )

    mapping = build_feature_dict(df, target_feature)

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(sparse=False))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, mapping["continuous_feature_names"]),
            ('cat', categorical_transformer, mapping["category_feature_names"]),
        ]
    )

    # 对数据进行预处理
    X = preprocessor.fit_transform(df.drop(target_feature, axis=1))
    y = df[target_feature]

    joblib.dump(preprocessor, r'E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\credit\.pkl\credit_pipeline.pkl')

    # 获取编码后的离散特征列名
    cat_feature_names = list(
        preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
            mapping["category_feature_names"]))

    feature_names = mapping["continuous_feature_names"] + cat_feature_names
    mapping["one_hot_train_sequence"] = feature_names
    mapping["one_hot_test_sequence"] = feature_names

    n_var = len(mapping.get("feature_names"))
    xl = [0] * n_var
    xu = [0] * n_var

    feature_range = mapping["feature_range"]
    for index, name in enumerate(mapping.get("feature_names")):
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

    def convert_to_serializable(obj):
        if isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    mapping = convert_to_serializable(mapping)

    X = pd.DataFrame(X, columns=feature_names)
    X[target_feature] = y
    X.to_csv(r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\credit\processed_credit\processed_credit.csv", index=False)

    with open("../../dataset/credit/mapping.json", "w") as f:
        f.write(json.dumps(mapping))
    print("----")


if __name__ == '__main__':
    PrepareCreditCardDefault(r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\credit\row_credit\credit_raw.csv")
