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

from sklearn.ensemble import RandomForestClassifier

from cf.preprocessing.base import build_feature_dict

from minepy import MINE

from sklearn.linear_model import LassoCV


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
    # df.to_csv("../../dataset/statlog+german+credit+data/german_credit_cast.csv", index=False)

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
    joblib.dump(preprocessor, r'E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\statlog\.pkl\statlog_pipeline_spearman.pkl')

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
    X.to_csv(r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\statlog\processed_statlog\processed_statlog_spearman.csv", index=False)

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

    with open("E:\yan\XAI\py\SFE-CF\dataset\statlog\mapping_spearman.json", "w") as f:
        f.write(json.dumps(mapping, default=default_dump))

    print("success")

def compute_mic(file_path1: str, file_path2: str):
    df = pd.read_csv(file_path1)

    features = df.columns[:-1]
    target = df.columns[-1]

    mine = MINE(alpha=0.6, c=15)
    mic_values = {}

    for feature in features:
        mine.compute_score(df[feature], df[target])
        mic_values[feature] = mine.mic()

    mic_df = pd.DataFrame(list(mic_values.items()), columns=['Feature', 'MIC'])
    mic_df = mic_df.sort_values(by='MIC', ascending=False)

    mic_df.to_csv(file_path2, index=False)

    print("done")

def compute_peason(file_path1: str, file_path2: str):

    df = pd.read_csv(file_path1)

    features = df.columns[:-1]
    target = df.columns[-1]

    correlations = df[features].corrwith(df[target])

    results = correlations.reset_index()
    results.columns = ['Feature', 'Pearson']
    results['Absolute Pearson'] = results['Pearson'].abs()

    results = results.sort_values(by='Absolute Pearson', ascending=False)

    results.to_csv(file_path2, index=False)

    print("done")

def compute_random_forest_importance(file_path1: str, file_path2: str):

    df = pd.read_csv(file_path1)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    feature_importances = rf.feature_importances_
    feature_names = X.columns
    feature_importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

    feature_importances_df.to_csv(file_path2, index=False)

    print("done")

def compute_spearman(file_path1: str, file_path2: str):
    df = pd.read_csv(file_path1)

    features = df.columns[:-1]
    target = df.columns[-1]

    spearman_corr = df[features].corrwith(df[target], method='spearman')

    results = spearman_corr.reset_index()
    results.columns = ['Feature', 'Spearman']
    results['Absolute Spearman'] = results['Spearman'].abs()

    results = results.sort_values(by='Absolute Spearman', ascending=False)

    results.to_csv(file_path2, index=False)

    print("done")

def compute_embedded_L1_importance(file_path1: str, file_path2: str):

    df = pd.read_csv(file_path1)

    features = df.columns[:-1]
    target = df.columns[-1]

    lasso = LassoCV(cv=5, random_state=42)

    lasso.fit(df[features], df[target])

    importances = lasso.coef_
    importances_abs = np.abs(importances)

    results = pd.DataFrame({
        'Feature': features,
        'Importance': importances,
        'Absolute Importance': importances_abs
    })

    results = results.sort_values(by='Absolute Importance', ascending=False)

    results.to_csv(file_path2, index=False)

    print("done")


if __name__ == '__main__':
    PrepareGermanCredit(r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\statlog\row_statlog\statlog_spearman.csv")
    #compute_spearman(r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\statlog\processed_statlog\processed_statlog.csv",
               #r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\statlog\feature\spearman.csv")