# -*- encoding: utf-8 -*-
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#from cf.preprocessing.base import build_feature_dict

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


def PrepareDiabetes(dataset_path):
    # Reading data from a csv file
    df = pd.read_csv(dataset_path)
    columns_to_convert = ["Pregnancies", "Glucose", "SkinThickness", "Insulin", "Age"]
    df[columns_to_convert] = df[columns_to_convert].astype(int)

    columns_to_fill_median = ["Glucose", "BMI"]
    for column in columns_to_fill_median:
        if column == "BloodPressure":
            df[column].replace(0, np.NaN, inplace=True)
            mean_diabetes = df[df['Outcome'] == 1][column].median()
            mean_no_diabetes = df[df['Outcome'] == 0][column].median()

            df.loc[(df[column].isnull()) & (df['Outcome'] == 1), column] = mean_diabetes
            df.loc[(df[column].isnull()) & (df['Outcome'] == 0), column] = mean_no_diabetes
        else:
            df[column].replace(0, df[column].mean(), inplace=True)

    columns_to_fill_regression = ["SkinThickness", "Insulin"]

    for column in columns_to_fill_regression:
        df[column].replace(0, np.NaN, inplace=True)

        df_with_insulin = df.dropna(subset=[column])
        df_missing_insulin = df[df[column].isnull()]
        if column == "SkinThickness":
            features_for_regression = ['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction',
                                       'Age']
        else:
            features_for_regression = ['Pregnancies', 'Glucose', 'BMI', 'DiabetesPedigreeFunction',
                                       'Age',
                                       'SkinThickness']

        target_feature_regression = column
        X_train, X_test, y_train, y_test = train_test_split(
            df_with_insulin[features_for_regression], df_with_insulin[target_feature_regression], test_size=0.2,
            random_state=17
        )

        # 训练回归模型
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 预测缺失值
        predictions_regression = model.predict(df_missing_insulin[features_for_regression])

        # 将预测值填充回原始数据集
        df.loc[df[column].isnull(), column] = predictions_regression

    # Recognizing inputs
    target_feature = 'Outcome'

    cols = list(df.columns)
    if cols[0] == target_feature:
        cols.append(cols.pop(0))
        df = df[cols]

    df[columns_to_convert] = df[columns_to_convert].astype(int)
    _dtypes = df.columns.to_series().groupby(df.dtypes).groups
    _dtypes = {k.name: v.tolist() for k, v in _dtypes.items()}

    continuous_features = _dtypes['int64'] + _dtypes['float64']
    continuous_features = [feat for feat in continuous_features if feat != target_feature]

    mapping = build_feature_dict(df, target_feature)

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, mapping["continuous_feature_names"]),
        ]
    )
    print(mapping["continuous_feature_names"])

    target = df[target_feature]
    raw_train_dataset, raw_test_dataset, raw_train_target, raw_test_target = train_test_split(df,
                                                                                              target,
                                                                                              test_size=0.2,
                                                                                              stratify=target,
                                                                                              random_state=17)

    # 对数据进行预处理
    _ = preprocessor.fit_transform(raw_train_dataset.drop(target_feature, axis=1))
    joblib.dump(preprocessor, r'E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\diabetes\.pkl\diabetes_mic_pipeline.pkl')

    processed_df = preprocessor.transform(df.drop(target_feature, axis=1))

    feature_names = mapping["continuous_feature_names"]
    mapping["one_hot_train_sequence"] = feature_names
    mapping["one_hot_test_sequence"] = feature_names

    X = pd.DataFrame(processed_df, columns=feature_names)
    X[target_feature] = target
    X.to_csv(r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\diabetes\processed_diabetes\processed_diabetes_mic.csv", index=False)

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

    with open("../../dataset/diabetes/mapping_mic.json", "w") as f:
        f.write(json.dumps(mapping, default=default_dump))

    print("success")
    df.to_csv(r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\diabetes\processed_diabetes\diabetes_mic_fill.csv", index=False)


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
    PrepareDiabetes(r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\diabetes\row_diabetes\diabetes_mic.csv")
    # compute_mic(r'E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\diabetes\processed_diabetes\processed_diabetes.csv',
                # r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\diabetes\feature\mic.csv")