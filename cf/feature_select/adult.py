import json
import os
import shutil
import zipfile
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from cf.preprocessing.base import build_feature_dict

from minepy import MINE

from sklearn.linear_model import LassoCV


def load_adult_income_dataset(is_train: bool = True, is_save: bool = False):
    #加载数据集，替换特征，选择特征，删除压缩包，保存返回
    """Loads adult income dataset from https://archive.ics.uci.edu/ml/datasets/Adult and prepares
       the data for data analysis based on https://rpubs.com/H_Zhu/235617

    :return adult_data: returns preprocessed adult income dataset.
    """
    # Download the adult dataset from https://archive.ics.uci.edu/static/public/2/adult.zip as a zip folder
    outdirname = 'adult'
    zipfilename = outdirname + '.zip'
    urlretrieve('https://archive.ics.uci.edu/static/public/2/adult.zip', zipfilename)
    with zipfile.ZipFile(zipfilename, 'r') as unzip:
        unzip.extractall(outdirname)

    if is_train:
        raw_data = np.genfromtxt(outdirname + '/adult.data',
                                 delimiter=', ', dtype=str, invalid_raise=False)
    else:
        raw_data = np.genfromtxt(outdirname + "/adult.test",
                                 delimiter=", ", dtype=str, skip_header=1, invalid_raise=False)

    #  column names from "https://archive.ics.uci.edu/ml/datasets/Adult"
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week',
                    'native-country',
                    'income']

    adult_data = pd.DataFrame(raw_data, columns=column_names)

    # For more details on how the below transformations are made, please refer to https://rpubs.com/H_Zhu/235617
    adult_data = adult_data.astype({"age": np.int64,  "hours-per-week": np.int64, "capital-gain": np.int64,
                                    "capital-loss": np.int64, "educational-num": np.int64, "fnlwgt": np.int64})

    adult_data = adult_data.replace({'workclass': {'Without-pay': 'Other/Unknown', 'Never-worked': 'Other/Unknown'}})
    adult_data = adult_data.replace({'workclass': {'Federal-gov': 'Government', 'State-gov': 'Government',
                                                   'Local-gov': 'Government'}})
    adult_data = adult_data.replace(
        {'workclass': {'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed'}})
    adult_data = adult_data.replace({'workclass': {'Never-worked': 'Self-Employed', 'Without-pay': 'Self-Employed'}})
    adult_data = adult_data.replace({'workclass': {'?': 'Other/Unknown'}})

    adult_data = adult_data.replace(
        {
            'occupation': {
                'Adm-clerical': 'White-Collar', 'Craft-repair': 'Blue-Collar',
                'Exec-managerial': 'White-Collar', 'Farming-fishing': 'Blue-Collar',
                'Handlers-cleaners': 'Blue-Collar',
                'Machine-op-inspct': 'Blue-Collar', 'Other-service': 'Service',
                'Priv-house-serv': 'Service',
                'Prof-specialty': 'Professional', 'Protective-serv': 'Service',
                'Tech-support': 'Service',
                'Transport-moving': 'Blue-Collar', 'Unknown': 'Other/Unknown',
                'Armed-Forces': 'Other/Unknown', '?': 'Other/Unknown'
            }
        }
    )

    adult_data = adult_data.replace({'marital-status': {'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married',
                                                        'Married-spouse-absent': 'Married', 'Never-married': 'Single'}})

    adult_data = adult_data.replace({'race': {'Black': 'Other', 'Asian-Pac-Islander': 'Other',
                                              'Amer-Indian-Eskimo': 'Other'}})

    adult_data = adult_data.replace({'native-country': {'?': 'Other/Unknown', 'Cuba': 'Other/Unknown',
                                                        'England': 'Other/Unknown', 'China': 'Other/Unknown', 'South': 'Other/Unknown', 'Jamaica': 'Other/Unknown',
                                                        'Italy': 'Other/Unknown', 'Dominican-Republic': 'Other/Unknown', 'Japan': 'Other/Unknown', 'Guatemala': 'Other/Unknown',
                                                        'Poland': 'Other/Unknown', 'Vietnam': 'Other/Unknown', 'Columbia': 'Other/Unknown', 'Haiti': 'Other/Unknown',
                                                        'Portugal': 'Other/Unknown', 'Taiwan': 'Other/Unknown', 'Iran': 'Other/Unknown', 'Greece': 'Other/Unknown',
                                                        'Nicaragua': 'Other/Unknown', 'Peru': 'Other/Unknown', 'Ecuador': 'Other/Unknown', 'France': 'Other/Unknown',
                                                        'Ireland': 'Other/Unknown', 'Hong': 'Other/Unknown', 'Thailand': 'Other/Unknown', 'Cambodia': 'Other/Unknown',
                                                        'Trinadad&Tobago': 'Other/Unknown', 'Laos': 'Other/Unknown', 'Yugoslavia': 'Other/Unknown', 'Outlying-US(Guam-USVI-etc)': 'Other/Unknown',
                                                        'Scotland': 'Other/Unknown', 'Honduras': 'Other/Unknown', 'Hungary': 'Other/Unknown', 'Holand-Netherlands': 'Other/Unknown'}})

    adult_data = adult_data.drop(columns=['fnlwgt'])

    adult_data = adult_data.drop(columns=['capital-gain'])
    adult_data = adult_data.drop(columns=['capital-loss'])

    adult_data = adult_data.drop(columns=['educational-num'])

    if is_train:
        adult_data = adult_data.replace({'income': {'<=50K': 0, '>50K': 1}})
    else:
        adult_data = adult_data.replace({'income': {'<=50K.': 0, '>50K.': 1}})

    adult_data = adult_data.replace({'education': {'Assoc-voc': 'Assoc', 'Assoc-acdm': 'Assoc',
                                                   '11th': 'School', '10th': 'School', '7th-8th': 'School',
                                                   '9th': 'School', '12th': 'School', '5th-6th': 'School',
                                                   '1st-4th': 'School', 'Preschool': 'School'}})

    adult_data = adult_data.rename(columns={'marital-status': 'marital_status', 'hours-per-week': 'hours_per_week', 'native-country': 'native_country'})

    # Remove the downloaded dataset 如果数据集已被成功加载和保存，删除下载的数据集文件夹以清理空间。
    if os.path.isdir(outdirname):
        entire_path = os.path.abspath(outdirname)
        shutil.rmtree(entire_path)

    #选特征
    #adult_data = adult_data[['education', 'occupation', 'workclass', 'relationship', 'gender', 'native_country', 'income']]

    if is_train:
        if is_save:
            adult_data.to_csv(r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\adult\adult.csv", index=False)
    else:
        if is_save:
            adult_data.to_csv(r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\adult\adult_test.csv", index=False)
    return adult_data


def PrepareAdult(file_path: str):
    ## Reading data from a csv file
    data = pd.read_csv(file_path)

    target_feature = 'income'
    mapping = build_feature_dict(data, target_feature)

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(sparse=False))])

    # 列变换器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, mapping["continuous_feature_names"]),
            ('cat', categorical_transformer, mapping["category_feature_names"])]
    )

    # 对数据进行预处理
    target = data[target_feature]
    raw_train_dataset, raw_test_dataset, raw_train_target, raw_test_target = train_test_split(data,
                                                                                              target,
                                                                                              test_size=0.2,
                                                                                              stratify=target,
                                                                                              random_state=17)

    _ = preprocessor.fit_transform(raw_train_dataset.drop(target_feature, axis=1))
    joblib.dump(preprocessor, 'dataset/adult/.pkl/adult_pipeline.pkl')

    processed_adult = preprocessor.transform(data.drop(target_feature, axis=1))

    # 获取编码后的离散特征列名
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
        input_features=mapping["category_feature_names"])

    # 合并连续特征和编码后的离散特征列名
    feature_names = mapping["continuous_feature_names"] + list(cat_feature_names)
    mapping["one_hot_train_sequence"] = feature_names
    mapping["one_hot_test_sequence"] = feature_names

    processed_adult_target = data[target_feature]
    processed_adult_df = pd.DataFrame(processed_adult, columns=feature_names)
    processed_adult_df[target_feature] = processed_adult_target
    processed_adult_df.to_csv("../../dataset/adult/processed_adult.csv", index=False)

    n_var = len(feature_names)
    xl = [0] * n_var
    xu = [0] * n_var

    feature_range = mapping["feature_range"]
    for index, name in enumerate(feature_names):
        if name in mapping["continuous_feature_names"]:
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


    with open("../../dataset/adult/mapping.json", "w") as f:
        f.write(json.dumps(mapping))
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
    train_data = load_adult_income_dataset(is_train=True, is_save=True)
    PrepareAdult(r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\adult\adult.csv")
    #compute_embedded_L1_importance(r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\adult\processed_adult.csv", r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\adult\feature\embedded_L1_importance.csv")