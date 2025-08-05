import os
import datetime
import shutil

import pandas as pd

import joblib
from scipy.stats import uniform

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from keras.callbacks import TensorBoard, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from imblearn.combine import SMOTETomek

from cf.model.baseline import BlackBox

params = {
    'RF': {
        'n_estimators': [8, 16, 32, 64, 128, 256, 512, 1024],
        'min_samples_split': [2, 0.002, 0.01, 0.05, 0.1, 0.2],
        'min_samples_leaf': [1, 0.001, 0.01, 0.05, 0.1, 0.2],
        'max_depth': [None, 2, 4, 6, 8, 10, 12, 16],
        'class_weight': [None, 'balanced'],
        'random_state': [0],
    },
    'NN': {
        'hidden_layer_sizes': [(4,), (8,), (16,), (32,), (64,), (64, 16,), (128, 64, 8,)],
        'activation': ['logistic', 'tanh', 'relu'],
        'alpha': uniform(0.001, 0.1),
        'learning_rate': ['constant'],
        'learning_rate_init': uniform(0.001, 0.1),
        'max_iter': [10000],
        'random_state': [0],
    },
    'SVM': {
        'C': uniform(0.01, 1.0),
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'gamma': uniform(0.01, 0.1),
        'coef0': uniform(0.01, 0.1),
        'class_weight': [None, 'balanced'],
        'max_iter': [10000],
        'random_state': [0],
    },
    'DNN': {
        'activation_0': ['sigmoid', 'tanh', 'relu'],
        'activation_1': ['sigmoid', 'tanh', 'relu'],
        'activation_2': ['sigmoid', 'tanh', 'relu'],
        'dim_1': [1024, 512, 256, 128, 64, 32, 16, 8, 4],
        'dim_2': [1024, 512, 256, 128, 64, 32, 16, 8, 4],
        'dropout_0': [None, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01],
        'dropout_1': [None, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01],
        'dropout_2': [None, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01],
        'optimizer': ['adam', 'rmsprop', 'sgd'],
    },
    'LGBM': {
        'boosting_type': ['gbdt'],
        'num_leaves': [4, 8, 16, 32, 64, 128],
        'max_depth': [-1, 2, 4, 6, 8, 10, 12, 16],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'n_estimators': [8, 16, 32, 64, 128, 256, 512, 1024],
        'random_state': [0],
    }
}

logboard = TensorBoard(log_dir='.logs', histogram_freq=0, write_graph=True, write_images=False, update_freq='epoch')


def build_dnn(dim_0, input_shape, dim_1, dim_2, activation_0, activation_1, activation_2, dropout_0, dropout_1,
              dropout_2,
              optimizer, loss, dim_out):
    model = Sequential()

    model.add(Dense(128, input_shape=(input_shape,), activation=activation_0))
    if dropout_0 is not None: model.add(Dropout(dropout_0))

    model.add(Dense(dim_1, activation=activation_1))
    if dropout_1 is not None: model.add(Dropout(dropout_1))

    model.add(Dense(dim_2, activation=activation_2))  # uniform, random_normal
    if dropout_2 is not None: model.add(Dropout(dropout_2))

    model.add(Dense(dim_out, activation='sigmoid'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return model


def build_logistic_regression(input_shape, ):
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', input_shape=(input_shape,)))

    # 编译模型
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_logistic_regression_by_kt(hp: kt.HyperParameters):
    model = Sequential()
    model.add(Dense(1, input_dim=29, activation='sigmoid'))

    # 编译模型
    optimizer = hp.Choice("optimizer", ['adam', 'rmsprop'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_dnn_model_by_kt(hp: kt.HyperParameters):
    model = Sequential()
    # todo input_shape需要根据数据集的不同进行修改
    # 第一层，设置输入形状
    model.add(Dense(
        units=hp.Int("units_input", min_value=4, max_value=1024, step=64),
        input_shape=(61, ),
        activation=hp.Choice("activation_input", ['sigmoid', 'tanh', 'relu'])
    ))
    model.add(Dropout(rate=hp.Float("dropout_input", min_value=0, max_value=0.75, step=0.1)))

    for i in range(hp.Int("num_hidden_layers", min_value=1, max_value=5)):
        model.add(Dense(
            units=hp.Int(f"units_{i}", min_value=4, max_value=1024, step=32),
            activation=hp.Choice(f"activation_{i}", ['sigmoid', 'tanh', 'relu'])
        ))
        model.add(Dropout(rate=hp.Float(f"dropout_{i}", min_value=0, max_value=0.75, step=0.01)))

    model.add(Dense(1, activation='sigmoid'))

    optimizer = hp.Choice("optimizer", ['adam', 'rmsprop', 'sgd'])
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    return model


def train_model(dataset_name, path_dataset, target_name, black_box="DNN", random_state=17):
    print(datetime.datetime.now(), dataset_name, black_box)
    dataset = pd.read_csv(path_dataset)
    target = dataset[target_name]
    dataset.drop(columns=target_name, inplace=True)
    train_dataset, test_dataset, train_target, test_target = train_test_split(dataset,
                                                                              target,
                                                                              test_size=0.2,
                                                                              stratify=target,
                                                                              random_state=random_state)

    # 数据集需要增强
    # oversample = SMOTETomek(random_state=0)
    # train_dataset, train_target = oversample.fit_resample(train_dataset, train_target)
    if black_box == 'DNN':
        tuner = kt.RandomSearch(
            build_dnn_model_by_kt,
            objective='val_accuracy',
            max_trials=20,  # 搜索次数
            directory=f'../../model/{dataset_name}/',  # 保存结果的目录
            project_name='dnn'  # 项目名称
        )
        early_stopping = EarlyStopping(monitor="val_loss", patience=5)
        tuner.search(x=train_dataset.values, y=train_target.ravel(), epochs=10, validation_split=0.2,
                     callbacks=[early_stopping])
        model = tuner.get_best_models(num_models=1)[0]
        # best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
        # model = tuner.hypermodel.build(best_hyperparameters)
        # model.fit(train_dataset, train_target, epochs=100)
    elif black_box == "LR":
        tuner = kt.RandomSearch(
            build_logistic_regression_by_kt,
            objective='val_accuracy',
            max_trials=20,  # 搜索次数
            executions_per_trial=3,
            directory=f'../../model/{dataset_name}/',  # 保存结果的目录
            project_name=f'{dataset_name}'  # 项目名称
        )
        early_stopping = EarlyStopping(monitor="val_loss", patience=5)
        tuner.search(x=train_dataset.values, y=train_target.ravel(), epochs=10, validation_split=0.1,
                     callbacks=[early_stopping])
        model = tuner.get_best_models(num_models=1)[0]
        # best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
        # model.fit(train_dataset, train_target, epochs=100)
    else:
        print('unknown black box %s' % black_box)
        raise Exception

    pred_model = BlackBox(model)

    if black_box in ['DNN', "LR"]:
        model.save(rf'E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\statlog\pred_model\{dataset_name}_{black_box}.h5')
    else:
        joblib.dump(model, os.path.dirname(path_dataset) + f'/{dataset_name}_{black_box}.pkl')

    y_pred_train = pred_model.predict(train_dataset.values)
    y_pred_test = pred_model.predict(test_dataset.values)

    res = {
        'dataset_name': dataset_name,
        'black_box': black_box,
        'accuracy_train': accuracy_score(train_target.ravel(), y_pred_train),
        'accuracy_test': accuracy_score(test_target.ravel(), y_pred_test),
        'f1_macro_train': f1_score(train_target.ravel(), y_pred_train, average='macro'),
        'f1_macro_test': f1_score(test_target.ravel(), y_pred_test, average='macro'),
        'f1_micro_train': f1_score(train_target.ravel(), y_pred_train, average='micro'),
        'f1_micro_test': f1_score(test_target.ravel(), y_pred_test, average='micro'),
    }

    recall = recall_score(test_target.ravel(), y_pred_test)
    precision = precision_score(test_target.ravel(), y_pred_test)
    print(res['accuracy_test'], precision, res['f1_micro_test'], recall, )

    df = pd.DataFrame(data=[res])
    columns = ['dataset_name', 'black_box', 'accuracy_train', 'accuracy_test', 'f1_macro_train', 'f1_macro_test',
               'f1_micro_train', 'f1_micro_test']
    df = df[columns]

    filename_results = rf"E:\yan\XAI\py\SFE-CF\dataset\performance\model_{dataset_name}_performance.csv"
    if not os.path.isfile(filename_results):
        df.to_csv(filename_results, index=False)
    else:
        df.to_csv(filename_results, mode='a', index=False, header=False)


if __name__ == "__main__":
    # train_model("compas", "../../dataset/compas/processed_compas.csv",
    #             target_name="two_year_recid", black_box="DNN")
    # train_model("diabetes", "../../dataset/diabetes/processed_diabetes.csv",
    #             target_name="Outcome", black_box="DNN")
    train_model("statlog_spearman", r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\statlog\processed_statlog\processed_statlog_spearman.csv",
                target_name="default", black_box='DNN')
