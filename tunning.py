# -*- encoding: utf-8 -*-
import warnings

warnings.filterwarnings('ignore')

import datetime
import json

import optuna
import pandas as pd
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

from base.LocalVariationalLogisticRegression import VariationalRegressionWithCF
from cf.explainer.kdtree_base import GeneratePopulationWithKmeans
from cf.mixed import CustomMixedVariableMating, CustomMixedVariableDuplicateElimination
from cf.problem.adult_mic_problem import AdultProblem


def objective(trial):
    with open("./dataset/adult/mapping.json", "r") as f:
        mapping = json.loads(f.read())

    raw_dataset = pd.read_csv("./dataset/adult/adult_all.csv")
    target = raw_dataset['income']
    raw_train_dataset, raw_test_dataset, _, _ = train_test_split(raw_dataset,
                                                                 target,
                                                                 test_size=0.2,
                                                                 stratify=target,
                                                                 random_state=1)

    dataset = pd.read_csv("cf/feature_select/dataset/adult/processed_adult/processed_adult.csv")
    target = dataset['income']
    dataset.drop(columns='income', inplace=True)
    train_dataset, test_dataset, train_target, test_target = train_test_split(dataset,
                                                                              target,
                                                                              test_size=0.2,
                                                                              stratify=target,
                                                                              random_state=1)

    vlr = VariationalRegressionWithCF()
    vlr.fit(train_dataset.values, train_target.ravel(), feature_names=train_dataset.columns.tolist())
    # bar = vlr.visual_bar(max_num=29, is_save=True, target_path="./image/Adult/")

    y_pred = vlr.predict(test_dataset.values)

    print(accuracy_score(test_target.ravel(), y_pred), precision_score(test_target.ravel(), y_pred),
          f1_score(test_target.ravel(), y_pred), recall_score(test_target.ravel(), y_pred))

    y_pred_original_dataset = raw_test_dataset.loc[[i == 0 for i in y_pred]].copy()

    n_var = len(mapping["one_hot_train_sequence"])
    xl = mapping["xl"]
    xu = mapping["xu"]

    df_original = pd.DataFrame(columns=mapping["one_hot_train_sequence"])
    df_cf = pd.DataFrame(columns=mapping["one_hot_train_sequence"])

    # 在这里定义超参数，例如种群大小、迭代次数等
    pop_size = trial.suggest_int('pop_size', 50, 1000)
    n_gen = trial.suggest_int('n_gen', 10, 200)
    n_clusters = trial.suggest_int("n_clusters", 2, 20)
    _population = trial.suggest_int("population_size", 50, 200)

    kd_tree = GeneratePopulationWithKmeans(vlr, raw_train_dataset, _population, mapping, n_clusters=n_clusters, random_state=0,
                                           pipeline_path="cf/feature_select/dataset/adult/.pkl/adult_pipeline.pkl")

    i = 0

    for index, series_data in y_pred_original_dataset.iterrows():
        i += 1

        print(f"======{i}：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}=======")
        _query_instance = series_data.to_frame()
        query_instance = pd.DataFrame(_query_instance.values.T, columns=_query_instance.index, ).astype(
            mapping['feature_dtypes'])
        del query_instance['income']

        population, query_instance_dummies = kd_tree.build_population(query_instance, 1, is_one_hot=True)

        problem = AdultProblem(
            query_instance_dummies,
            query_instance,
            vlr,
            kd_tree,
            n_var=29,
            n_obj=6,
            n_ieq_constr=3,
            n_eq_constr=6,
            xl=xl,
            xu=xu,
        )

        pop = Population.new("X", population.values)
        Evaluator().eval(problem, pop)

        algorithm = NSGA2(pop_size=pop_size, n_gen=n_gen, sampling=pop,
                          mating=CustomMixedVariableMating
                              (
                              eliminate_duplicates=CustomMixedVariableDuplicateElimination(
                                  one_hot_train_sequence=kd_tree.one_hot_train_sequence)
                          ),
                          eliminate_duplicates=CustomMixedVariableDuplicateElimination(
                              one_hot_train_sequence=kd_tree.one_hot_train_sequence)
                          )

        result = minimize(problem,
                          algorithm,
                          seed=1)
        break

    return result.F  # 返回要最小化的目标值


def main():
    # 创建Optuna试验
    print("start tunning ----- ")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)  # 指定要运行的试验次数

    # 获取最佳超参数配置
    best_params = study.best_params
    best_objective = study.best_value
    print("Best parameters:", best_params)
    print("Best objective value:", best_objective)


if __name__ == "__main__":
    main()
