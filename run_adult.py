# -*- encoding: utf-8 -*-
import datetime
import json
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import tensorflow as tf
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.algorithms.moo.nsga2 import binary_tournament
from pymoo.util.display.multi import MultiObjectiveOutput
from sklearn.model_selection import train_test_split

from cf.explainer.kdtree_base import GeneratePopulationWithKmeans
from cf.mixed import CustomMixedVariableMating, CustomMixedVariableDuplicateElimination, CustomRankAndCrowdingSurvival
from cf.problem.adult_mic_problem import AdultProblemAVarConstVersion1
from cf.utils import find_best_solution_with_post_fix_original_dataset, \
    find_best_solution_with_post_fix_original_dataset_back
from cf.model.baseline import BlackBox

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

if __name__ == '__main__':
    with open("./dataset/adult/mapping.json", "r") as f:
        mapping = json.loads(f.read())

    raw_dataset = pd.read_csv(r"\dataset\dataset\adult\row_adult\adult.csv")
    target = raw_dataset['income']
    raw_train_dataset, raw_test_dataset, _, _ = train_test_split(raw_dataset,
                                                                 target,
                                                                 test_size=0.2,
                                                                 stratify=target,
                                                                 random_state=17)

    dataset = pd.read_csv(r"dataset\dataset\adult\processed_adult\processed_adult.csv")
    target = dataset['income']
    dataset.drop(columns='income', inplace=True)
    train_dataset, test_dataset, train_target, test_target = train_test_split(dataset,
                                                                              target,
                                                                              test_size=0.2,
                                                                              stratify=target,
                                                                              random_state=17)

    # load model
    model = tf.keras.models.load_model(r'dataset\dataset\adult\pred_model\adult_DNN.h5')

    # eval feature importance
    black_model = BlackBox(model)
    black_model.evaluation_importance(train_dataset, train_target, target='income', is_show=False)
    prediction = black_model.predict(test_dataset.values)

    y_pred_original_dataset = raw_test_dataset.loc[(prediction == 0) & (raw_test_dataset['income'] == 0)]
    y_pred_original_dataset = y_pred_original_dataset.sample(n=100)

    n_var = len(mapping["one_hot_train_sequence"])
    xl = mapping["xl"]
    xu = mapping["xu"]

    df_original = pd.DataFrame(columns=mapping["feature_names"])
    df_cf = pd.DataFrame(columns=mapping["feature_names"])

    k = 1
    kd_tree = GeneratePopulationWithKmeans(black_model, raw_train_dataset, 500, mapping, n_clusters=50,
                                           random_state=0,
                                           pipeline_path=r"E:\yan\XAI\py\SFE-CF\cf\feature_select\dataset\adult\.pkl\adult_pipeline.pkl")

    i = 0
    is_random = False
    repeat_count = 0

    for index, series_data in y_pred_original_dataset.iterrows():
        i += 1
        while True and repeat_count <= 2:
            print(f"======{i}：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}=======")
            _query_instance = series_data.to_frame()
            query_instance = pd.DataFrame(_query_instance.values.T, columns=_query_instance.index, ).astype(
                mapping['feature_dtypes'])
            raw_instance_target = query_instance['income'].values[0]
            del query_instance['income']

            population, query_instance_dummies = kd_tree.build_population(query_instance, 1, True, is_random)

            problem = AdultProblemAVarConstVersion1(
                query_instance_dummies,
                query_instance,
                black_model,
                kd_tree,
                n_var=44,#变量数
                n_obj=5,#目标数
                n_ieq_constr=0,#不等式约束
                n_eq_constr=8,#等式约束
                xl=xl,
                xu=xu,
            )

            pop = Population.new("X", population.values)
            Evaluator().eval(problem, pop)

            algorithm = NSGA2(pop_size=500, sampling=pop,
                              selection=TournamentSelection(func_comp=binary_tournament),
                              survival=CustomRankAndCrowdingSurvival(),
                              output=MultiObjectiveOutput(),
                              mating=CustomMixedVariableMating
                                  (
                                  eliminate_duplicates=CustomMixedVariableDuplicateElimination(
                                      one_hot_train_sequence=kd_tree.one_hot_train_sequence)
                              ),
                              eliminate_duplicates=CustomMixedVariableDuplicateElimination(
                                  one_hot_train_sequence=kd_tree.one_hot_train_sequence)
                              )

            res = minimize(problem,
                           algorithm,
                           ('n_gen', 2),
                           seed=1)

            try:
                X = res.X
                if X is None:
                    raise Exception(f"{i}解决方案为None")

                X = pd.DataFrame(data=X, columns=kd_tree.one_hot_train_sequence)
                cf_instance = find_best_solution_with_post_fix_original_dataset(
                    X, query_instance_dummies, black_model, kd_tree, 1, k=k,
                    continuous_feature_names=mapping.get("continuous_feature_names"),
                    normalized_train_df=raw_train_dataset,
                    features_not_to_vary=mapping.get("features_not_to_vary"),
                    ohe_index=mapping.get("one_hot_index"),
                    optimal=True,
                )

                if not len(cf_instance):
                    raise Exception(f"{i}符合预期的解决方案为None({len(X)})")

                cf_mixed = kd_tree.reverse_transform(cf_instance.values)
                cf_instance["age"] = cf_mixed["age"]
                cf_instance["hours_per_week"] = cf_mixed["hours_per_week"]


                print(f"raw target: {raw_instance_target}，cf target: {1} \n")
                print(f"q: {query_instance}")
                print(f"c: {cf_mixed[query_instance.columns.tolist()]} \n")

                if isinstance(cf_mixed, pd.DataFrame):
                    df_original = pd.concat([df_original, query_instance], ignore_index=True)
                    df_cf = pd.concat([df_cf, cf_mixed], ignore_index=True)

            except Exception as e:
                if repeat_count < 2:
                    repeat_count += 1
                    is_random = True
                else:
                    del problem
                    del algorithm
                    del res

                    repeat_count = 0
                    is_random = False
                    break
            else:
                del problem
                del algorithm
                del res

                repeat_count = 0
                is_random = False
                break

    df_original.to_csv(f"./results/adult/adult_mic/predict_samples.csv", index=False)
    df_cf.to_csv(f"./results/adult/adult_mic/counterfactual_samples.csv", index=False)
