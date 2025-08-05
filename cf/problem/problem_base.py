# -*- encoding: utf-8 -*-
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import lru_cache

import numpy as np
from sklearn.metrics import log_loss
from pymoo.core.problem import ElementwiseProblem


class DistanceProblemVersion5(ElementwiseProblem, ABC):
    def __init__(self, x1, raw_x1, pred_model, kd_tree_model, n_var, n_obj, n_ieq_constr, n_eq_constr, xl, xu,
                 **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, n_ieq_constr=n_ieq_constr, n_eq_constr=n_eq_constr, xl=xl, xu=xu,
                         elementwise_evaluation=True,
                         **kwargs)
        self.x1 = x1
        self.pred_model = pred_model
        self.kd_tree = kd_tree_model
        self.raw_x1 = raw_x1

    def label_encode_proxy(self, cfs=None):
        if cfs:
            return self.kd_tree.label_encode(cfs)
        else:
            self.label_encode_x1 = self.kd_tree.label_encode(self.x1).values[0].tolist()

    def label_decode_proxy(self, cfs):
        return self.kd_tree.label_decode(cfs)

    def get_cfs_dummies_proxy(self, label_decode_cfs):
        return self.kd_tree.get_query_instance_dummies(label_decode_cfs)

    def get_cfs_normalized_proxy(self, cfs):
        return self.kd_tree.pipeline.named_transformers_['cat'].transform(cfs)

    def _concat(self, cfs):
        dataset = np.vstack((self.normalization_lof_dataset(), cfs))
        return dataset

    @lru_cache()
    def normalization_lof_dataset(self):
        return self.kd_tree.lof_dataset

    def compute_hinge_loss_and_lof(self, cfs):
        """
        compute hinge loss of Multi-Category
        :param cfs:
        :return:
        """

        normalized_cfs_with_dummies = cfs.reshape(1, -1).astype("float64")
        self.kd_tree.single_transform(normalized_cfs_with_dummies, operate="num")

        lof_scores = -self.kd_tree.lof.decision_function(normalized_cfs_with_dummies)[-1]
        predicted_value = self.pred_model.predict_proba(normalized_cfs_with_dummies)

        # hinge loss
        maxvalue = np.full((len(predicted_value)), -np.inf)
        for c in range(self.kd_tree.query_instance_output_nodes):
            if c != self.kd_tree.desired_class:
                maxvalue = np.maximum(maxvalue, predicted_value[:, c])

        return (
            np.maximum(0, maxvalue - predicted_value[:, int(self.kd_tree.desired_class)]),
            normalized_cfs_with_dummies, lof_scores
        )

    def compute_log_loss(self, cfs):
        normalized_cfs_with_dummies = cfs.reshape(1, -1).astype("float64")
        self.kd_tree.single_transform(normalized_cfs_with_dummies, operate="num")

        lof_scores = -self.kd_tree.lof.score_samples(normalized_cfs_with_dummies)[-1]
        predicted_value = self.pred_model.predict_proba(normalized_cfs_with_dummies)
        loss = log_loss([1, ], predicted_value, labels=[0, 1])
        return loss, normalized_cfs_with_dummies, lof_scores

    def change_column_sequence(self):
        result = list()
        sequence = self.kd_tree.one_hot_train_sequence
        for feature_name in sequence:
            for name in self.normalized_weight:
                if name[0] == feature_name:
                    result.append(name[-1])
                    break
        if 0 in result:
            result_without_zero = [i for i in result if i != 0]
            decimal_places = len(str(min(result_without_zero)).split(".")[1])
            min_value = 1 / (10 ** decimal_places)

        result = [1 / i if i else 1 / min_value for i in result]
        self.feature_importance_weight = result
        return result

    def compute_feature_distance(self, normalized_cfs):
        """
        compute the distance between cfs and x1
        :param normalized_cfs:
        :return:
        """
        if not hasattr(self, "normalized_weight"):
            setattr(self, "normalized_weight", self.pred_model.normalized_weight)
        if not hasattr(self, "feature_importance_weight"):
            self.change_column_sequence()
        if not hasattr(self, "normalized_x1"):
            arr = self.x1.values
            arr = deepcopy(arr.astype("float64"))
            self.kd_tree.single_transform(arr, operate="num")
            setattr(self, "normalized_x1", arr)

        squared_diff = (self.normalized_x1 - normalized_cfs) ** 2
        weighted_sum = np.sum(squared_diff * self.feature_importance_weight)
        return np.sqrt(weighted_sum)

    def compute_sparsity_loss(self, normalized_cfs):
        """
        compute sparsity between x1 and cfs
        :param normalized_cfs:
        :return:
        """
        index = 0
        cont_result = 0.0
        cat_result = 0.0
        cat_count = 0
        cont_count = 0

        for name, value, weight in zip(self.kd_tree.one_hot_train_sequence, normalized_cfs.tolist()[0],
                                       self.feature_importance_weight):
            raw_value = self.normalized_x1.tolist()[0][index]

            if name in self.kd_tree.continuous_feature_names:
                if raw_value != value:
                    if raw_value == 0:
                        raw_value = 1e-5
                    grade = abs(((raw_value - value) / raw_value)) * weight
                    cont_result += grade
                    cont_count += 1
            else:
                if raw_value != value:
                    cat_result += weight
                    cat_count += 1
            index += 1
        return cont_result * cont_count, cat_result * cat_count, cat_count + cont_count

    @abstractmethod
    def _evaluate(self, x, out, *args, **kwargs):
        """
        compute loss function
        :param x:
        :param out:
        :param args:
        :param kwargs:
        :return:
        """
        pass
