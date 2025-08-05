# -*- encoding: utf-8 -*-
import autograd.numpy as anp
import numpy as np
from pymoo.core.variable import Integer, Choice

from .problem_base import DistanceProblemVersion5


# 可变约束
class DiabetesProblemAVarConstVersion1(DistanceProblemVersion5):

    def __init__(self, x1, raw_x1, pred_model, kd_tree_model, n_var, n_obj, n_ieq_constr, n_eq_constr, xl, xu,
                 **kwargs):
        vars = dict()
        self.feature_index = dict()
        for index, feature in enumerate(kd_tree_model.one_hot_train_sequence):
            self.feature_index[feature] = index
            if feature in kd_tree_model.continuous_feature_names:
                _range = kd_tree_model.feature_range[feature]
                vars[feature] = Integer(bounds=(_range[0], _range[1]))
            else:
                vars[feature] = Choice(options=[0, 1])
        super().__init__(x1, raw_x1, pred_model, kd_tree_model, n_var, n_obj, n_ieq_constr=n_ieq_constr,
                         n_eq_constr=n_eq_constr, xl=xl, xu=xu, vars=vars, **kwargs)

    # ---------------------------------------------------------------------------------------------
    # 平等实验、将结果与Dice方法进行对比
    def _evaluate(self, x, out, *args, **kwargs):
        # 预测损失 —— hinge loss
        yloss, normalized_cfs_with_dummies, lof_scores = self.compute_hinge_loss_and_lof(x)

        # 样本距离  —— 目前未区分 连续特征和分类特征
        feature_distance = self.compute_feature_distance(normalized_cfs_with_dummies)

        # 添加稀疏度损失，考虑在原始的稀疏度衡量下加入变化幅度
        cont_result, cat_result, count = self.compute_sparsity_loss(normalized_cfs_with_dummies)

        # 年龄不可变、以往妊娠次数不可变
        # h1 = self.x1['Age'][0] - x[7]
        # h2 = self.x1['Pregnancies'][0] - x[0]
        # h3 = self.x1['DiabetesPedigreeFunction'][0] - x[6]

        out["F"] = anp.column_stack([yloss, feature_distance, lof_scores, cont_result, cat_result])
        # out["H"] = anp.column_stack([h1, h2, h3])


# 可变约束
class DiabetesProblemAVarConstVersion3(DistanceProblemVersion5):

    def __init__(self, x1, raw_x1, pred_model, kd_tree_model, n_var, n_obj, n_ieq_constr, n_eq_constr, xl, xu,
                 **kwargs):
        vars = dict()
        self.feature_index = dict()
        for index, feature in enumerate(kd_tree_model.one_hot_train_sequence):
            self.feature_index[feature] = index
            if feature in kd_tree_model.continuous_feature_names:
                _range = kd_tree_model.feature_range[feature]
                vars[feature] = Integer(bounds=(_range[0], _range[1]))
            else:
                vars[feature] = Choice(options=[0, 1])
        super().__init__(x1, raw_x1, pred_model, kd_tree_model, n_var, n_obj, n_ieq_constr=n_ieq_constr,
                         n_eq_constr=n_eq_constr, xl=xl, xu=xu, vars=vars, **kwargs)

    # ---------------------------------------------------------------------------------------------
    # 平等实验、将结果与Dice方法进行对比
    def _evaluate(self, x, out, *args, **kwargs):
        # 预测损失 —— hinge loss
        yloss, normalized_cfs_with_dummies, lof_scores = self.compute_hinge_loss_and_lof(x)

        # 样本距离  —— 目前未区分 连续特征和分类特征
        feature_distance = self.compute_feature_distance(normalized_cfs_with_dummies)

        # 添加稀疏度损失，考虑在原始的稀疏度衡量下加入变化幅度
        cont_result, cat_result, count = self.compute_sparsity_loss(normalized_cfs_with_dummies)

        # 年龄不可变、以往妊娠次数不可变
        h1 = self.x1['Age'][0] - x[7]
        h2 = self.x1['Pregnancies'][0] - x[0]
        h3 = self.x1['DiabetesPedigreeFunction'][0] - x[6]

         #Insulin不可变
        # h4 = self.x1['Insulin'][0] - x[4]

        # out["F"] = anp.column_stack([yloss, feature_distance, lof_scores, cont_result, cat_result])
        out["F"] = anp.column_stack([yloss, feature_distance, lof_scores, cont_result, cat_result, count])
        # out["H"] = anp.column_stack([h1, h2, h3, h4])
        out["H"] = anp.column_stack([h1, h2, h3])
