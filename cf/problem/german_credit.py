# -*- encoding: utf-8 -*-
import autograd.numpy as anp
import numpy as np
from pymoo.core.variable import Integer, Choice

from .problem_base import DistanceProblemVersion5


# 可变约束
class GermanCreditProblemAVarConstVersion1(DistanceProblemVersion5):
    weight = {
        'education_Assoc': 1,
        'education_Bachelors': 1,
        'education_Doctorate': 3,
        'education_HS-grad': 0,
        'education_Masters': 2,
        'education_Prof-school': 2,
        'education_School': 0,
        'education_Some-college': 1,

        'marital_status_Divorced': 2,
        'marital_status_Married': 1,
        'marital_status_Separated': 2,
        'marital_status_Single': 0,
        'marital_status_Widowed': 2,
    }

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

        # 年龄约束 - 只能增长
        #g1 = self.x1['age'][0] - x[4]

        # 15个分类特征
        # account_check_status
        h3 = normalized_cfs_with_dummies[:, 4:8].sum() - 1
        # # credit_history
        h4 = normalized_cfs_with_dummies[:, 8:13].sum() - 1
        # # purpose
        h5 = normalized_cfs_with_dummies[:, 13:23].sum() - 1
        # # savings
        h6 = normalized_cfs_with_dummies[:, 23:28].sum() - 1
        # # present_emp_since
        h7 = normalized_cfs_with_dummies[:, 28:33].sum() - 1
        # # personal_status_sex
        h8 = normalized_cfs_with_dummies[:, 33:37].sum() - 1
        # # other_debtors
        h10 = normalized_cfs_with_dummies[:, 37:40].sum() - 1
        # # property
        h11 = normalized_cfs_with_dummies[:, 40:44].sum() - 1
        # # other_installment_plans
        h12 = normalized_cfs_with_dummies[:, 44:47].sum() - 1
        # # housing
        h13 = normalized_cfs_with_dummies[:, 47:50].sum() - 1
        # # credits_this_bank
        h14 = normalized_cfs_with_dummies[:, 50:54].sum() - 1
        # # job
        h16 = normalized_cfs_with_dummies[:, 54:58].sum() - 1
        # # people_under_maintenance
        h17 = normalized_cfs_with_dummies[:, 58:60].sum() - 1
        # #
        # h18 = normalized_cfs_with_dummies[:, 59:61].sum() - 1
        # #
        #h19 = normalized_cfs_with_dummies[:, 63:65].sum() - 1
        out["F"] = anp.column_stack([yloss, feature_distance, lof_scores, cont_result, cat_result])
        # out["G"] = anp.column_stack([g1, ])
        #out["H"] = anp.column_stack([h3, h4, h5, h6, h7, h8, h10, h11, h12, h13, h14, h16, h17])

        # out["F"] = anp.column_stack([yloss, feature_distance, lof_scores, cont_result, cat_result, count])
        #out["G"] = anp.column_stack([g1, ])
        out["H"] = anp.column_stack([h3, h4, h5, h6, h7, h8, h10, h11, h12, h13, h14, h16, h17])

class GermanCreditProblemAVarConstVersion2(DistanceProblemVersion5):
    weight = {
        'education_Assoc': 1,
        'education_Bachelors': 1,
        'education_Doctorate': 3,
        'education_HS-grad': 0,
        'education_Masters': 2,
        'education_Prof-school': 2,
        'education_School': 0,
        'education_Some-college': 1,

        'marital_status_Divorced': 2,
        'marital_status_Married': 1,
        'marital_status_Separated': 2,
        'marital_status_Single': 0,
        'marital_status_Widowed': 2,
    }

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

    @staticmethod
    def _normalized_weight(data):
        min_value = np.min(data)
        max_value = np.max(data)
        normalized_data = (data - min_value) / (max_value - min_value)
        return normalized_data

    # ---------------------------------------------------------------------------------------------
    # age education race gender作为不可变约束 with post fix
    def _evaluate(self, x, out, *args, **kwargs):
        # 预测损失 —— hinge loss
        yloss, normalized_cfs_with_dummies, lof_scores = self.compute_hinge_loss_and_lof(x)

        # 样本距离  —— 目前未区分 连续特征和分类特征
        feature_distance = self.compute_feature_distance(normalized_cfs_with_dummies)

        # 添加稀疏度损失，考虑在原始的稀疏度衡量下加入变化幅度
        cont_result, cat_result, count = self.compute_sparsity_loss(normalized_cfs_with_dummies)

        # 年龄约束 - 更改为不可变因素
        g1 = self.x1['age'][0] - x[0]

        # 性别约束
        _h1 = self.normalized_x1[:, 27:] - normalized_cfs_with_dummies[:, 27:]
        h1 = 1 if _h1.any() else 0

        # 种族约束
        _h3 = self.normalized_x1[:, 25:27] - normalized_cfs_with_dummies[:, 25:27]
        h3 = 1 if _h3.any() else 0

        # 教育约束
        _h7 = self.normalized_x1[:, 6:14] - normalized_cfs_with_dummies[:, 6:14]
        h7 = 1 if _h7.any() else 0

        h5 = normalized_cfs_with_dummies[:, 19:25].sum() - 1
        h6 = normalized_cfs_with_dummies[:, 14:19].sum() - 1
        h8 = normalized_cfs_with_dummies[:, 2:6].sum() - 1

        try:
            cf_mixed = self.kd_tree.reverse_transform(normalized_cfs_with_dummies)
        except ValueError:
            g6 = 1
        else:
            g6 = 1
            if not h6:
                raw_name = 'marital_status_' + self.raw_x1['marital_status'][0]
                current_name = 'marital_status_' + cf_mixed['marital_status'][0]
                g6 = self.weight[raw_name] - self.weight[current_name]

        out["F"] = anp.column_stack([yloss, feature_distance, lof_scores, count])
        out["G"] = anp.column_stack([g6, ])
        out["H"] = anp.column_stack([g1, h1, h3, h5, h6, h7, h8])
