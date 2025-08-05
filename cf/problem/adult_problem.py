# -*- encoding: utf-8 -*-
import autograd.numpy as anp
import numpy as np
from pymoo.core.variable import Integer, Choice

from .problem_base import DistanceProblemVersion5

"""
'age': 年龄
'capital-gain':额外收入
'hours_per_week': 每周工作小时数
'relationship_Husband':丈夫家庭关系
‘relationship_Not-in-family':无家庭
'relationship_Other-relative':其他关系
'relationship_Own-child':抚育小孩
'relationship_Unmarried':未婚
'relationship_Wife':妻子
'education_Assoc': 副学士
'education_Bachelors': 本科
'education_Doctorate': 博士
'education_HS-grad': 高中
'education_Masters': 硕士
'education_Prof-school': 专科
'education_School': 学校教育水平
'education_Some-college': 学院
'marital_status_Divorced': 离异婚姻状况
'marital_status_Married': 已婚婚姻状况
'marital_status_Separated': 分居婚姻状况
'marital_status_Single': 单身婚姻状况
'marital_status_Widowed': 丧偶婚姻状况
'occupation_Blue-Collar': 蓝领职业类别
'occupation_Other/Unknown': 其他/未知职业类别
'occupation_Professional': 专业职业类别
'occupation_Sales': 销售职业类别
'occupation_Service': 服务职业类别
'occupation_White-Collar': 白领职业类别
'gender_Female': 女性性别
'gender_Male': 男性性别
"""


class AdultProblem(DistanceProblemVersion5):
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
                 **kwargs):#初始化问题，设置变量的范围和类型，以及特征索引。
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
    def _normalized_weight(data):#归一化处理
        min_value = np.min(data)
        max_value = np.max(data)
        normalized_data = (data - min_value) / (max_value - min_value)
        return normalized_data

    def _evaluate(self, x, out, *args, **kwargs):#计算目标函数值和约束条件值
        # 预测损失 —— hinge loss
        yloss, normalized_cfs_with_dummies, lof_scores = self.compute_hinge_loss_and_lof(x)

        # 样本距离  —— 目前未区分 连续特征和分类特征
        feature_distance = self.compute_feature_distance(normalized_cfs_with_dummies)

        # 添加稀疏度损失，考虑在原始的稀疏度衡量下加入变化幅度
        cont_result, cat_result = self.compute_sparsity_loss(normalized_cfs_with_dummies)

        # 年龄约束 - 更改为不可变因素
        g1 = self.normalized_x1[:, 0] - normalized_cfs_with_dummies[:, 0]

        # 性别约束
        _h1 = self.normalized_x1[:, 27:] - normalized_cfs_with_dummies[:, 27:]
        h1 = 1 if _h1.any() else 0

        # 种族约束
        _h3 = self.normalized_x1[:, 25:27] - normalized_cfs_with_dummies[:, 25:27]
        h3 = 1 if _h3.any() else 0

        h5 = normalized_cfs_with_dummies[:, 19:25].sum() - 1#如果和不为1，则意味着有多个特征被激活，这违反了互斥性约束。
        h6 = normalized_cfs_with_dummies[:, 14:19].sum() - 1
        h7 = normalized_cfs_with_dummies[:, 6:14].sum() - 1
        h8 = normalized_cfs_with_dummies[:, 2:6].sum() - 1

        try:
            cf_mixed = self.kd_tree.reverse_transform(normalized_cfs_with_dummies)
        except ValueError:
            g6 = 10
            g7 = 10
        else:
            g6 = 10
            if not h6:
                raw_name = 'marital_status_' + self.raw_x1['marital_status'][0]
                current_name = 'marital_status_' + cf_mixed['marital_status'][0]
                g6 = self.weight[raw_name] - self.weight[current_name]

            g7 = 10
            if not h7:
                raw_name = 'education_' + self.raw_x1['education'][0]
                current_name = 'education_' + cf_mixed['education'][0]
                g7 = self.weight[raw_name] - self.weight[current_name]
                # 学历的增长必然伴随着年龄的增长
                edu_age_const = np.sign(g7, g1)

        out["F"] = anp.column_stack([yloss[0], feature_distance, cont_result, cat_result, lof_scores])
        out["G"] = anp.column_stack([g1, g6, g7])
        out["H"] = anp.column_stack([h1, h3, h5, h6, h7, h8])


# 可变约束
class AdultProblemAVarConstVersion1(DistanceProblemVersion5):
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
    # 无任何合理性约束、将结果与Dice方法进行对比
    def _evaluate(self, x, out, *args, **kwargs):
        # 预测损失 —— hinge loss
        yloss, normalized_cfs_with_dummies, lof_scores = self.compute_hinge_loss_and_lof(x)

        # 样本距离  —— 目前未区分 连续特征和分类特征
        feature_distance = self.compute_feature_distance(normalized_cfs_with_dummies)

        # 添加稀疏度损失，考虑在原始的稀疏度衡量下加入变化幅度
        cont_result, cat_result, count = self.compute_sparsity_loss(normalized_cfs_with_dummies)

        # 其他分类特征约束，保证最终只有一个可选
        #h1 = normalized_cfs_with_dummies[:, 31:33].sum() - 1
        #h3 = normalized_cfs_with_dummies[:, 25:31].sum() - 1
        h5 = normalized_cfs_with_dummies[:, 21:23].sum() - 1
        h6 = normalized_cfs_with_dummies[:, 15:21].sum() - 1
        h7 = normalized_cfs_with_dummies[:, 10:15].sum() - 1
        h8 = normalized_cfs_with_dummies[:, 2:10].sum() - 1
        #h9 = normalized_cfs_with_dummies[:, 33:35].sum() - 1
        #h10 = normalized_cfs_with_dummies[:, 35:44].sum() - 1
        # cont_result,
        out["F"] = anp.column_stack([yloss, feature_distance, lof_scores, cont_result, cat_result])
        # out["F"] = anp.column_stack([yloss, feature_distance, lof_scores,  count])
        # out["G"] = anp.column_stack([g1, ])
        out["H"] = anp.column_stack([h5, h6, h7, h8])
        # out["H"] = anp.column_stack([h3, h5, h6, h7])


class AdultProblemAVarConstVersion3(DistanceProblemVersion5):
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
    # age education race gender作为不可变约束 with post fix
    def _evaluate1(self, x, out, *args, **kwargs):
        # 预测损失 —— hinge loss
        yloss, normalized_cfs_with_dummies, lof_scores = self.compute_hinge_loss_and_lof(x)

        # 样本距离  —— 目前未区分 连续特征和分类特征
        feature_distance = self.compute_feature_distance(normalized_cfs_with_dummies)

        # 添加稀疏度损失，考虑在原始的稀疏度衡量下加入变化幅度
        cont_result, cat_result, count = self.compute_sparsity_loss(normalized_cfs_with_dummies)

        # 年龄约束 - 只能上升
        g1 = self.x1['age'][0] - x[0]

        # 性别约束
        _h1 = self.normalized_x1[:, 27:] - normalized_cfs_with_dummies[:, 27:]
        h1 = 1 if _h1.any() else 0

        # 种族约束
        _h3 = self.normalized_x1[:, 25:27] - normalized_cfs_with_dummies[:, 25:27]
        h3 = 1 if _h3.any() else 0

        h5 = normalized_cfs_with_dummies[:, 19:25].sum() - 1
        h6 = normalized_cfs_with_dummies[:, 14:19].sum() - 1
        h7 = normalized_cfs_with_dummies[:, 6:14].sum() - 1
        h8 = normalized_cfs_with_dummies[:, 2:6].sum() - 1

        try:
            cf_mixed = self.kd_tree.reverse_transform(normalized_cfs_with_dummies)
        except ValueError:
            g6 = 1
            g7 = 1
        else:
            g6 = 1
            if not h6:
                raw_name = 'marital_status_' + self.raw_x1['marital_status'][0]
                current_name = 'marital_status_' + cf_mixed['marital_status'][0]
                g6 = self.weight[raw_name] - self.weight[current_name]

            g7 = 1
            if not h7:
                raw_name = 'education_' + self.raw_x1['education'][0]
                current_name = 'education_' + cf_mixed['education'][0]
                g7 = self.weight[raw_name] - self.weight[current_name]

        out["F"] = anp.column_stack([yloss, feature_distance, lof_scores, cont_result, cat_result])
        out["G"] = anp.column_stack([g1, g6, g7])
        out["H"] = anp.column_stack([h1, h3, h5, h6, h7, h8])

    def _evaluate(self, x, out, *args, **kwargs):
        # 预测损失 —— hinge loss
        yloss, normalized_cfs_with_dummies, lof_scores = self.compute_hinge_loss_and_lof(x)

        # 样本距离  —— 目前未区分 连续特征和分类特征
        feature_distance = self.compute_feature_distance(normalized_cfs_with_dummies)

        # 添加稀疏度损失，考虑在原始的稀疏度衡量下加入变化幅度
        cont_result, cat_result, count = self.compute_sparsity_loss(normalized_cfs_with_dummies)

        # 年龄约束 - 只能上升
        g1 = self.x1['age'][0] - x[0]

        # 性别约束
        _h1 = self.normalized_x1[:, 27:] - normalized_cfs_with_dummies[:, 27:]
        h1 = 1 if _h1.any() else 0

        # 种族约束
        _h3 = self.normalized_x1[:, 25:27] - normalized_cfs_with_dummies[:, 25:27]
        h3 = 1 if _h3.any() else 0

        # 教育也不可变
        _h4 = self.normalized_x1[:, 6:14] - normalized_cfs_with_dummies[:, 6:14]
        h4 = 1 if _h4.any() else 0

        h5 = normalized_cfs_with_dummies[:, 19:25].sum() - 1
        h6 = normalized_cfs_with_dummies[:, 14:19].sum() - 1
        # h7 = normalized_cfs_with_dummies[:, 6:14].sum() - 1
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

        out["F"] = anp.column_stack([yloss, feature_distance, lof_scores, cont_result, cat_result])
        out["G"] = anp.column_stack([g6, ])
        out["H"] = anp.column_stack([g1, h1, h3, h4, h5, h6, h8])
