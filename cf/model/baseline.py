# -*- encoding: utf-8 -*-
"""
@File    :   baseline.py    
@Contact :   1053522308@qq.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2023/10/11 19:16   wuxiaoqiang      1.0         None
"""
import numpy as np
from ACME.ACME import ACME
from sklearn.preprocessing import StandardScaler
import shap


class BlackBox:

    def __init__(self, model):
        self.model = model
        if hasattr(self.model, 'predict_proba'):
            self.pred_fn = self.model.predict_proba#返回概率矩阵
        else:
            self.pred_fn = self.model.predict#返回最高概率

    def predict(self, X):
        proba = self.pred_fn(X, verbose=0)
        if proba.shape[-1] > 1:#多分类模型
            classes = proba.argmax(axis=-1)
        else:#二分类模型
            classes = proba.flatten()
            classes = np.array([1 if y_pred > 0.5 else 0 for y_pred in classes])#合页损失函数！呜呜呜
        return classes

    def predict_proba(self, X):
        probs = self.pred_fn(X, verbose=0)
        # check if binary classification
        if probs.shape[1] == 1:
            # first column is probability of class 0 and second is of class 1
            probs = np.hstack([1 - probs, probs])
        return probs

    def evaluation_importance(self, train_dataset, train_target, target, cat_features=True, is_show=False):#AcME全局特征重要性！呜呜呜
        df = train_dataset.copy()
        df[target] = train_target
        self.check_classes(train_target)

        if cat_features:
            _cat_features = []
            if target == "income":
                _cat_features = [  "age",
    "hours_per_week",
    "education_Assoc",
    "education_Bachelors",
    "education_Doctorate",
    "education_HS-grad",
    "education_Masters",
    "education_Prof-school",
    "education_School",
    "education_Some-college",
    "marital_status_Divorced",
    "marital_status_Married",
    "marital_status_Separated",
    "marital_status_Single",
    "marital_status_Widowed",
    "relationship_Husband",
    "relationship_Not-in-family",
    "relationship_Other-relative",
    "relationship_Own-child",
    "relationship_Unmarried",
    "relationship_Wife",
    "gender_Female",
    "gender_Male",]
            elif target == "default":
                _cat_features = [ "account_check_status_0 <= ... < 200 DM",
    "account_check_status_< 0 DM",
    "account_check_status_>= 200 DM / salary assignments for at least 1 year",
    "account_check_status_no checking account",
    "credit_history_all credits at this bank paid back duly",
    "credit_history_critical account/ other credits existing (not at this bank)",
    "credit_history_delay in paying off in the past",
    "credit_history_existing credits paid back duly till now",
    "credit_history_no credits taken/ all credits paid back duly",
    "purpose_(vacation - does not exist?)",
    "purpose_business",
    "purpose_car (new)",
    "purpose_car (used)",
    "purpose_domestic appliances",
    "purpose_education",
    "purpose_furniture/equipment",
    "purpose_radio/television",
    "purpose_repairs",
    "purpose_retraining",
    "savings_.. >= 1000 DM ",
    "savings_... < 100 DM",
    "savings_100 <= ... < 500 DM",
    "savings_500 <= ... < 1000 DM ",
    "savings_unknown/ no savings account",
    "present_emp_since_.. >= 7 years",
    "present_emp_since_... < 1 year ",
    "present_emp_since_1 <= ... < 4 years",
    "present_emp_since_4 <= ... < 7 years",
    "present_emp_since_unemployed",
    "personal_status_sex_female : divorced/separated/married",
    "personal_status_sex_male : divorced/separated",
    "personal_status_sex_male : married/widowed",
    "personal_status_sex_male : single",
    "other_debtors_co-applicant",
    "other_debtors_guarantor",
    "other_debtors_none",
    "property_if not A121 : building society savings agreement/ life insurance",
    "property_if not A121/A122 : car or other, not in attribute 6",
    "property_real estate",
    "property_unknown / no property",
    "other_installment_plans_bank",
    "other_installment_plans_none",
    "other_installment_plans_stores",
    "housing_for free",
    "housing_own",
    "housing_rent",
    "credits_this_bank_1",
    "credits_this_bank_2",
    "credits_this_bank_3",
    "credits_this_bank_4",
    "job_management/ self-employed/ highly qualified employee/ officer",
    "job_skilled employee / official",
    "job_unemployed/ unskilled - non-resident",
    "job_unskilled - resident",
    "foreign_worker_no",
    "foreign_worker_yes"]
            elif target == "two_year_recid":
                _cat_features = [
                    "sex_Female",
                    "sex_Male",
                    "race_African-American",
                    "race_Asian",
                    "race_Caucasian",
                    "race_Hispanic",
                    "race_Native American",
                    "race_Other",
                    "age_cat_25 - 45",
                    "age_cat_Greater than 45",
                    "age_cat_Less than 25",
                    "c_charge_degree_F",
                    "c_charge_degree_M"
                ]
            elif target == "loan_status":
                _cat_features = [
                    "grade_A",
                    "grade_B",
                    "grade_C",
                    "grade_D",
                    "grade_E",
                    "grade_F",
                    "grade_G",
                    "home_ownership_MORTGAGE",
                    "home_ownership_OTHER",
                    "home_ownership_OWN",
                    "home_ownership_RENT",
                    "purpose_debt",
                    "purpose_educational",
                    "purpose_purchase",
                    "purpose_small_business",
                    "addr_state_CA",
                    "addr_state_FL",
                    "addr_state_GA",
                    "addr_state_IL",
                    "addr_state_NY",
                    "addr_state_Other",
                    "addr_state_TX"
                ]
            acme = ACME(self.model, target=target, features=train_dataset.columns,
                        cat_features=_cat_features, task="class")
        else:
            acme = ACME(self.model, target=target, features=train_dataset.columns, task="class")

        if is_show:
            acme_1 = acme.explain(df, robust=True, label_class=1)
            summary_plot_1 = acme_1.summary_plot()
            summary_plot_1.show()
            summary_plot_1.write_image(file='./adult_1.pdf')

            acme_0 = acme.explain(df, robust=True, label_class=0)
            summary_plot_0 = acme_0.summary_plot()
            summary_plot_0.show()
            summary_plot_0.write_image(file='./adult_0.pdf')

        acme = acme.explain(df, robust=True, label_class=None)
        importance_df = acme.feature_importance()
        weight = list(zip(importance_df.index, importance_df.iloc[:, -1]))
        print(weight)

        if not hasattr(self.model, "normalized_weight"):
            setattr(self.model, "normalized_weight", weight)

    def check_classes(self, train_target):#确保模型对象具备两个重要的属性：classes_ 和 predict_proba
        if not hasattr(self.model, "classes_"):
            setattr(self.model, "classes_", np.unique(train_target.ravel()))

        if not hasattr(self.model, "predict_proba"):
            setattr(self.model, "predict_proba", self.predict_proba)

    def __getattr__(self, name):#属性访问钩子
        return getattr(self.model, name)
