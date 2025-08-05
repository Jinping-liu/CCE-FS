# # -*- encoding: utf-8 -*-
# """
# @File    :   test.py
# @Contact :   1053522308@qq.com
# @License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
#
# @Modify Time      @Author    @Version    @Desciption
# ------------      -------    --------    -----------
# 2023/10/23 10:50   wuxiaoqiang      1.0         None
# """
# # import pickle
# #
# # import pandas as pd
# #
# # dice_total_1_cfs_path = "/Users/wxq/Desktop/SFE-CF/results/lending_dice_result/nonlinear/prox_0.5+div_1.0+algo_RandomInitCF+yloss_hinge_loss+divloss_dpp_style_inverse_dist+lr_0.05+postfix_0.1+init_near_x1_False/tot_cf_1.data"
# # target_file_path = "./results/lending/randominit/"
# #
# # feature_names = [
# #     "emp_length",
# #     "annual_inc",
# #     "open_acc",
# #     "credit_years",
# #     "grade",
# #     "home_ownership",
# #     "purpose",
# #     "addr_state"
# # ]
# #
# # with open(dice_total_1_cfs_path, "rb") as f:
# #     dice_summary_all = pickle.load(f)
# #
# # withoutSpare = dice_summary_all['without_postfix']
# # withoutSpare = [i[0][:-1] for i in withoutSpare]
# # withoutSpare = pd.DataFrame(withoutSpare, columns=feature_names)
# # withoutSpare.to_csv(f"./{target_file_path}/Lending_nonliner_randominit_without_postfix_tot_1.csv", index=False)
# #
# #
# # withSpare = dice_summary_all['with_postfix']
# # withSpare = [i[0][:-1] for i in withSpare]
# # withSpare = pd.DataFrame(withSpare, columns=feature_names)
# # withSpare.to_csv(f"./{target_file_path}/Lending_nonliner_randominit_with_postfix_tot_1.csv", index=False)
#
#
# import pandas as pd
#
#
# def find_changed_rows(file1, file2, feature_name, old_value, new_value):
#     # 读取两个 CSV 文件
#     df1 = pd.read_csv(file1)
#     df2 = pd.read_csv(file2)
#
#     # 在两个数据框中查找特定特征值的行
#     changed_rows = []
#     for index, row1 in df1.iterrows():
#         value1 = row1[feature_name]
#         value2 = df2.iloc[index][feature_name]
#         if value1 == old_value and value2 == new_value:
#             print(index)
#             changed_rows.append(row1)
#
#     return changed_rows
#
#
# # Example usage:
# changed_rows = find_changed_rows("/Users/wxq/Desktop/SFE-CF/dataset/lending/predict_samples_version_1_100_IWFM.csv", "/Users/wxq/Desktop/SFE-CF/dataset/lending/counterfactual_samples_version_1_100_IWFM.csv", "addr_state", "FL", "TX")
# print("Rows where the feature value has changed:")
# for row in changed_rows:
#     print(row)

# import threading
#
# lock_a = threading.Lock()
# lock_b = threading.Lock()
# lock_c = threading.Lock()
#
# lock_b.acquire()
# lock_c.acquire()
#
#
# def printA():
#     while True:
#         lock_a.acquire()
#         print("a")
#         lock_b.release()
#
#
# def printB():
#     while True:
#         lock_b.acquire()
#         print("b")
#         lock_c.release()
#
#
# def printC():
#     while True:
#         lock_c.acquire()
#         print("c")
#         lock_a.release()
#
#
# tasks = []
# tasks.append(threading.Thread(target=printA))
# tasks.append(threading.Thread(target=printB))
# tasks.append(threading.Thread(target=printC))
#
# for t in tasks:
#     t.start()
#
# for t in tasks:
#     t.join()


import threading
import time

lock_a = threading.Lock()
lock_b = threading.Lock()
currentSum = 1

lock_b.acquire()


def add_odd():
    global currentSum
    while True:
        if currentSum <= 100:
            lock_a.acquire()
            print(currentSum)
            currentSum += 1
            lock_b.release()
        time.sleep(0)


def add_even():
    global currentSum
    while True:
        if currentSum <= 100:
            lock_b.acquire()
            print(currentSum)
            currentSum += 1
            lock_a.release()
        time.sleep(0)


tasks = []
tasks.append(threading.Thread(target=add_odd))
tasks.append(threading.Thread(target=add_even))

for t in tasks:
    t.start()

for t in tasks:
    t.join()
