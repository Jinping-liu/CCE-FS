import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import time

# 载入数据集
original_data = pd.read_csv("./dataset/diabetes/diabetes_fill.csv")
X_original = original_data.iloc[:, :-1].values
original_means = X_original.mean(axis=0)

data = pd.read_csv("./dataset/diabetes/processed_diabetes.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 获取特征名
feature_names = data.columns[:-1]

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

# 载入已有的模型
model = tf.keras.models.load_model('./dataset/diabetes/diabetes_DNN.h5')
preprocessor = joblib.load("./dataset/diabetes/diabetes_pipeline.pkl")

# 原始模型性能
original_predictions = (model.predict(X_test) > 0.5).astype("int32")
original_accuracy = accuracy_score(y_test, original_predictions)
original_precision = precision_score(y_test, original_predictions)
original_recall = recall_score(y_test, original_predictions)
original_f1 = f1_score(y_test, original_predictions)
print(f"Original Model Accuracy: {original_accuracy}")
print(f"Original Model Precision: {original_precision}")
print(f"Original Model Recall: {original_recall}")
print(f"Original Model F1 Score: {original_f1}")

necessary_features = []
sufficient_features = []

# 记录测试时间和预测准确率
necessary_times = []
necessary_accuracies = []
sufficient_times = []
sufficient_accuracies = []

# 必要性测试
for i in range(X_test.shape[1]):
    X_test_necessity = X_test.copy()
    X_test_necessity[:, i] = 0

    start_time = time.time()
    necessity_predictions = model.predict(X_test_necessity)
    end_time = time.time()

    necessary_times.append(end_time - start_time)
    necessity_accuracy = accuracy_score(y_test, (necessity_predictions > 0.5).astype("int32"))
    necessary_accuracies.append(necessity_accuracy)

    if necessity_accuracy < original_accuracy:
        necessary_features.append(i)

# 充分性测试
for i in range(X_test.shape[1]):
    X_test_sufficiency = X_test.copy()
    X_test_sufficiency[:, np.arange(X_test.shape[1]) != i] = 0

    start_time = time.time()
    sufficiency_predictions = model.predict(X_test_sufficiency)
    end_time = time.time()

    sufficient_times.append(end_time - start_time)
    sufficiency_accuracy = accuracy_score(y_test, (sufficiency_predictions > 0.5).astype("int32"))
    sufficient_accuracies.append(sufficiency_accuracy)

    if sufficiency_accuracy >= original_accuracy * 0.9:
        sufficient_features.append(i)

print(f"Necessary Features: {necessary_features}")
print(f"Sufficient Features: {sufficient_features}")

# 保留必要特征，将非必要特征设置为0
X_test_optimized = X_test.copy()
for i in range(X_test.shape[1]):
    if i not in necessary_features:
        X_test_optimized[:, i] = 0

# 评估优化后的模型性能
optimized_predictions = (model.predict(X_test_optimized) > 0.5).astype("int32")
optimized_accuracy = accuracy_score(y_test, optimized_predictions)
optimized_precision = precision_score(y_test, optimized_predictions)
optimized_recall = recall_score(y_test, optimized_predictions)
optimized_f1 = f1_score(y_test, optimized_predictions)
print(f"Optimized Model Accuracy: {optimized_accuracy}")
print(f"Optimized Model Precision: {optimized_precision}")
print(f"Optimized Model Recall: {optimized_recall}")
print(f"Optimized Model F1 Score: {optimized_f1}")

plt.figure(figsize=(12, 6))  # 调整图表尺寸

# 绘制必要性和充分性测试结果的折线图，调整颜色和标记
plt.plot(necessary_accuracies, label='Necessary', marker='o', color='blue', linewidth=2, markersize=8, alpha=0.8)
plt.plot(sufficient_accuracies, label='Sufficient', marker='s', color='red', linewidth=2, markersize=8, alpha=0.8)

# 添加原始模型准确率的水平参考线
plt.axhline(y=original_accuracy, color='green', linestyle='--', label='Original Model', linewidth=2)
# plt.axhline(y=optimized_accuracy, color='purple', linestyle='--', label='Optimized Model', linewidth=2)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.5)

# 设置图表信息
plt.xlabel('Feature')
plt.ylabel('Accuracy')
plt.title('Necessary and Sufficient Features Accuracy')
plt.xticks(np.arange(X_test.shape[1]), feature_names, rotation=45)
plt.legend()

plt.tight_layout()
plt.savefig("./Necessary_Sufficient_Optimized.pdf", dpi=600)
plt.show()
