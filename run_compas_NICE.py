from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from nice import NICE
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import os

# 加载数据
adult = pd.read_csv('cf/feature_select/dataset/compas/row_compas/compass.csv')
X = adult.drop(columns=['two_year_recid'])
y = adult.loc[:, 'two_year_recid']
feature_names = list(X.columns)

X = X.values
y = y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

cat_feat = [1, 2, 3, 4]
num_feat = [0]

# 预处理转换器
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_feat),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_feat)
])
preprocessor.fit(X_train)

# 加载模型并验证输出形状
model = load_model('cf/feature_select/dataset/compas/pred_model/compas_DNN.h5')
print("模型输出形状:", model.output_shape)  # 关键诊断步骤

# 生成训练数据的预测标签（适配单输出模型）
X_train_transformed = preprocessor.transform(X_train)
if model.output_shape[1] == 1:  # 处理sigmoid单输出
    y_proba = model.predict(X_train_transformed).flatten()
    y_train_pred = (y_proba > 0.5).astype(int)
else:  # 处理softmax双输出
    y_train_pred = np.argmax(model.predict(X_train_transformed), axis=1)

# 定义与模型输出兼容的预测函数
def predict_fn(x):
    x_transformed = preprocessor.transform(x)
    if model.output_shape[1] == 1:  # 单输出转双通道
        prob = model.predict(x_transformed)
        return np.hstack([1 - prob, prob])
    else:  # 原始softmax输出
        return model.predict(x_transformed)

# 初始化NICE解释器
NICE_adult = NICE(
    X_train=X_train,
    predict_fn=predict_fn,
    y_train=y_train_pred,
    cat_feat=cat_feat,
    num_feat=num_feat,
    distance_metric='HEOM',
    num_normalization='minmax',
    optimization='proximity',
    justified_cf=True
)

# 筛选target=0的原始样本
target_zero_indices = np.where(y == 1)[0]
selected_indices = np.random.choice(target_zero_indices, 100, replace=False)
selected_samples = X[selected_indices]

# 生成反事实解释
original_samples = []
cf_samples = []

for i, sample in enumerate(selected_samples):
    try:
        # 确保输入为二维数组
        sample_2d = sample.reshape(1, -1)
        CF = NICE_adult.explain(sample_2d)

        if CF is not None and len(CF) > 0:
            original_samples.append(sample)
            cf_samples.append(CF[0])
            print(f"Generated CF for sample {i + 1}/100")
        else:
            print(f"No CF found for sample {i + 1}, skipping...")
    except Exception as e:
        print(f"Error processing sample {i + 1}: {str(e)}")

# 转换为DataFrame
df_original = pd.DataFrame(original_samples, columns=feature_names)
df_cf = pd.DataFrame(cf_samples, columns=feature_names)

# 创建保存目录
save_dir = "E:/yan/XAI/py/SFE-CF/results/"
os.makedirs(save_dir, exist_ok=True)

# 保存文件
df_original.to_csv(os.path.join(save_dir, "compas/NICE/predict_samples.csv"), index=False)
df_cf.to_csv(os.path.join(save_dir, "compas/NICE/counterfactual_samples.csv"), index=False)

print(f"Saved {len(original_samples)} original samples and {len(cf_samples)} CF samples")