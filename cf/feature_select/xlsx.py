import pandas as pd


# 读取Excel文件
def read_excel(file_path):
    return pd.read_excel(file_path, engine='openpyxl')


# 统计空缺值占比
def calculate_missing_percentage(df):
    missing_percentage = df.isnull().mean() * 100
    return missing_percentage


# 主函数
def main():
    # 替换为你的Excel文件路径
    file_path = r"C:\Users\Administrator\Desktop\数据.xlsx"

    # 读取Excel文件
    df = read_excel(file_path)

    # 计算每列的空缺值占比
    missing_percentage = calculate_missing_percentage(df)

    # 打印列名和空缺值占比
    for column, percentage in missing_percentage.items():
        print(f"列名: {column}, 空缺值占比: {percentage:.2f}%")


if __name__ == "__main__":
    main()