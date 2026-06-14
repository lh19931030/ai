# ML CSV Lab

一个可本地运行、适合提交到 Git 的 CSV 机器学习小应用。

## 功能

- 只允许上传 CSV 文件
- 自动检测表头，并支持手动切换
- 预览前 5 行数据
- 默认最后一列为目标变量，可手动选择
- 自动判断分类/回归任务，可手动覆盖
- 展示均值、标准差、缺失值、唯一值和分布
- 使用 80/20 train-test split 训练模型
- 回归模型：Linear Regression、Random Forest Regressor、SVR
- 分类模型：Logistic Regression、Random Forest Classifier、SVM
- 输出 Accuracy/MSE 等指标和可视化图表

## 运行

建议先创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

然后打开：

```text
http://127.0.0.1:8000
```

如果 8000 端口被占用：

```bash
PORT=8001 python server.py
```

## 快速测试

目录内包含两个示例文件：

- `data/sample_iris.csv`：分类任务
- `data/sample_housing.csv`：回归任务

## Git 提交

这个目录只包含源码和小型示例数据，运行时缓存、虚拟环境、`.DS_Store`、`__pycache__` 等都会被 `.gitignore` 忽略。
