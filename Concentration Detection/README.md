# 荧光响应抗生素识别与浓度定量预测平台 (Antibiotic Detection ML Platform)

本项目基于多维图像色彩特征提取与机器学习（Machine Learning）算法，构建了一套高效的混合/单组分抗生素特异性荧光识别与定量浓度预测系统。系统能够自动读取荧光图像，提取多维色彩特征（RGB、RGB 比值、归一化 RGB 以及 HSV 空间），通过分类与回归模型实现抗生素种类鉴别与浓度定量，并支持消融实验、多维响应热图及盲样泛化验证。

---

## 目录

* [项目架构与核心功能](#项目架构与核心功能)
* [目录结构](#目录结构)
* [环境依赖与安装](#环境依赖与安装)
* [快速上手与使用说明](#快速上手与使用说明)
* [消融实验与论文级可视化 (审稿人应对模块)](#消融实验与论文级可视化-审稿人应对模块)

---

## 项目架构与核心功能

1. **多维特征工程 (`src/image_processing.py`)**：
   * 支持图像 ROI 中心截取，消除边缘杂光干扰。
   * 提取 12 维全面色彩与光学特征：
     * **RGB 基础通道**：$R, G, B$
     * **颜色比值**：$R/G, R/B, G/B$
     * **归一化色彩分量**：$r_{norm}, g_{norm}, b_{norm}$
     * **HSV 空间特征**：$H$ (色调), $S$ (饱和度), $V$ (明度/荧光强度)

2. **双任务机器学习 Pipeline (`main.py`)**：
   * **任务一：抗生素种类识别（分类）**：对比了 Logistic Regression、LDA、SVM、KNN、Decision Tree 与 Random Forest 在不同特征组合下的表现。
   * **任务二：浓度定量预测（回归）**：评估了 Linear Regression、SVR、KNN、Decision Tree 及 Random Forest 的 $R^2$、$\text{RMSE}$ 与 $\text{MAE}$。
   * **防数据泄露交叉验证**：内置 `GroupKFold`，以批次 (`Batch`) 为组别隔离训练集与验证集。

3. **模型持久化与自动化预测 (`predict.py`)**：
   * 训练完成后自动将标准化器、编码器及模型打包保存至 `models/`。
   * 运行预测脚本可直接对新样本图片进行自动分类与浓度预测，并导出至 `outputs/prediction_results.xlsx`。

---

## 目录结构

```text
AntibioticDetection/
│
├── data/                         # 数据存储目录
│   ├── raw_images/               # 原始荧光样本图片
│   ├── test_samples/             # 待实际检测的新样品图片
│   └── test.xlsx                 # 核心特征与标签数据集
│
├── models/                       # 模型持久化目录
│   ├── scaler.pkl                # 特征标准化器
│   ├── label_encoder.pkl         # 种类标签编码器
│   ├── classifier_species.pkl    # 抗生素分类模型
│   └── regressor_concentration.pkl # 浓度回归模型
│
├── outputs/                      # 输出目录
│   ├── prediction_results.xlsx   # 新样本预测汇总表
│   └── figures/                  # 论文级高清图表（柱状图、热图、PCA等）
│
├── src/                          # 核心功能模块
│   ├── __init__.py
│   ├── image_processing.py       # ROI 截取与多维色彩特征提取
│   ├── dataset_builder.py        # 批量处理原始图片与数据集更新
│   └── visualizer.py             # 可视化模块（热图、PCA、四维HSV响应曲线等）
│
├── main.py                       # 主程序1：模型训练、交叉验证与消融实验
├── predict.py                    # 主程序2：新图像批量预测与结果导出
└── requirements.txt              # 依赖库清单
```

---

## 环境依赖与安装

推荐使用 Python 3.8+ 环境。在项目根目录下运行以下命令安装依赖：

```bash
pip install -r requirements.txt
```

**`requirements.txt` 内容：**
```text
numpy
pandas
scikit-learn
opencv-python
matplotlib
seaborn
openpyxl
joblib
```

---

## 快速上手与使用说明

### 1. 模型训练与交叉验证
运行 `main.py` 启动全流程模型训练、交叉验证与评估：
```bash
python main.py
```
* 执行后将自动加载 `data/test.xlsx`，输出各大分类与回归模型的准确率、$R^2$、RMSE 等指标，并将最优模型打包保存至 `models/`。

### 2. 运行消融实验与学术图表生成
如果你想复现特征消融实验柱状图或生成多维特征响应热图，可运行对应的分析脚本：
```bash
python ablation/run_ablation.py
python src/visualizer.py
```
* 生成的高清图表（如 `ablation_bar_chart.png`、`antibiotic_feature_heatmap.png`）将自动保存在 `outputs/figures/` 目录下。

### 3. 新样本图像智能检测与预测
当有实际测样图片需要分析时，将图片放入 `data/test_samples/` 后运行：
```bash
python predict.py
```
* 程序将自动计算图片特征，载入模型预测其抗生素种类与浓度，并导出标准 Excel 报告至 `outputs/prediction_results.xlsx`。

---

## 消融实验与论文级可视化 (审稿人应对模块)

本系统在架构与算法设计上充分兼顾了学术发表与实际应用的需求：

1. **防数据泄露机制**：采用 `GroupKFold` 结合批次 (`Batch`) 标识进行交叉验证，杜绝同批次平行样切分造成的过拟合与数据泄露。
2. **多维特征消融分析 (Ablation Study)**：系统对比了单一特征（如 Baseline）与全特征组合（RGB + Ratios + Normalized + HSV）的性能差异，证明多特征融合在复杂抗生素识别中的必要性。
3. **泛化能力验证**：支持盲样测试（Unseen Concentrations），有效评估模型在未知浓度区间的回归预测稳定性。
4. **出版级可视化支持**：内置 Seaborn 风格的学术图表渲染逻辑（高 DPI 导出、严谨的坐标轴与误差棒支持），可直接无缝嵌入论文。