# src/visualizer.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from matplotlib.patches import Ellipse
import os


class ExperimentVisualizer:
    def __init__(self):
        print("=" * 50)
        print(" 欢迎使用多维可视化绘图模块 (全交互键盘指定版)")
        print("=" * 50)

        # 1. 输入数据路径
        self.data_path = input("请输入数据文件路径 (默认 data/test.xlsx): ").strip()
        if not self.data_path:
            self.data_path = 'data/test.xlsx'

        # 2. 输入目标浓度
        conc_input = input("请输入需要绘制的特定浓度值 (默认 10): ").strip()
        self.target_concentration = float(conc_input) if conc_input else 10.0

        # 3. 【新增】键盘交互指定特征参数
        print("\n可选特征参考（表格中的常见列名）：")
        print("  - 基础色彩: R, G, B")
        print("  - 通道比值: R/G, R/B, G/B")
        print("  - 归一化值: r_norm, g_norm, b_norm")
        print("  - HSV空间: H, S, V")
        print("  - Lab空间: L*, a*, b*")

        feat_input = input("请输入要参考的特征参数，用逗号或空格隔开 (默认 R/B, R): ").strip()
        if feat_input:
            # 支持用逗号、空格或者顿号分隔，并去除多余空格
            import re
            self.feature_cols = [f.strip() for f in re.split(r'[,，\s]+', feat_input) if f.strip()]
        else:
            self.feature_cols = ['R/B', 'R']

        print(f"-> 当前已成功锁定参考特征参数: {self.feature_cols}\n")

        os.makedirs('outputs/figures', exist_ok=True)
        self.load_and_filter_data()

    def load_and_filter_data(self):
        """加载数据并按输入浓度过滤，同时校验用户指定的特征列是否存在"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"找不到数据文件: {self.data_path}")

        df = pd.read_excel(self.data_path)
        df = df[df['antibiotic'] != 'Unknown'].reset_index(drop=True)

        # 校验用户输入的特征列是否真实存在于 Excel 中
        missing_cols = [col for col in self.feature_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️ 警告: 表格中未找到以下特征列: {missing_cols}")
            print(f"表格现有可用列为: {list(df.columns)}")
            # 自动过滤掉不存在的列，防止报错
            self.feature_cols = [col for col in self.feature_cols if col in df.columns]
            if not self.feature_cols:
                raise ValueError("您输入的特征列全部无效，请检查 Excel 表头！")
            print(f"-> 已自动调整为有效特征列: {self.feature_cols}")

        self.df_filtered = df[df['concentration'] == self.target_concentration].reset_index(drop=True)
        print(f"成功加载数据！当前浓度 {self.target_concentration} 下共有样本数: {len(self.df_filtered)}")

        self.X = self.df_filtered[self.feature_cols].values
        self.y = self.df_filtered['antibiotic'].values

        self.le = LabelEncoder()
        self.y_encoded = self.le.fit_transform(self.y)
        self.class_names = self.le.classes_

    def plot_pca(self):
        """绘制基于用户指定特征的 PCA 得分图与置信椭圆"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        # 如果用户指定的特征数少于2个，PCA需要做特殊处理或提示
        n_components = min(2, X_scaled.shape[1])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        if n_components < 2:
            print("⚠️ 提示: 您指定的特征维度小于 2，PCA 降维图将扩展为伪二维展示。")
            X_pca = np.column_stack((X_pca[:, 0], np.zeros_like(X_pca[:, 0])))
            exp = [100.0, 0.0]
        else:
            exp = pca.explained_variance_ratio_

        plt.figure(figsize=(3.2, 2.2), dpi=600)
        ax = plt.gca()

        palette = sns.color_palette('Set2', n_colors=len(self.class_names))
        color_map = dict(zip(self.class_names, palette))

        sns.scatterplot(
            x=X_pca[:, 0], y=X_pca[:, 1],
            hue=self.y, palette=color_map,
            s=18, edgecolor='black', linewidth=0.3, zorder=3, ax=ax, legend=False
        )

        for idx, cls in enumerate(self.class_names):
            mask = (self.y_encoded == idx)
            x_cls = X_pca[mask, 0]
            y_cls = X_pca[mask, 1]
            if len(x_cls) < 3:
                continue

            mean_x, mean_y = np.mean(x_cls), np.mean(y_cls)
            cov = np.cov(x_cls, y_cls)
            vals, vecs = np.linalg.eigh(cov)

            val_w = max(0.1, np.sqrt(max(1e-5, vals[1])))
            val_h = max(0.1, np.sqrt(max(1e-5, vals[0])))

            width = 2.2 * val_w
            height = 2.2 * val_h
            theta = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))

            ell = Ellipse(xy=(mean_x, mean_y), width=width, height=height, angle=theta,
                          edgecolor=color_map[cls], lw=0.6, facecolor='none', linestyle='-', zorder=2)
            ax.add_patch(ell)

            ax.text(mean_x, mean_y + height / 2 + 0.25, cls,
                    color=color_map[cls], fontsize=6.5, fontweight='bold',
                    ha='center', va='bottom', zorder=4)

        x_margin, y_margin = 1.2, 1.0
        ax.set_xlim(X_pca[:, 0].min() - x_margin, X_pca[:, 0].max() + x_margin)
        ax.set_ylim(X_pca[:, 1].min() - y_margin, X_pca[:, 1].max() + y_margin)

        ax.tick_params(axis='both', labelsize=6, width=0.6)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

        plt.xlabel(f'Factor 1 ({exp[0] * 100:.1f}%)', fontsize=6.5, fontweight='bold')
        plt.ylabel(f'Factor 2 ({exp[1] * 100:.1f}%)', fontsize=6.5, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--', zorder=1)
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        feat_tag = "_".join(self.feature_cols)
        save_path = f'outputs/figures/pca_{feat_tag}_conc_{self.target_concentration}.tif'
        plt.savefig(save_path, format='tiff', dpi=800, bbox_inches='tight', pil_kwargs={'compression': 'tif_lzw'})
        plt.show()
        print(f"高分辨率 TIF 格式 PCA 图已成功保存至: {save_path}")

    def plot_confusion_matrix(self):
        """基于用户指定特征绘制分类混淆矩阵图"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_predict

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        y_pred_encoded = cross_val_predict(clf, X_scaled, self.y_encoded, cv=min(3, len(np.unique(self.y_encoded))))

        cm = confusion_matrix(self.y_encoded, y_pred_encoded)

        plt.figure(figsize=(3.5, 3.0), dpi=600)
        ax = plt.gca()

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    annot_kws={"size": 7}, ax=ax)

        ax.tick_params(axis='both', labelsize=6)
        plt.title(f'Confusion Matrix (Features: {", ".join(self.feature_cols)})', fontsize=7, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=6.5, fontweight='bold')
        plt.ylabel('True Label', fontsize=6.5, fontweight='bold')

        plt.tight_layout()
        feat_tag = "_".join(self.feature_cols)
        save_path = f'outputs/figures/confusion_matrix_{feat_tag}_conc_{self.target_concentration}.png'
        plt.savefig(save_path, dpi=800, bbox_inches='tight')
        plt.show()
        print(f"混淆矩阵图已成功保存至: {save_path}")

    def plot_feature_response_curve(self):
        """【升级版】绘制用户指定的特征参数随浓度变化的响应曲线图"""
        print("\n--- 绘制指定特征随浓度的响应曲线 ---")
        antibiotic_input = input("请输入要分析的抗生素名称 (例如 DOX): ").strip()

        df_full = pd.read_excel(self.data_path)
        df_sub = df_full[df_full['antibiotic'] == antibiotic_input]

        if len(df_sub) == 0:
            print(f"未能在数据源中找到抗生素 '{antibiotic_input}'！")
            return

        # 针对用户当前指定的特征动态创建子图
        n_feats = len(self.feature_cols)
        fig, axes = plt.subplots(1, n_feats, figsize=(3.5 * n_feats, 3.2), dpi=600)
        if n_feats == 1:
            axes = [axes]

        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

        for i, metric in enumerate(self.feature_cols):
            ax = axes[i]
            grouped = df_sub.groupby('concentration')[metric].agg(['mean', 'std']).reset_index()
            grouped = grouped.sort_values('concentration')

            color = colors[i % len(colors)]
            ax.errorbar(
                grouped['concentration'], grouped['mean'], yerr=grouped['std'],
                fmt='-o', color=color, ecolor='#b0b0b0', elinewidth=1.0,
                capsize=2.5, markersize=4.5, linewidth=1.3, zorder=3
            )

            ax.set_title(f'{metric} Response', fontsize=7, fontweight='bold', pad=6)
            ax.set_xlabel('Concentration', fontsize=6.5, fontweight='bold')
            ax.set_ylabel(f'Feature Value', fontsize=6.5, fontweight='bold')

            ax.tick_params(axis='both', labelsize=5.5, width=0.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            ax.grid(True, alpha=0.3, linestyle='--', zorder=1)

        plt.suptitle(f'Response Curves for {antibiotic_input}', fontsize=8, fontweight='bold', y=1.02)
        plt.tight_layout()

        feat_tag = "_".join(self.feature_cols)
        save_path = f'outputs/figures/response_curve_{antibiotic_input}_{feat_tag}.png'
        plt.savefig(save_path, dpi=800, bbox_inches='tight')
        plt.show()
        print(f"指定特征响应曲线图已成功保存至: {save_path}")

    def plot_feature_heatmap(self):
        """绘制抗生素种类与用户指定特征的平均响应热图"""
        print("\n--- 绘制指定特征响应热图 ---")

        df_full = pd.read_excel(self.data_path)
        df_full = df_full[df_full['antibiotic'] != 'Unknown'].reset_index(drop=True)

        feature_means = df_full.groupby('antibiotic')[self.feature_cols].mean()
        feature_normalized = (feature_means - feature_means.min()) / (feature_means.max() - feature_means.min() + 1e-5)

        plt.figure(figsize=(4.2, 3.0), dpi=600)
        ax = plt.gca()

        sns.heatmap(
            feature_normalized, annot=True, fmt=".2f", cmap='Blues',
            cbar=True, linewidths=0.8, linecolor='white',
            annot_kws={"size": 8, "weight": "bold"}, ax=ax
        )

        plt.title('Normalized Feature Response Heatmap', fontsize=8, fontweight='bold', pad=10)
        plt.xlabel('Selected Features', fontsize=7, fontweight='bold')
        plt.ylabel('Antibiotic Species', fontsize=7, fontweight='bold')

        ax.tick_params(axis='x', rotation=0, labelsize=7)
        ax.tick_params(axis='y', rotation=0, labelsize=7)

        plt.tight_layout()
        feat_tag = "_".join(self.feature_cols)
        save_path = f'outputs/figures/heatmap_{feat_tag}.png'
        plt.savefig(save_path, dpi=800, bbox_inches='tight')
        plt.show()
        print(f"特征响应热图已成功保存至: {save_path}")

    def plot_pairplot(self):
        """绘制用户指定特征的配对散点矩阵图 (Pairplot)"""
        print("\n--- 绘制指定特征配对散点矩阵图 (Pairplot) ---")

        df_full = pd.read_excel(self.data_path)
        df_full = df_full[df_full['antibiotic'] != 'Unknown'].reset_index(drop=True)

        subset_cols = self.feature_cols + ['antibiotic']
        hsv_subset = df_full[subset_cols]

        g = sns.pairplot(
            hsv_subset,
            hue='antibiotic',
            palette='Set2',
            diag_kind='kde',
            plot_kws={'s': 20, 'alpha': 0.8, 'edgecolor': 'k', 'linewidth': 0.3},
            diag_kws={'fill': True}
        )

        g.fig.set_size_inches(5.5, 5.5)
        g.fig.suptitle('Pairwise Distribution and Correlation of Selected Features', y=1.02, fontsize=8.5,
                       fontweight='bold')

        plt.tight_layout()
        feat_tag = "_".join(self.feature_cols)
        save_path = f'outputs/figures/pairplot_{feat_tag}.png'
        g.savefig(save_path, dpi=800, bbox_inches='tight')
        plt.show()
        print(f"配对散点矩阵图已成功保存至: {save_path}")


def plot_entropy_reduction(entropy_monitor, save_path='outputs/entropy_reduction_curve.png'):
    """
    根据交叉验证过程中记录的不确定性/熵值，绘制系统熵减收敛曲线图
    """
    if not entropy_monitor:
        print("提示: 没有接收到有效的熵监控数据，无法绘图。")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    folds = list(range(1, len(entropy_monitor) + 1))

    plt.figure(figsize=(8, 5), dpi=600)
    plt.plot(folds, entropy_monitor, marker='o', color='#1f77b4', linewidth=2.5, markersize=6,
             label='模型决策不确定性 (熵减轨迹)')

    plt.title('Concentration Model Training Entropy Reduction Curve', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Cross-Validation Folds / Iteration', fontsize=11)
    plt.ylabel('System Uncertainty / Entropy Measure', fontsize=11)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(folds)
    plt.legend(frameon=True, facecolor='white', edgecolor='none')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"🎉 熵减收敛曲线图已成功绘制并保存至: {save_path}")




if __name__ == '__main__':
    visualizer = ExperimentVisualizer()

    while True:
        print("\n" + "=" * 45)
        print(f" 当前参考特征参数: {visualizer.feature_cols}")
        print(" 请选择要绘制的图表类型：")
        print("  1. PCA 得分与置信椭圆图")
        print("  2. 分类混淆矩阵图")
        print("  3. 指定特征随浓度变化的响应曲线图")
        print("  4. 指定特征响应热图 (Heatmap)")
        print("  5. 指定特征配对散点矩阵图 (Pairplot)")
        print("  0. 退出程序")
        print("=" * 45)

        choice = input("请输入对应的数字编号 (0-5): ").strip()

        if choice == '1':
            visualizer.plot_pca()
        elif choice == '2':
            visualizer.plot_confusion_matrix()
        elif choice == '3':
            visualizer.plot_feature_response_curve()
        elif choice == '4':
            visualizer.plot_feature_heatmap()
        elif choice == '5':
            visualizer.plot_pairplot()
        elif choice == '0':
            print("感谢使用，再见！")
            break
        else:
            print("输入错误，请重新输入 0-5 之间的数字！")