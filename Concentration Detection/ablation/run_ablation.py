# ablation/run_ablation.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def run_ablation_study():
    # 1. 读取数据
    data_path = 'data/test.xlsx'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到数据文件: {data_path}")

    df = pd.read_excel(data_path)
    df = df[df['antibiotic'] != 'Unknown'].reset_index(drop=True)

    if 'Batch' in df.columns:
        groups = df['antibiotic'].astype(str) + "_" + df['Batch'].astype(str)
    else:
        groups = df.index

    # 2. 定义消融特征组合字典（已将 Baseline 改为 ["R/B", "R"]）
    feature_sets = {
        "Baseline (R/B, R)": ["R/B", "R"],
        "RGB Channels": ["R", "G", "B"],
        "Color Ratios": ["R/G", "R/B", "G/B"],
        "Normalized RGB": ["r_norm", "g_norm", "b_norm"],
        "HSV Space": ["H", "S", "V"],
        "Full Features (All)": ['R', 'G', 'B', 'R/G', 'R/B', 'G/B', 'r_norm', 'g_norm', 'b_norm', 'H', 'S', 'V']
    }

    le = LabelEncoder()
    y = le.fit_transform(df['antibiotic'])

    results = []
    gkf = GroupKFold(n_splits=5)

    print("=" * 50)
    print(" 正在执行特征消融实验 (Ablation Study) ...")
    print("=" * 50)

    for name, feats in feature_sets.items():
        X = df[feats].values
        accs = []
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            if len(X_tr) <= len(np.unique(y_tr)):
                continue

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_va_sc = scaler.transform(X_va)

            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_tr_sc, y_tr)
            preds = clf.predict(X_va_sc)
            accs.append(accuracy_score(y_va, preds))

        mean_acc = np.mean(accs) * 100 if accs else 0.0
        results.append({'Feature Set': name, 'Accuracy (%)': mean_acc})
        print(f"[{name}] 平均准确率: {mean_acc:.4f}%")

    # 3. 保存表格结果
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/figures', exist_ok=True)
    df_res = pd.DataFrame(results)
    df_res.to_excel('outputs/ablation_results.xlsx', index=False)
    print("\n消融实验数据已更新并保存至: outputs/ablation_results.xlsx")

    # ==============================================================================
    # 4. 绘制学术风格柱状图
    # ==============================================================================
    plt.figure(figsize=(6.0, 3.5), dpi=600)
    ax = plt.gca()

    colors = sns.color_palette('Blues_d', n_colors=len(df_res))
    bars = plt.bar(df_res['Feature Set'], df_res['Accuracy (%)'], color=colors, edgecolor='black', linewidth=0.6,
                   width=0.55)

    # 在柱状图上方标出精确数值
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    plt.ylim(0, 105)
    plt.xticks(rotation=20, ha='right', fontsize=6.5, fontweight='bold')
    plt.yticks(fontsize=6, fontweight='bold')

    ax.tick_params(axis='both', width=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

    plt.title('Ablation Study of Feature Combinations', fontsize=7.5, fontweight='bold', pad=10)
    plt.xlabel('Feature Combinations', fontsize=7, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=7, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)

    plt.tight_layout()
    save_path = 'outputs/figures/ablation_bar_chart.png'
    plt.savefig(save_path, dpi=800, bbox_inches='tight')
    plt.show()
    print(f"消融实验柱状图已同步更新并保存至: {save_path}")


if __name__ == '__main__':
    run_ablation_study()