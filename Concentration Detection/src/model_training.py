import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def train_and_evaluate_model(excel_path='data/test.xlsx'):
    """
    读取整理好的特征 Excel 表格，执行特征工程、10次分组交叉验证、独立盲测评估，并记录熵减/不确定性指标
    """
    if not pd.io.common.file_exists(excel_path):
        print(f"错误: 找不到数据文件 {excel_path}，请先运行 dataset_builder 生成特征表！")
        return None, None

    # 1. 读取数据
    df_full = pd.read_excel(excel_path)

    if 'R/B' not in df_full.columns and 'R' in df_full.columns and 'B' in df_full.columns:
        df_full['R/B'] = df_full['R'] / (df_full['B'] + 1e-5)

    feature_cols = [col for col in ['R', 'G', 'B', 'R/G', 'R/B', 'G/B', 'r_norm', 'g_norm', 'b_norm',
                                    'H', 'S', 'V', 'L*', 'a*', 'b*'] if col in df_full.columns]

    if not feature_cols:
        print("错误: 表格中未找到任何可用的特征列！")
        return None, None

    print(f"成功加载数据，参与建模的特征列有: {feature_cols}")

    # 2. 依据 group_id 动态拆分
    if 'group_id' in df_full.columns:
        df_full['temp_rep'] = df_full['group_id'].astype(str).str.extract(r'(\d+)')[0].astype(float).fillna(1).astype(
            int)
    else:
        df_full['temp_rep'] = np.repeat(np.arange(1, len(df_full) // 20 + 2), 20)[:len(df_full)]

    max_rep = df_full['temp_rep'].max()
    if max_rep >= 5:
        df_internal = df_full[df_full['temp_rep'] < max_rep].copy()
        df_external_test = df_full[df_full['temp_rep'] == max_rep].copy()
    else:
        split_idx = int(len(df_full) * 0.8)
        df_internal = df_full.iloc[:split_idx].copy()
        df_external_test = df_full.iloc[split_idx:].copy()

    X_internal = df_internal[feature_cols].values
    y_internal = df_internal['concentration'].values
    groups_internal = df_internal['temp_rep'].values

    X_ext = df_external_test[feature_cols].values
    y_ext = df_external_test['concentration'].values

    # 3. 内部 10 次随机分组验证 (记录系统不确定性/熵的收敛轨迹)
    gss = GroupShuffleSplit(n_splits=10, test_size=0.25, random_state=42)

    r2_internal_list = []
    rmse_internal_list = []
    entropy_monitor = []  # 用于记录每轮交叉验证的系统预测不确定性（熵替代指标）

    print("\n--- 开始内部 10 次随机分组验证与熵减追踪 ---")
    for fold_idx, (train_idx, val_idx) in enumerate(gss.split(X_internal, y_internal, groups=groups_internal)):
        X_tr, X_va = X_internal[train_idx], X_internal[val_idx]
        y_tr, y_va = y_internal[train_idx], y_internal[val_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_tr_sc, y_tr)

        preds = model.predict(X_va_sc)

        r2_internal_list.append(r2_score(y_va, preds))
        rmse_internal_list.append(np.sqrt(mean_squared_error(y_va, preds)))

        # 计算当前 Fold 中随机森林多棵树之间的预测标准差（作为模型决策不确定性/熵的度量）
        tree_preds = np.array([tree.predict(X_va_sc) for tree in model.estimators_])
        fold_entropy = np.mean(np.std(tree_preds, axis=0))
        entropy_monitor.append(fold_entropy)

    print(f"内部 10 次抽样评估结果 (Mean ± Std):")
    print(f"  平均 R²  : {np.mean(r2_internal_list):.4f} (±{np.std(r2_internal_list):.4f})")
    print(f"  平均 RMSE: {np.mean(rmse_internal_list):.4f} (±{np.std(rmse_internal_list):.4f})")

    # 4. 全量内部训练与最终独立测试集盲测
    final_scaler = StandardScaler()
    X_internal_sc = final_scaler.fit_transform(X_internal)

    final_model = RandomForestRegressor(n_estimators=100, random_state=42)
    final_model.fit(X_internal_sc, y_internal)

    X_ext_sc = final_scaler.transform(X_ext)
    ext_preds = final_model.predict(X_ext_sc)

    ext_r2 = r2_score(y_ext, ext_preds)
    ext_rmse = np.sqrt(mean_squared_error(y_ext, ext_preds))
    ext_mae = mean_absolute_error(y_ext, ext_preds)

    print("\n" + "=" * 50)
    print(">>> 最终外部独立测试集泛化表现 (Blind Test):")
    print(f"  独立测试集 R²  : {ext_r2:.4f}")
    print(f"  独立测试集 RMSE: {ext_rmse:.4f}")
    print(f"  独立测试集 MAE : {ext_mae:.4f}")
    print("=" * 50)

    # 返回模型、评估指标以及熵减监控数据
    metrics_dict = {
        'ext_r2': ext_r2,
        'ext_rmse': ext_rmse,
        'entropy_monitor': entropy_monitor
    }
    return final_model, metrics_dict


if __name__ == '__main__':
    train_and_evaluate_model()