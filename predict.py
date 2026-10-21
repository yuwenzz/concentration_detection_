import os
import joblib
import pandas as pd
from src.image_processing import extract_image_features


def predict_new_samples(image_paths, output_excel="outputs/prediction_results.xlsx"):
    # 1. 加载模型与 Scaling 预处理器（模型保存在根目录下的 models/ 文件夹中）
    scaler = joblib.load("models/scaler.pkl")
    classifier = joblib.load("models/classifier_species.pkl")
    regressor = joblib.load("models/regressor_concentration.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")

    feature_cols = [ 'R/B', 'R']

    results = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"警告: 找不到图片文件 {img_path}，已跳过。")
            continue

        feats = extract_image_features(img_path)

        # 组装特征向量
        x_raw = pd.DataFrame([feats])[feature_cols]
        x_scaled = scaler.transform(x_raw)

        # 2. 模型预测
        pred_species_idx = classifier.predict(x_scaled)[0]
        pred_species = label_encoder.inverse_transform([pred_species_idx])[0]
        pred_conc = regressor.predict(x_scaled)[0]

        feats['Predicted_Species'] = pred_species
        feats['Predicted_Concentration'] = pred_conc
        feats['Image_Name'] = os.path.basename(img_path)
        results.append(feats)

    if len(results) == 0:
        print("没有成功处理任何图片，请检查图片路径！")
        return

    # 3. 确保 outputs 文件夹存在，并输出导出的 Excel 汇总表
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    df_res = pd.DataFrame(results)
    # 把文件名放到最前面，方便查看
    cols = ['Image_Name'] + [c for c in df_res.columns if c != 'Image_Name']
    df_res = df_res[cols]

    df_res.to_excel(output_excel, index=False)
    print(f"预测完成！检测结果已统计保存至: {output_excel}")


if __name__ == "__main__":
    # --- 自动扫描 data/test_samples 文件夹下的所有图片进行批量预测 ---
    samples_dir = "data/test_samples"
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG')

    if not os.path.exists(samples_dir):
        print(f"提示: 文件夹 {samples_dir} 不存在，请检查路径。")
    else:
        # 获取文件夹下所有符合格式的图片
        image_files = [
            os.path.join(samples_dir, f)
            for f in os.listdir(samples_dir)
            if f.endswith(valid_extensions)
        ]

        if len(image_files) == 0:
            print(f"提示: 文件夹 {samples_dir} 下没有找到支持的图片文件！")
        else:
            print(f"成功扫描到 {len(image_files)} 张测试图片，开始批量预测...")
            predict_new_samples(image_files, output_excel="outputs/prediction_results.xlsx")