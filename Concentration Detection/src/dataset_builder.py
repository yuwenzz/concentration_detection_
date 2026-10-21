# src/dataset_builder.py
import os
import pandas as pd
import numpy as np
import cv2
import re
from image_processing import extract_image_features  # 调用已写的 ROI 提取模块


def parse_filename(filename):
    """
    专门适配你当前命名格式的解析函数：
    例如：'5 DOX-1(1).jpg' -> 浓度: 5, 抗生素: DOX, 组别: 1-1
    """
    name_without_ext = os.path.splitext(filename)[0]

    # 使用正则表达式匹配：数字(浓度) + 空格 + 字母(抗生素) + 横杠/数字等
    match = re.match(r'^([\d.]+)\s+([A-Za-z]+)-?(.*)$', name_without_ext)

    if match:
        concentration = float(match.group(1))
        antibiotic = match.group(2)
        group_id = match.group(3) if match.group(3) else '1'
    else:
        # 兜底默认值
        concentration = 0.0
        antibiotic = 'Unknown'
        group_id = '1'

    return antibiotic, concentration, group_id


def build_dataset(image_dir='data/raw_images', blank_dir='data/blank_images', output_excel='data/test.xlsx'):
    """
    批量提取 raw_images 下图片的特征，并结合 data/blank_images 中的同组平行空白对照进行底噪扣除与相对特征增强，保存至 test.xlsx
    """
    if not os.path.exists(image_dir):
        print(f"错误: 找不到文件夹 {image_dir}，请先新建该文件夹并放入图片！")
        return

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]

    if not image_files:
        print(f"提示: {image_dir} 文件夹中没有找到任何图片。")
        return

    records = []
    print(f"开始处理 {len(image_files)} 张荧光图片（并进行同组空白对照校正）...")

    for fname in image_files:
        img_path = os.path.join(image_dir, fname)

        # 1. 解析文件名获取基本元数据
        antibiotic, concentration, group_id = parse_filename(fname)

        # 2. 提取当前样本图片的 15 维色彩特征
        features = extract_image_features(img_path)
        if features is None:
            continue

        # 3. 寻找对应的同组平行空白对照图片 (例如 blank-1(1), blank-2(1) 等)
        # 从 group_id 中提取平行样数字编号（例如从 '1(1)' 或 '1' 中提取出 '1'）
        rep_match = re.search(r'\((\d+)\)', group_id) or re.search(r'(\d+)', group_id)
        rep_num = rep_match.group(1) if rep_match else '1'

        blank_features = None
        if os.path.exists(blank_dir):
            possible_blank_names = [f"blank-{rep_num}(1)", f"blank-{rep_num}", f"blank{rep_num}"]
            blank_img_path = None

            for b_name in possible_blank_names:
                for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.PNG', '.JPG']:
                    p = os.path.join(blank_dir, b_name + ext)
                    if os.path.exists(p):
                        blank_img_path = p
                        break
                if blank_img_path:
                    break

            # 如果找到了对应的空白对照，提取其特征用于背景扣除
            if blank_img_path:
                blank_features = extract_image_features(blank_img_path)

        # 4. 组织数据行，对齐你的 Excel 表头结构
        row_data = {
            'image_id': fname,  # 图片文件名
            'antibiotic': antibiotic,  # 抗生素种类 (Species)
            'concentration': concentration,  # 浓度 (Concentration)
            'group_id': group_id,  # 实验组别/批次 (Batch/Group)

            # 基础 RGB 特征
            'R': features.get('R', 0.0),
            'G': features.get('G', 0.0),
            'B': features.get('B', 0.0),

            # 丰富特征集（增强模型表达能力与消融实验）
            'R/G': features.get('R/G', 0.0),
            'R/B': features.get('R/B', 0.0),
            'G/B': features.get('G/B', 0.0),
            'r_norm': features.get('r_norm', 0.0),
            'g_norm': features.get('g_norm', 0.0),
            'b_norm': features.get('b_norm', 0.0),
            'H': features.get('H', 0.0),
            'S': features.get('S', 0.0),
            'V': features.get('V', 0.0),
            'L*': features.get('L*', 0.0),
            'a*': features.get('a*', 0.0),
            'b*': features.get('b*', 0.0)
        }

        # 5. 如果成功匹配到空白对照，计算差分特征（Delta 相对特征，有效消除底噪与光照干扰）
        if blank_features:
            for k in ['R', 'G', 'B', 'H', 'S', 'V', 'L*', 'a*', 'b*']:
                if k in features and k in blank_features:
                    row_data[f'delta_{k}'] = features[k] - blank_features[k]

        records.append(row_data)

    # 6. 转为 DataFrame 并导出保存至 Excel
    df_new = pd.DataFrame(records)

    # 确保 data 目录存在
    os.makedirs(os.path.dirname(output_excel), exist_ok=True)
    df_new.to_excel(output_excel, index=False)

    print(f"数据提取与空白背景校正成功！共计 {len(df_new)} 条样本已更新至: {output_excel}")


if __name__ == '__main__':
    build_dataset()