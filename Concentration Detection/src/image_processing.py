import cv2
import numpy as np

def extract_image_features(image_path, roi_ratio=0.8):
    """
    读取图像并提取多维色彩特征 (RGB, Normalized RGB, HSV, CIE-Lab, 比值)
    :param image_path: 图片路径
    :param roi_ratio: 中心ROI截取比例（默认截取中心 80% 区域，避开边缘杂光）
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")

    # 1. 截取中心 ROI 区域
    h, w, _ = image.shape
    dh, dw = int(h * (1 - roi_ratio) / 2), int(w * (1 - roi_ratio) / 2)
    roi = image[dh:h-dh, dw:w-dw]

    # 2. 转换为 RGB (OpenCV 默认 BGR)
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32)

    # 计算通道均值
    r_mean = np.mean(rgb[:, :, 0])
    g_mean = np.mean(rgb[:, :, 1])
    b_mean = np.mean(rgb[:, :, 2])

    # 比值特征
    rg_ratio = r_mean / (g_mean + 1e-5)
    rb_ratio = r_mean / (b_mean + 1e-5)
    gb_ratio = g_mean / (b_mean + 1e-5)

    # 归一化 RGB
    rgb_sum = r_mean + g_mean + b_mean + 1e-5
    r_norm = r_mean / rgb_sum
    g_norm = g_mean / rgb_sum
    b_norm = b_mean / rgb_sum

    # HSV 色彩空间
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv[:, :, 0])
    s_mean = np.mean(hsv[:, :, 1])
    v_mean = np.mean(hsv[:, :, 2])

    # CIE-Lab 色彩空间
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l_mean = np.mean(lab[:, :, 0])
    a_mean = np.mean(lab[:, :, 1])
    b_lab_mean = np.mean(lab[:, :, 2])

    return {
        'R': r_mean, 'G': g_mean, 'B': b_mean,
        'R/G': rg_ratio, 'R/B': rb_ratio, 'G/B': gb_ratio,
        'r_norm': r_norm, 'g_norm': g_norm, 'b_norm': b_norm,
        'H': h_mean, 'S': s_mean, 'V': v_mean,
        'L*': l_mean, 'a*': a_mean, 'b*': b_lab_mean
    }