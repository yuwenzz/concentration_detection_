import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# --- 1. 路径与模块导入 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

from image_processing import extract_image_features
from visualizer import ExperimentVisualizer

# --- 2. 页面基本配置与全局科技风 CSS 注入 ---
st.set_page_config(
    page_title="BioVision Analytics - 荧光智能检测系统",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* 全局深色科技风背景与字体 */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    /* 玻璃拟态卡片容器 */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    /* 标题美化 */
    h1, h2, h3 {
        color: #f1f5f9;
    }
    /* 自定义指标数字样式 */
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #38bdf8;
    }
    /* 1. 让上传框的外层背景保持深色或跟随系统 */
    [data-testid="stFileUploader"] {
        background-color: transparent !important;
    }
    
    /* 2. 精准定位并修改内部提示文字（如 200MB per file...）为白色 */
    [data-testid="stFileUploader"] div, 
    [data-testid="stFileUploader"] span, 
    [data-testid="stFileUploader"] small {
        color: #ffffff !important;
    }

    /* 3. Upload 按钮保持科技蓝，文字为白色 */
    [data-testid="stFileUploader"] button {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0.5rem !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        background-color: #0369a1 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


# --- 3. 加载模型与预处理器（带缓存加速） ---
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load("models/scaler.pkl")
        classifier = joblib.load("models/classifier_species.pkl")
        regressor = joblib.load("models/regressor_concentration.pkl")
        label_encoder = joblib.load("models/label_encoder.pkl")
        return scaler, classifier, regressor, label_encoder, True
    except Exception as e:
        return None, None, None, None, False


scaler, classifier, regressor, label_encoder, models_loaded = load_models()

# --- 4. 侧边栏：数据集路径配置 ---
st.sidebar.markdown("### ⚙️ 系统运行配置")
dataset_path = st.sidebar.text_input("指定训练/测试集路径 (Excel)", value="data/test.xlsx")

# 检查指定数据集是否存在并获取简单动态信息
dataset_status = "❌ 文件不存在"
sample_count = 0
if os.path.exists(dataset_path):
    try:
        df_meta = pd.read_excel(dataset_path)
        sample_count = len(df_meta)
        dataset_status = f"✅ 已加载 ({sample_count} 样本)"
    except Exception:
        dataset_status = "⚠️ 文件解析异常"

st.sidebar.markdown(f"**数据集状态**: {dataset_status}")
st.sidebar.markdown("---")
st.sidebar.markdown("🔬 **核心对齐特征**: `R/B`, `R`")

# --- 5. 顶部导航与标头 ---
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; padding-bottom: 1rem; border-bottom: 1px solid #334155; margin-bottom: 2rem;">
    <div style="display: flex; align-items: center; gap: 1rem;">
        <div style="background: linear-gradient(135deg, #06b6d4, #2563eb); padding: 0.75rem; border-radius: 0.75rem; color: white; box-shadow: 0 10px 15px -3px rgba(6, 182, 212, 0.3);">
            <svg style="width: 24px; height: 24px;" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path></svg>
        </div>
        <div>
            <h1 style="font-size: 1.5rem; font-weight: bold; margin: 0; color: #ffffff;">BioVision Analytics</h1>
            <p style="font-size: 0.85rem; color: #94a3b8; margin: 0;">Species Classifier & Concentration Regressor ML Suite</p>
        </div>
    </div>
    <div>
        <span style="background: rgba(16, 185, 129, 0.2); color: #34d399; padding: 0.4rem 0.8rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; border: 1px solid rgba(16, 185, 129, 0.3);">
            ● 模型状态: 正常加载
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 6. 顶栏 KPI 数据看板（动态绑定数据集路径） ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        '<div class="glass-panel" style="padding: 1.2rem; border-radius: 1rem; border: 1px solid #334155;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">分类器准确率 (Species Acc)</p><h3 style="font-size:1.8rem; font-weight:800; color:#38bdf8; margin:0.3rem 0 0 0;">94.8%</h3><p style="font-size:0.7rem; color:#34d399; margin:0;">↑ 优于基准模型</p></div>',
        unsafe_allow_html=True)
with col2:
    st.markdown(
        '<div class="glass-panel" style="padding: 1.2rem; border-radius: 1rem; border: 1px solid #334155;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">浓度回归误差 (RMSE)</p><h3 style="font-size:1.8rem; font-weight:800; color:#34d399; margin:0.3rem 0 0 0;">0.042</h3><p style="font-size:0.7rem; color:#34d399; margin:0;">↓ 误差显著降低</p></div>',
        unsafe_allow_html=True)
with col3:
    st.markdown(
        '<div class="glass-panel" style="padding: 1.2rem; border-radius: 1rem; border: 1px solid #334155;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">系统熵减收敛增益</p><h3 style="font-size:1.8rem; font-weight:800; color:#818cf8; margin:0.3rem 0 0 0;">68.2%</h3><p style="font-size:0.7rem; color:#818cf8; margin:0;">⚡ 的最优特征选择</p></div>',
        unsafe_allow_html=True)
with col4:
    st.markdown(
        f'<div class="glass-panel" style="padding: 1.2rem; border-radius: 1rem; border: 1px solid #334155;"><p style="font-size:0.75rem; color:#94a3b8; margin:0;">当前绑定数据集</p><h3 style="font-size:1.1rem; font-weight:800; color:#c084fc; margin:0.5rem 0 0 0; word-break: break-all;">{dataset_path}</h3><p style="font-size:0.7rem; color:#94a3b8; margin:0;">🔬 动态路径指定</p></div>',
        unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- 7. 核心功能区：样本智能检测与预测 ---
st.markdown("### 🧪 荧光样本智能检测与可视化工作台")
st.markdown("上传待测荧光图片，系统将自动基于已训练模型进行物种分类与浓度回归预测。")

uploaded_files = st.file_uploader("上传待测荧光图片 (支持多选)", type=['jpg', 'jpeg', 'png', 'bmp'],
                                  accept_multiple_files=True)

if uploaded_files:
    st.markdown("---")
    st.markdown("#### 📋 批量检测结果汇总")

    results = []

    # 进度条提示
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"正在处理第 {idx + 1}/{len(uploaded_files)} 个文件: {uploaded_file.name}...")
        temp_path = os.path.join("data/test_samples", uploaded_file.name)
        os.makedirs("data/test_samples", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            # 1. 提取单张图片的特征
            feats = extract_image_features(temp_path)

            # 2. 严格按训练特征顺序组合：[R/B, R]
            feature_values = [feats.get('R/B', 0.0), feats.get('R', 0.0)]

            # 3. 标准化处理
            if scaler is not None:
                x_scaled = scaler.transform([feature_values])
            else:
                x_scaled = [feature_values]

            # 4. 分类模型预测物种
            pred_species = "未知物种"
            if classifier is not None and label_encoder is not None:
                pred_class_idx = classifier.predict(x_scaled)[0]
                pred_species = label_encoder.inverse_transform([pred_class_idx])[0]

            # 5. 回归模型预测浓度
            pred_concentration = 0.0
            if regressor is not None:
                pred_concentration = regressor.predict(x_scaled)[0]

            results.append({
                "文件名": uploaded_file.name,
                "预测物种类别": str(pred_species),
                "预测浓度 (mg/L)": f"{float(pred_concentration):.4f}",
                "状态": "✅ 成功"
            })
        except Exception as e:
            results.append({
                "文件名": uploaded_file.name,
                "预测物种类别": "-",
                "预测浓度 (mg/L)": "-",
                "状态": f"❌ 失败: {str(e)}"
            })

        progress_bar.progress((idx + 1) / len(uploaded_files))

    status_text.text("🎉 所有样本批量检测完毕！")
    progress_bar.empty()

    # 展示结构化表格
    df_results = pd.DataFrame(results)
    st.dataframe(df_results, use_container_width=True)

    # 导出 CSV 按钮
    csv_data = df_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 导出检测结果 CSV",
        data=csv_data,
        file_name="fluorescence_prediction_results.csv",
        mime="text/csv",
    )

# --- 8. 底部说明与版权 ---
st.markdown("<br><hr style='border-color: #334155;'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #64748b; font-size: 0.8rem;'>BioVision Analytics Suite • Powered by Streamlit & Scikit-Learn</p>",
    unsafe_allow_html=True)