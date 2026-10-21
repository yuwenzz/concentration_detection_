import os
import sys

# 将 src 目录加入环境变量
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from dataset_builder import build_dataset
from model_training import train_and_evaluate_model
from visualizer import ExperimentVisualizer, plot_entropy_reduction  # 导入可视化模块与熵减绘图函数


def main():
    print("=" * 60)
    print("步骤 1/2: 正在处理图片文件夹并构建特征数据集（含空白对照校正）...")
    print("=" * 60)
    build_dataset(image_dir='data/raw_images', blank_dir='data/blank_images', output_excel='data/test.xlsx')

    print("\n" + "=" * 60)
    print("步骤 2/2: 特征构建完毕，开始执行模型训练与交叉验证...")
    print("=" * 60)

    # 接收 model_training 返回的最终模型与指标字典
    model, metrics_dict = train_and_evaluate_model(excel_path='data/test.xlsx')

    # 从 metrics_dict 中安全提取熵减监控数据
    entropy_monitor = None
    if metrics_dict and isinstance(metrics_dict, dict):
        entropy_monitor = metrics_dict.get('entropy_monitor', None)

    # 步骤 3/3: 自动化贯通绘图（包含系统熵减收敛曲线 + 核心科研图表）
    print("\n" + "=" * 60)
    print("步骤 3/3: 正在自动生成并保存核心可视化分析图表与熵减收敛曲线...")
    print("=" * 60)

    try:
        # 1. 自动绘制并保存系统熵减收敛曲线图
        if entropy_monitor:
            plot_entropy_reduction(entropy_monitor, save_path='outputs/figures/entropy_reduction_curve.png')
            print("📈 系统熵减收敛曲线图已自动保存至: outputs/figures/entropy_reduction_curve.png")
        else:
            print("⚠️ 提示: 未检测到有效的熵监控数据，跳过熵减曲线绘制。")

        # 2. 实例化可视化类并自动批量输出 PCA、混淆矩阵、热图
        viz = ExperimentVisualizer.__new__(ExperimentVisualizer)
        viz.data_path = 'data/test.xlsx'
        viz.target_concentration = 10.0  # 默认目标浓度
        viz.feature_cols = ['R/B', 'R', 'H']  # 默认选取的特征组合

        # 加载并过滤数据
        viz.load_and_filter_data()

        # 自动批量输出核心图表到 outputs/figures/
        viz.plot_pca()
        viz.plot_confusion_matrix()
        viz.plot_feature_heatmap()

        print("🎉 所有核心科研图表与熵减收敛曲线已全部自动生成，并已就绪存放在 outputs/figures/ 目录下！")
    except Exception as e:
        print(f"⚠️ 自动绘图部分提示: {e}")

    print("\n🎉 全流程自动化流水线运行完毕！")


if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()
    # src/visualizer.py 末尾的交互与自动保存控制逻辑
    if __name__ == '__main__':
        visualizer = ExperimentVisualizer()

        while True:
            print("\n" + "=" * 45)
            print(f" 当前参考特征参数: {visualizer.feature_cols}")
            print(" 请选择要绘制的图表类型（每绘制一种将自动保存至 outputs/figures/）：")
            print("  1. PCA 得分与置信椭圆图 (自动保存为高质量 TIF)")
            print("  2. 分类混淆矩阵图 (自动保存为 PNG)")
            print("  3. 指定特征随浓度变化的响应曲线图 (自动保存为 PNG)")
            print("  4. 指定特征响应热图 (Heatmap) (自动保存为 PNG)")
            print("  5. 指定特征配对散点矩阵图 (Pairplot) (自动保存为 PNG)")
            print("  0. 退出程序")
            print("=" * 45)

            choice = input("请输入对应的数字编号 (0-5): ").strip()

            if choice == '1':
                visualizer.plot_pca()  # 内部已集成 plt.savefig 自动保存
            elif choice == '2':
                visualizer.plot_confusion_matrix()  # 内部已集成自动保存
            elif choice == '3':
                visualizer.plot_feature_response_curve()  # 内部已集成自动保存
            elif choice == '4':
                visualizer.plot_feature_heatmap()  # 内部已集成自动保存
            elif choice == '5':
                visualizer.plot_pairplot()  # 内部已集成自动保存
            elif choice == '0':
                print("感谢使用，再见！")
                break
            else:
                print("输入错误，请重新输入 0-5 之间的数字！")