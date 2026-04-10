"""测试噪音提取算法 - 迭代 TIF 文件"""
import os
import sys
import numpy as np
import cv2

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入预处理页面模块
from densenet_page import DenseNetPage


def test_tif_files():
    """迭代测试所有 TIF 文件"""
    # 查找所有 TIF 文件
    tif_files = []
    search_dirs = [
        'D:/xray-denoiser',
        'D:/xray-denoiser/dataset',
        'D:/xray-denoiser/data',
        'D:/xray-denoiser/images',
    ]

    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for f in files:
                    if f.endswith(('.tif', '.tiff')):
                        tif_files.append(os.path.join(root, f))

    # 也查找当前目录
    for f in os.listdir('.'):
        if f.endswith(('.tif', '.tiff')) and os.path.abspath(os.path.join('.', f)) not in tif_files:
            tif_files.append(os.path.abspath(os.path.join('.', f)))

    print(f"找到 {len(tif_files)} 个 TIF 文件")

    if len(tif_files) == 0:
        print("未找到 TIF 文件，尝试加载 90.tif...")
        tif_path = 'D:/xray-denoiser/90.tif'
        if os.path.exists(tif_path):
            tif_files = [tif_path]
        else:
            print("90.tif 也不存在")
            return

    # 迭代测试
    for tif_path in tif_files:
        print(f"\n{'='*60}")
        print(f"测试：{tif_path}")
        print(f"{'='*60}")

        # 加载图像
        try:
            img_data = np.fromfile(tif_path, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE)

            if img is None or len(img) == 0:
                # 尝试直接读取
                img = cv2.imread(tif_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"  [错误] 无法加载图像")
                continue

            print(f"  图像尺寸：{img.shape[1]}x{img.shape[0]}")
            print(f"  图像类型：{img.dtype}")
            print(f"  亮度范围：{img.min()} - {img.max()}")

            # 创建临时页面对象进行测试
            page = DenseNetPage()

            # 调用噪声参数估计方法
            params = page._estimate_noise_params(img)

            print(f"\n  噪声参数:")
            print(f"    Poisson : {params['poisson_lambda']:.2f}")
            print(f"    AWGN σ: {params['awgn_sigma']:.4f}")
            print(f"    盒子数量：{params['box_count']}")

            # 检查每层的盒子数量
            box_data_list = params.get('box_data_list', [])
            layer1_count = len([b for b in box_data_list if b['layer'] == '1'])
            layer2_count = len([b for b in box_data_list if b['layer'] == '2'])

            print(f"\n  每层盒子数量:")
            print(f"    暗部 (层 1): {layer1_count} 个")
            print(f"    亮部 (层 2): {layer2_count} 个")

            # 检查是否满足至少 2 个
            if layer1_count < 2:
                print(f"    [警告] 暗部盒子不足 2 个!")
            if layer2_count < 2:
                print(f"    [警告] 亮部盒子不足 2 个!")

            # 打印每个盒子的信息
            print(f"\n  盒子详情:")
            for box in box_data_list:
                print(f"    Box {box['label']}: layer={box['layer_name']}, pos={box.get('pos_in_layer', 'N/A')}")

        except Exception as e:
            print(f"  [错误] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    test_tif_files()
