"""
测试均匀区域选择算法 - 使用亮度分位数方法
"""

import cv2
import numpy as np

def estimate_noise_params(image):
    """估计噪声参数（Poisson + AWGN）"""

    # 转换到 float [0, 1]
    if image.dtype == np.uint16:
        img_float = image.astype(np.float64) / 65535.0
    elif image.dtype == np.uint8:
        img_float = image.astype(np.float64) / 255.0
    else:
        img_float = image.astype(np.float64)
        img_float = (img_float - img_float.min()) / (img_float.max() - img_float.min())

    # 转换为灰度
    if len(img_float.shape) == 3:
        img_gray = cv2.cvtColor(img_float, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_float

    h, w = img_gray.shape
    print(f"图像尺寸：{w}x{h}, 数据类型：{image.dtype}")

    # ========== 步骤 1: 计算全图局部方差图 ==========
    kernel_size = 15
    local_mean = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)
    local_var = cv2.GaussianBlur((img_gray - local_mean) ** 2, (kernel_size, kernel_size), 0)

    # ========== 步骤 2: 在均匀区域内选择盒子 ==========
    box_h, box_w = 100, 100
    if h < 200 or w < 200:
        box_h, box_w = min(50, h // 4), min(50, w // 4)

    step_y, step_x = 50, 50
    num_samples = 8

    # 收集所有候选区域
    all_candidates = []
    for i in range(0, h - box_h, step_y):
        for j in range(0, w - box_w, step_x):
            avg_var = np.mean(local_var[i:i+box_h, j:j+box_w])
            box_mean = np.mean(img_gray[i:i+box_h, j:j+box_w])
            all_candidates.append((i, j, avg_var, box_mean))

    # 计算亮度分位数（25%-75%）
    brightness_values = [c[3] for c in all_candidates]
    p25 = np.percentile(brightness_values, 25)
    p75 = np.percentile(brightness_values, 75)
    print(f"亮度 25% 分位：{p25:.4f}, 75% 分位：{p75:.4f}")

    # 选择亮度适中且方差较小的区域
    candidates = [(cy, cx, var, br) for cy, cx, var, br in all_candidates
                  if p25 <= br <= p75]

    print(f"候选区域数量（25%-75%）: {len(candidates)}")

    # 如果没有找到候选区域，放宽亮度范围
    if len(candidates) < 4:
        print("放宽亮度范围到 10%-90%...")
        p10 = np.percentile(brightness_values, 10)
        p90 = np.percentile(brightness_values, 90)
        candidates = [(cy, cx, var, br) for cy, cx, var, br in all_candidates
                      if p10 <= br <= p90]
        print(f"候选区域数量（10%-90%）: {len(candidates)}")

    # 按方差排序
    candidates.sort(key=lambda x: x[2])
    print(f"\n前 10 个候选区域（按方差排序）:")
    for idx, (cy, cx, avg_var, box_mean) in enumerate(candidates[:10]):
        print(f"  {idx+1}: 位置=({cx},{cy}), 方差={avg_var:.8f}, 亮度={box_mean:.4f}")

    # 选择互不重叠的盒子
    selected_boxes = []
    min_distance = box_h // 2

    for cy, cx, avg_var, box_mean in candidates:
        too_close = False
        for sy, sx, _, _ in selected_boxes:
            dist = np.sqrt((cy - sy) ** 2 + (cx - sx) ** 2)
            if dist < min_distance:
                too_close = True
                break
        if not too_close:
            selected_boxes.append((cy, cx, avg_var, box_mean))
            if len(selected_boxes) >= num_samples:
                break

    print(f"\n选中的盒子数量：{len(selected_boxes)}")

    # ========== 步骤 3: 在选中的盒子内估计噪声参数 ==========
    lambda_estimates = []
    box_coords = []

    for idx, (cy, cx, _, _) in enumerate(selected_boxes):
        y1, y2 = cy, min(cy + box_h, h)
        x1, x2 = cx, min(cx + box_w, w)
        box = img_gray[y1:y2, x1:x2]

        box_coords.append({'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)})

        box_max = box.max()
        box_normalized = box / box_max if box_max > 0 else box

        box_mean = np.mean(box_normalized)
        box_var = np.var(box_normalized)

        if box_var > 0 and box_mean > 0:
            box_lambda = box_mean / box_var
            if 2 < box_lambda < 200:
                lambda_estimates.append(box_lambda)
                print(f"  盒子 {idx+1}: 位置=({x1},{y1})-({x2},{y2}), λ={box_lambda:.2f}")
            else:
                print(f"  盒子 {idx+1}: λ超出范围 {box_lambda:.2f}")

    # ========== 步骤 4: 聚合估计结果 ==========
    if lambda_estimates:
        poisson_lambda = np.median(lambda_estimates)
    else:
        poisson_lambda = 10

    print(f"\n最终估计结果:")
    print(f"  Poisson λ = {poisson_lambda:.2f}")
    print(f"  有效估计数量 = {len(lambda_estimates)}")
    print(f"  区域盒坐标 = {box_coords}")

    return box_coords, selected_boxes


if __name__ == '__main__':
    image_path = '90.tif'
    print(f"测试均匀区域选择算法（亮度分位数方法）")
    print(f"输入图像：{image_path}\n")

    # 加载图像
    img_data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

    if img is None:
        print("无法读取图像")
        exit(1)

    print(f"图像加载成功：{img.shape}, dtype={img.dtype}\n")

    # 运行算法
    box_coords, selected_boxes = estimate_noise_params(img)

    if box_coords:
        print(f"\n成功找到 {len(box_coords)} 个均匀区域!")
    else:
        print("\n警告：没有找到任何盒子")
