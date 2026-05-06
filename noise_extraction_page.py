"""
噪音提取页面 - 从单张 X 光图像提取噪声特征
"""

import sys
import os
import json
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QProgressBar,
                             QGroupBox, QFrame, QMessageBox, QTextEdit,
                             QScrollArea, QApplication, QCheckBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2

# matplotlib 用于绘制直方图
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class NoiseExtractionThread(QThread):
    """后台线程用于噪音提取 - 从单张 X 光图像提取噪声特征。"""
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(int, str)

    def __init__(self, image_path, output_dir, extraction_method='noise_profile', patch_size=64):
        super().__init__()
        self.image_path = image_path
        self.output_dir = output_dir
        self.extraction_method = extraction_method
        self.patch_size = patch_size

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)

            self.progress.emit(10, "正在加载图像...")
            img_data = np.fromfile(self.image_path, dtype=np.uint8)
            if img_data is None or len(img_data) == 0:
                self.finished.emit(False, "无法读取图像文件")
                return

            noisy_img = cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)
            if noisy_img is None:
                self.finished.emit(False, "无法解码图像")
                return

            self.progress.emit(30, "正在分析噪声特征...")

            noise_params = self._estimate_noise_params(noisy_img)

            self.progress.emit(60, "正在提取噪声图...")

            if self.extraction_method == 'local_std':
                noise_map = self._extract_noise_local_std(noisy_img)
            elif self.extraction_method == 'homogeneous_region':
                noise_map = self._extract_noise_homogeneous(noisy_img)
            else:
                noise_map = self._extract_noise_local_std(noisy_img)

            self.progress.emit(80, "正在保存结果...")

            params_path = os.path.join(self.output_dir, 'noise_params.json')
            with open(params_path, 'w') as f:
                json.dump(noise_params, f, indent=2)

            noise_path = os.path.join(self.output_dir, 'noise_map.png')
            noise_normalized = self._normalize_for_save(noise_map)
            cv2.imwrite(noise_path, noise_normalized)

            src_path = os.path.join(self.output_dir, 'source_image.png')
            cv2.imwrite(src_path, noisy_img)

            self.progress.emit(100, "噪声提取完成")
            self.finished.emit(True, f"噪声提取完成 - Poisson λ={noise_params['poisson_lambda']:.1f}, AWGN σ={noise_params['awgn_sigma']:.2f}")

        except Exception as e:
            import traceback
            self.finished.emit(False, f"提取失败：{e}\n{traceback.format_exc()}")

    @staticmethod
    def _boxes_overlap(cy1, cx1, cy2, cx2, box_h, box_w, gap=5):
        """判断两个盒子（含间隔 gap 像素）是否矩形相交。"""
        return not (cx1 + box_w + gap <= cx2 or cx2 + box_w + gap <= cx1 or
                    cy1 + box_h + gap <= cy2 or cy2 + box_h + gap <= cy1)

    def _estimate_noise_params(self, image):
        """估计噪声参数（Poisson + AWGN）- 基于论文 Rev. Sci. Instrum. 95, 063508 (2024) 方法。"""
        if image.dtype == np.uint16:
            img_float = image.astype(np.float64) / 65535.0
        elif image.dtype == np.uint8:
            img_float = image.astype(np.float64) / 255.0
        else:
            img_float = image.astype(np.float64)
            img_float = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-10)

        if len(img_float.shape) == 3:
            img_gray = cv2.cvtColor(img_float, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_float

        h, w = img_gray.shape
        img_max = float(img_gray.max()) if img_gray.max() > 0 else 1.0

        kernel_size = 15
        local_mean = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)
        local_var = cv2.GaussianBlur((img_gray - local_mean) ** 2, (kernel_size, kernel_size), 0)

        img_uint8 = np.clip(img_gray * 255, 0, 255).astype(np.uint8)
        otsu_thresh, _ = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_OTSU)
        beam_thresh = float(otsu_thresh) / 255.0
        beam_thresh = max(beam_thresh, np.percentile(img_gray, 30))
        beam_mask = img_gray > beam_thresh
        beam_coverage = beam_mask.mean()
        print(f"X 射线活跃区：阈值={beam_thresh:.4f}，覆盖率={beam_coverage:.1%}")

        min_dim = min(h, w)
        box_h = max(30, min(80, int(min_dim * 0.04)))
        box_w = max(30, min(80, int(min_dim * 0.04)))
        step_y = max(15, box_h // 2)
        step_x = max(15, box_w // 2)

        print(f"图像尺寸：{w}x{h}, 盒子大小：{box_w}x{box_h}, 采样步长：{step_x}x{step_y}")

        all_candidates = []
        for i in range(0, h - box_h, step_y):
            for j in range(0, w - box_w, step_x):
                box_mask_region = beam_mask[i:i+box_h, j:j+box_w]
                if box_mask_region.mean() < 0.8:
                    continue
                avg_var  = np.mean(local_var[i:i+box_h, j:j+box_w])
                box_mean = np.mean(img_gray[i:i+box_h, j:j+box_w])
                all_candidates.append((i, j, avg_var, box_mean))

        print(f"活跃区内候选框数量：{len(all_candidates)}")

        if len(all_candidates) < 8:
            print("警告：活跃区候选框不足，回退到全图模式")
            all_candidates = []
            for i in range(0, h - box_h, step_y):
                for j in range(0, w - box_w, step_x):
                    avg_var  = np.mean(local_var[i:i+box_h, j:j+box_w])
                    box_mean = np.mean(img_gray[i:i+box_h, j:j+box_w])
                    all_candidates.append((i, j, avg_var, box_mean))

        brightness_values = [c[3] for c in all_candidates]
        p50 = np.percentile(brightness_values, 50)
        print(f"活跃区亮度中位数：{p50:.4f}")

        dark_regions  = sorted([(cy, cx, v, br) for cy, cx, v, br in all_candidates if br <= p50], key=lambda x: x[2])
        bright_regions = sorted([(cy, cx, v, br) for cy, cx, v, br in all_candidates if br > p50],  key=lambda x: x[2])
        print(f"候选区域：低透射={len(dark_regions)}, 高透射={len(bright_regions)}")

        def pick_nonoverlapping(regions, n=4, min_center_dist=None):
            if min_center_dist is None:
                min_center_dist = 2 * max(box_h, box_w)

            def is_rejected(cy, cx, chosen, use_dist=True):
                for sy, sx, *_ in chosen:
                    if self._boxes_overlap(cy, cx, sy, sx, box_h, box_w):
                        return True
                    if use_dist and ((cy - sy) ** 2 + (cx - sx) ** 2) < min_center_dist ** 2:
                        return True
                return False

            chosen = []
            for cy, cx, var, br in regions:
                if not is_rejected(cy, cx, chosen, use_dist=True):
                    chosen.append((cy, cx, var, br))
                    if len(chosen) >= n:
                        break

            if len(chosen) < n:
                for cy, cx, var, br in regions:
                    if any(cy == sy and cx == sx for sy, sx, *_ in chosen):
                        continue
                    if not is_rejected(cy, cx, chosen, use_dist=False):
                        chosen.append((cy, cx, var, br))
                        if len(chosen) >= n:
                            break
            return chosen

        proj_col = beam_mask.sum(axis=0).astype(np.float64)
        proj_row = beam_mask.sum(axis=1).astype(np.float64)
        sym_x = int(np.average(np.arange(w), weights=proj_col / (proj_col.sum() + 1e-10)))
        sym_y = int(np.average(np.arange(h), weights=proj_row / (proj_row.sum() + 1e-10)))

        half_w = min(sym_x, w - sym_x)
        if half_w > 10:
            l_proj = proj_col[sym_x - half_w: sym_x]
            r_proj = proj_col[sym_x: sym_x + half_w][::-1]
            with np.errstate(invalid='ignore'):
                _corr = np.corrcoef(l_proj, r_proj)[0, 1]
            sym_score = float(_corr) if np.isfinite(_corr) else 0.0
            sym_score = max(0.0, sym_score)
        else:
            sym_score = 0.0
        print(f"对称轴：x={sym_x}, y={sym_y}, 对称分数={sym_score:.3f}")

        _bv = local_var[beam_mask] if beam_mask.sum() > 100 else local_var.ravel()
        _var_q = float(np.percentile(_bv, 55))
        _bv_br = img_gray[beam_mask] if beam_mask.sum() > 100 else img_gray.ravel()
        _bright_lo = float(np.percentile(_bv_br, 25))
        _homog = beam_mask & (local_var < _var_q) & (img_gray > _bright_lo)

        _ks = max(5, min(box_h // 4, box_w // 4))
        if _ks % 2 == 0:
            _ks += 1
        _kern_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_ks, _ks))
        _bin_s  = _homog.astype(np.uint8) * 255
        _cls    = cv2.morphologyEx(_bin_s, cv2.MORPH_CLOSE, _kern_s)
        _opn    = cv2.morphologyEx(_cls,   cv2.MORPH_OPEN,  _kern_s)
        _n_lbl, _, _stats, _centroids = cv2.connectedComponentsWithStats(_opn, connectivity=8)

        _min_area = max(box_h * box_w * 2, int(h * w * 0.002))
        obj_regs = []
        for _i in range(1, _n_lbl):
            if _stats[_i, cv2.CC_STAT_AREA] < _min_area:
                continue
            _cx_r = float(_centroids[_i][0])
            _cy_r = float(_centroids[_i][1])
            _xl   = int(_stats[_i, cv2.CC_STAT_LEFT])
            _xr   = _xl + int(_stats[_i, cv2.CC_STAT_WIDTH])
            _yt   = int(_stats[_i, cv2.CC_STAT_TOP])
            _yb   = _yt + int(_stats[_i, cv2.CC_STAT_HEIGHT])
            _side = 'left' if _cx_r < sym_x else 'right'
            obj_regs.append({'cx': _cx_r, 'cy': _cy_r, 'xl': _xl, 'xr': _xr,
                              'yt': _yt, 'yb': _yb, 'side': _side,
                              'area': _stats[_i, cv2.CC_STAT_AREA]})

        obj_regs.sort(key=lambda r: r['area'], reverse=True)
        print(f"检测到 {len(obj_regs)} 个均匀子区域（方差阈值={_var_q:.2e}）")

        _use_sym = sym_score >= 0.65 and len(obj_regs) >= 2

        def _refine_box(bx0, by0, search_r, x_fixed=False):
            sx = max(1, step_x // 2)
            sy = max(1, step_y // 2)
            best_bx, best_by = bx0, by0
            best_v = float(np.mean(local_var[by0:by0 + box_h, bx0:bx0 + box_w]))
            dx_range = [0] if x_fixed else range(-search_r, search_r + 1, sx)
            for _dy in range(-search_r, search_r + 1, sy):
                for _dx in dx_range:
                    _bx_t = max(0, min(bx0 + _dx, w - box_w))
                    _by_t = max(0, min(by0 + _dy, h - box_h))
                    _v = float(np.mean(local_var[_by_t:_by_t + box_h, _bx_t:_bx_t + box_w]))
                    if _v < best_v:
                        best_v = _v
                        best_bx, best_by = _bx_t, _by_t
            return best_bx, best_by

        _search_r = max(box_w // 3, box_h // 3)

        if _use_sym:
            _sym_boxes = []
            for _reg in obj_regs[:4]:
                _r_out = (_reg['xr'] - _reg['xl']) / 2.0
                _r_out_ext = _r_out * 1.2
                _cx    = _reg['cx']
                _by0   = max(0, min(int(_reg['cy'] - box_h / 2), h - box_h))
                if _reg['side'] == 'left':
                    _bx_out = max(0, int(_cx - _r_out_ext - box_w))
                else:
                    _bx_out = max(0, min(int(_cx + _r_out_ext), w - box_w))
                _scan_sx = max(1, step_x // 2)
                _scan_sy = max(1, step_y // 2)
                _iy_lo = max(0, int(_reg['yt']))
                _iy_hi = max(_iy_lo, min(h - box_h, int(_reg['yb']) - box_h))
                if _reg['side'] == 'left':
                    _scan_xlo = int(_reg['xl'])
                    _scan_xhi = max(_scan_xlo, int(_cx) - box_w)
                else:
                    _scan_xlo = int(_cx)
                    _scan_xhi = max(_scan_xlo, int(_reg['xr']) - box_w)
                _best_bx_in, _best_by_in, _best_bright = _scan_xlo, _by0, -1.0
                for _ty in range(_iy_lo, _iy_hi + 1, _scan_sy):
                    _ty_c = max(0, min(_ty, h - box_h))
                    for _tx in range(_scan_xlo, _scan_xhi + 1, _scan_sx):
                        _tx_c = max(0, min(_tx, w - box_w))
                        _bv = float(np.mean(img_gray[_ty_c:_ty_c + box_h, _tx_c:_tx_c + box_w]))
                        if _bv > _best_bright:
                            _best_bright = _bv
                            _best_bx_in = _tx_c
                            _best_by_in = _ty_c
                _bx_in  = _best_bx_in
                _by0_in = _best_by_in
                _bx_o, _by_o = _refine_box(_bx_out, _by0,    _search_r, x_fixed=True)
                _bx_i, _by_i = _refine_box(_bx_in,  _by0_in, _search_r, x_fixed=True)
                for _bx, _by in [(_bx_o, _by_o), (_bx_i, _by_i)]:
                    _br = float(np.mean(img_gray[_by:_by + box_h, _bx:_bx + box_w]))
                    _sym_boxes.append((_by, _bx, 0.0, _br))

            if _sym_boxes:
                _p50s = np.percentile([_b[3] for _b in _sym_boxes], 50)
                dark_chosen   = [b for b in _sym_boxes if b[3] <= _p50s]
                bright_chosen = [b for b in _sym_boxes if b[3] >  _p50s]
                print(f"对称放框：共 {len(_sym_boxes)} 框，暗={len(dark_chosen)}，亮={len(bright_chosen)}")
            else:
                _use_sym = False

        if not _use_sym:
            print("回退到方差最小非重叠选取")
            dark_chosen   = pick_nonoverlapping(dark_regions,   n=4)
            bright_chosen = pick_nonoverlapping(bright_regions, n=4)

        selected_boxes = dark_chosen + bright_chosen
        dark_count   = len(dark_chosen)
        bright_count = len(bright_chosen)

        lambda_estimates    = []
        noise_std_estimates = []
        box_coords    = []
        box_data_list = []

        layer_labels = ['1', '2']
        sub_labels   = ['a', 'b', 'c', 'd']

        for idx, (cy, cx, _, _) in enumerate(selected_boxes):
            y1, y2 = cy, min(cy + box_h, h)
            x1, x2 = cx, min(cx + box_w, w)
            box = img_gray[y1:y2, x1:x2]
            box_coords.append({'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)})

            box_normalized = box / img_max

            box_mean = float(np.mean(box_normalized))
            box_var  = float(np.var(box_normalized))

            bstd = float(np.std(box_normalized))
            hist_lo = max(0.0, box_mean - 4 * bstd)
            hist_hi = min(1.0, box_mean + 4 * bstd)
            if hist_hi - hist_lo < 1e-6:
                hist_lo, hist_hi = 0.0, 1.0
            hist_range = (hist_lo, hist_hi)

            hist_signal, bin_edges = np.histogram(box_normalized.flatten(), bins=50, range=hist_range)
            hist_signal = hist_signal / (hist_signal.sum() + 1e-10)

            if idx < dark_count:
                layer_idx_val = 0
                layer_name    = '低透射区'
                pos_in_layer  = idx
            else:
                layer_idx_val = 1
                layer_name    = '高透射区'
                pos_in_layer  = idx - dark_count
            box_label = f"{layer_labels[layer_idx_val]}{sub_labels[pos_in_layer]}"

            kernel_size_box = 5
            local_mean_box  = cv2.GaussianBlur(box, (kernel_size_box, kernel_size_box), 0)
            local_var_box   = cv2.GaussianBlur((box - local_mean_box) ** 2,
                                               (kernel_size_box, kernel_size_box), 0)
            flat_thresh     = np.percentile(local_var_box, 20)
            flat_mask       = local_var_box < flat_thresh
            box_awgn_sigma  = 0.0
            if np.sum(flat_mask) > 10:
                noise_std_raw  = float(np.sqrt(np.mean(local_var_box[flat_mask])))
                box_awgn_sigma = noise_std_raw / img_max
                noise_std_estimates.append(box_awgn_sigma)
            awgn_var = box_awgn_sigma ** 2

            from scipy.stats import poisson as _poisson
            from scipy.ndimage import gaussian_filter1d as _gf1d
            from scipy.optimize import minimize_scalar

            bin_centers = np.array([(bin_edges[k] + bin_edges[k + 1]) / 2
                                    for k in range(len(bin_edges) - 1)])
            bin_width   = float(bin_centers[1] - bin_centers[0]) if len(bin_centers) > 1 else 1.0
            k_centers   = np.arange(len(bin_centers), dtype=float)

            hist_smooth = _gf1d(hist_signal.astype(float), sigma=1.5)
            hist_smooth /= hist_smooth.sum() + 1e-10

            lam_bins_init = max(1.0, (box_mean - hist_range[0]) / (bin_width + 1e-10))
            awgn_sigma_bins = max(0.5, box_awgn_sigma / (bin_width + 1e-10))

            def _poisson_gauss_loss(log_lam):
                lam = np.exp(log_lam)
                pmf = _poisson.pmf(k_centers.astype(int), max(0.5, lam))
                pmf /= pmf.sum() + 1e-10
                pmf_conv = _gf1d(pmf.astype(float), sigma=awgn_sigma_bins)
                pmf_conv /= pmf_conv.sum() + 1e-10
                return float(np.sum((pmf_conv - hist_smooth) ** 2))

            try:
                _res = minimize_scalar(
                    _poisson_gauss_loss,
                    bounds=(np.log(max(0.5, lam_bins_init * 0.1)),
                            np.log(lam_bins_init * 10)),
                    method='bounded'
                )
                lam_bins = float(np.exp(_res.x))
            except Exception:
                lam_bins = lam_bins_init

            sigma2_poisson = max(1e-10, lam_bins * bin_width ** 2)
            box_lambda = float(box_mean / sigma2_poisson) if box_mean > 1e-6 else 0.0
            if box_lambda > 0:
                lambda_estimates.append(box_lambda)

            ci_lo, ci_hi = _poisson.interval(0.95, max(1.0, lam_bins))
            box_lambda_lo = hist_range[0] + ci_lo * bin_width
            box_lambda_hi = hist_range[0] + ci_hi * bin_width

            poisson_pmf = _poisson.pmf(k_centers.astype(int), max(0.5, lam_bins))
            poisson_pmf /= poisson_pmf.sum() + 1e-10
            fitted_hist = _gf1d(poisson_pmf.astype(float), sigma=awgn_sigma_bins)
            fitted_hist /= fitted_hist.sum() + 1e-10
            fitted_bins = bin_edges

            noise_residual  = (box - local_mean_box) / img_max
            hist_noise, _   = np.histogram(noise_residual.flatten(), bins=50)
            hist_noise      = hist_noise / (hist_noise.sum() + 1e-10)
            noise_bin_edges = np.linspace(noise_residual.min(), noise_residual.max(), 51)

            box_data_list.append({
                'box_id':       idx + 1,
                'label':        box_label,
                'layer':        layer_labels[layer_idx_val],
                'layer_name':   layer_name,
                'pos_in_layer': pos_in_layer,
                'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                'signal_hist':  hist_signal.tolist(),
                'signal_bins':  bin_edges.tolist(),
                'fitted_hist':  fitted_hist.tolist(),
                'fitted_bins':  fitted_bins.tolist(),
                'hist_range':   list(hist_range),
                'noise_hist':   hist_noise.tolist(),
                'noise_bins':   noise_bin_edges.tolist(),
                'box_mean':       box_mean,
                'box_lambda':     box_lambda,
                'box_lambda_lo':  box_lambda_lo,
                'box_lambda_hi':  box_lambda_hi,
                'awgn_sigma':     box_awgn_sigma,
            })

        layer_groups = {'1': [], '2': []}
        for box in box_data_list:
            layer_groups[box['layer']].append(box)
        for layer, boxes in layer_groups.items():
            for pos, box in enumerate(boxes):
                box['pos_in_layer'] = pos

        if lambda_estimates:
            poisson_lambda = float(np.median(lambda_estimates))
        else:
            global_mean = float(np.mean(img_gray))
            global_var  = float(np.var(img_gray))
            poisson_lambda = global_mean / global_var if global_var > 0 else 10.0

        if noise_std_estimates:
            noise_std = float(np.median(noise_std_estimates))
        else:
            residual  = img_gray - cv2.GaussianBlur(img_gray, (5, 5), 0)
            noise_std = float(np.sqrt(np.mean(
                cv2.GaussianBlur(residual ** 2, (7, 7), 0)))) / img_max

        awgn_sigma = max(0.02, min(0.10, noise_std))

        lambda_range = {
            'min':     float(np.percentile(lambda_estimates, 10)) if lambda_estimates else poisson_lambda * 0.6,
            'max':     float(np.percentile(lambda_estimates, 90)) if lambda_estimates else poisson_lambda * 1.8,
            'nominal': poisson_lambda,
        }
        awgn_range = {
            'min':     max(0.02, awgn_sigma - 0.03),
            'max':     min(0.10, awgn_sigma + 0.05),
            'nominal': awgn_sigma,
        }

        return {
            'poisson_lambda':       float(poisson_lambda),
            'poisson_lambda_range': lambda_range,
            'awgn_sigma':           float(awgn_sigma),
            'awgn_sigma_range':     awgn_range,
            'gaussian_blur_sigma':  1.0,
            'estimated_noise_std':  float(noise_std),
            'image_dtype':          str(image.dtype),
            'image_shape':          list(image.shape),
            'box_coords':           box_coords,
            'box_data_list':        box_data_list,
            'box_count':            len(selected_boxes),
            'lambda_estimates_count': len(lambda_estimates),
            'lambda_min': float(np.min(lambda_estimates)) if lambda_estimates else 0.0,
            'lambda_max': float(np.max(lambda_estimates)) if lambda_estimates else 0.0,
            'method':    'homogeneous_region_box',
            'reference': 'Rev. Sci. Instrum. 95, 063508 (2024)',
        }

    def _extract_noise_local_std(self, image):
        """使用局部标准差法提取噪声图。"""
        if image.dtype == np.uint16:
            img_float = image.astype(np.float64) / 65535.0
        else:
            img_float = image.astype(np.float64) / 255.0

        kernel_size = 7
        mean = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), 0)
        squared_diff = (img_float - mean) ** 2
        local_std = np.sqrt(cv2.GaussianBlur(squared_diff, (kernel_size, kernel_size), 0))
        return local_std

    def _extract_noise_homogeneous(self, image):
        """从均匀区域提取噪声。"""
        if image.dtype == np.uint16:
            img_float = image.astype(np.float64) / 65535.0
        else:
            img_float = image.astype(np.float64) / 255.0

        kernel_size = 15
        local_std = cv2.GaussianBlur((img_float - cv2.GaussianBlur(img_float, (5, 5), 0)) ** 2,
                                      (kernel_size, kernel_size), 0)
        local_std = np.sqrt(local_std)
        return local_std

    def _normalize_for_save(self, noise_map):
        """归一化噪声图用于保存。"""
        noise_min = np.min(noise_map)
        noise_max = np.max(noise_map)
        if noise_max - noise_min > 0:
            noise_normalized = (noise_map - noise_min) / (noise_max - noise_min)
        else:
            noise_normalized = noise_map - noise_min
        return (noise_normalized * 255).astype(np.uint8)


class NoiseExtractionPage(QWidget):
    """噪音提取页面 - 从单张 X 光图像提取噪声特征。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.noise_params = None
        self.source_image_path = None
        self.extraction_output_dir = None
        self.selected_boxes = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel("噪音提取")
        title.setStyleSheet("""
            font-size: 24px;
            font-weight: 600;
            color: #1e293b;
            padding: 14px 18px;
            border-left: 4px solid #0ea5e9;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #f0f9ff, stop:1 transparent);
            border-radius: 8px;
        """)
        layout.addWidget(title)

        content = self._create_content()
        layout.addWidget(content)

    def _create_content(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # 左侧控制面板
        left_panel = QFrame()
        left_panel.setObjectName("step1ControlPanel")
        left_panel.setStyleSheet("""
            QFrame#step1ControlPanel {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
            }
        """)
        left_panel.setMinimumWidth(420)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(12)
        left_layout.setContentsMargins(14, 14, 14, 14)

        # 1. 加载 X 光图像
        load_group = QGroupBox("1. 加载 X 光图像")
        load_layout = QVBoxLayout(load_group)
        load_layout.setSpacing(8)
        load_layout.setContentsMargins(10, 10, 10, 14)

        self.source_image_label = QLabel()
        self.source_image_label.setMinimumHeight(280)
        self.source_image_label.setStyleSheet("""
            QLabel {
                background-color: #f8fafc;
                border: 2px dashed #e2e8f0;
                border-radius: 8px;
            }
        """)
        self.source_image_label.setAlignment(Qt.AlignCenter)
        self.source_image_label.setText("图像预览")
        load_layout.addWidget(self.source_image_label)

        self.load_btn = QPushButton("加载")
        self.load_btn.setObjectName("loadBtn")
        self.load_btn.clicked.connect(self.load_source_image)
        self.load_btn.setMinimumHeight(40)
        self.load_btn.setStyleSheet("""
            QPushButton#loadBtn {
                background-color: #f1f5f9;
                color: #64748b;
                border: 1px solid #e2e8f0;
                padding: 10px 20px;
                border-radius: 8px;
                font-weight: 600;
                font-size: 16px;
            }
            QPushButton#loadBtn:hover {
                background-color: #e2e8f0;
                border-color: #0ea5e9;
            }
        """)
        load_layout.addWidget(self.load_btn)

        self.source_info_label = QLabel("未加载图像")
        self.source_info_label.setStyleSheet("color: #94a3b8; font-size: 16px;")
        self.source_info_label.setWordWrap(False)
        self.source_info_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.source_info_label.setFixedHeight(18)
        load_layout.addWidget(self.source_info_label)

        left_layout.addWidget(load_group)

        # 2. 提取参数
        param_group = QGroupBox("2. 提取参数")
        param_group.setObjectName("compactParamGroup")
        param_group.setMaximumHeight(100)
        param_layout = QVBoxLayout(param_group)
        param_layout.setSpacing(6)
        param_layout.setContentsMargins(10, 10, 10, 10)

        method_label = QLabel("使用均匀区域法提取噪音")
        method_label.setObjectName("methodLabel")
        method_label.setStyleSheet("color: #64748b; font-size: 16px;")
        method_label.setWordWrap(False)
        method_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        method_label.setFixedHeight(16)
        param_layout.addWidget(method_label)

        left_layout.addWidget(param_group)

        # 3. 执行按钮
        self.extract_btn = QPushButton("▶ 开始提取噪声")
        self.extract_btn.setObjectName("primaryBtn")
        self.extract_btn.setStyleSheet("""
            QPushButton#primaryBtn {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0ea5e9, stop:1 #0284c7);
                color: white;
                font-size: 16px;
                font-weight: 600;
                padding: 14px 28px;
                border-radius: 12px;
            }
            QPushButton#primaryBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #0369a1);
            }
            QPushButton#primaryBtn:disabled {
                background: #cbd5e1;
                color: #94a3b8;
            }
        """)
        self.extract_btn.clicked.connect(self.start_noise_extraction)
        self.extract_btn.setEnabled(False)
        self.extract_btn.setMinimumHeight(48)
        left_layout.addWidget(self.extract_btn)

        self.no_crop_checkbox = QCheckBox("不裁剪，显示完整图像")
        self.no_crop_checkbox.setChecked(False)
        left_layout.addWidget(self.no_crop_checkbox)

        self.step1_progress = QProgressBar()
        self.step1_progress.setVisible(False)
        self.step1_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                text-align: center;
                height: 24px;
                font-weight: 600;
                font-size: 14px;
                background-color: #f8fafc;
                color: #475569;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0ea5e9, stop:1 #0284c7);
                border-radius: 7px;
            }
        """)
        left_layout.addWidget(self.step1_progress)

        left_layout.addStretch()

        # 右侧面板
        right_top = self._create_params_only_panel()
        right_top.setMinimumWidth(500)

        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setSpacing(16)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(left_panel, 2)
        top_layout.addWidget(right_top, 3)

        # 直方图
        self.histogram_group = QGroupBox("均匀区域窗口强度分布")
        self.histogram_group.setMinimumHeight(350)
        hist_layout = QVBoxLayout(self.histogram_group)
        hist_layout.setSpacing(8)
        hist_layout.setContentsMargins(12, 12, 12, 12)
        self.histogram_label = QLabel()
        self.histogram_label.setAlignment(Qt.AlignCenter)
        self.histogram_label.setMinimumSize(500, 280)
        self.histogram_label.setStyleSheet("""
            QLabel {
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        hist_layout.addWidget(self.histogram_label)

        layout.addWidget(top_widget, 1)
        layout.addWidget(self.histogram_group, 1)

        return widget

    def _create_params_only_panel(self):
        """创建右侧面板：噪音分析说明 + 噪声参数。"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
                padding: 4px;
            }
        """)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameStyle(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")

        scroll_content = QWidget()
        scroll.setWidget(scroll_content)

        main_layout = QVBoxLayout(scroll_content)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(8, 8, 8, 8)

        analysis_group = QGroupBox("基于均匀区域法的噪声估计")
        analysis_group.setMinimumHeight(220)
        analysis_layout = QVBoxLayout(analysis_group)
        analysis_layout.setSpacing(10)
        analysis_layout.setContentsMargins(12, 12, 12, 12)

        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setMinimumHeight(200)
        self.analysis_text.setHtml(self._get_noise_analysis_html())
        self.analysis_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Microsoft YaHei', sans-serif;
                font-size: 16px;
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 12px;
                color: #475569;
                line-height: 1.6;
            }
        """)
        analysis_layout.addWidget(self.analysis_text)
        main_layout.addWidget(analysis_group)

        params_group = QGroupBox("提取的噪声参数")
        params_group.setMinimumHeight(140)
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(8)
        params_layout.setContentsMargins(8, 8, 8, 8)
        self.extracted_params_text = QTextEdit()
        self.extracted_params_text.setReadOnly(True)
        self.extracted_params_text.setMinimumHeight(120)
        self.extracted_params_text.setPlaceholderText("提取噪声参数后显示...")
        self.extracted_params_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', monospace;
                font-size: 16px;
                background-color: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 8px;
                color: #475569;
            }
        """)
        params_layout.addWidget(self.extracted_params_text)
        main_layout.addWidget(params_group)

        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(scroll)

        return panel

    def _get_noise_analysis_html(self):
        """根据实际数据生成噪声分析 HTML。"""
        if not self.noise_params or not self.noise_params.get('box_data_list'):
            return self._get_noise_analysis_placeholder()

        box_data_list = self.noise_params['box_data_list']
        p = self.noise_params

        html = """
        <h4 style="color: #0284c7; margin-top: 10px;">计算流程</h4>
        <ol style="margin: 6px 0; padding-left: 20px; font-size: 14px; color: #64748b;">
        <li>Otsu 阈值检测 X 射线活跃区，排除未曝光背景</li>
        <li>按亮度分为低透射/高透射两层，每层选取方差最小的 4 个不重叠均匀区域盒</li>
        <li>对每个盒子：从局部方差最平坦区域估计 AWGN σ</li>
        <li>对每个盒子：Gaussian blur 平滑直方图，拟合泊松 PMF ⊗ Gaussian(σ_awgn)（最小化 MSE）</li>
        <li>由拟合 λ_bins 反推 σ²_poisson = λ_bins × bin_width²，再得 λ = μ / σ²_poisson；计算 95% CI 泊松区间</li>
        <li>取各盒子估计值的中位数，得到最终噪声参数</li>
        </ol>
        """

        html += """<h4 style="color: #0284c7; margin-top: 16px;">各盒子估计值</h4>"""
        for box in box_data_list:
            lo = box.get('box_lambda_lo', 0)
            hi = box.get('box_lambda_hi', 0)
            html += f"""
            <div style="background: #f1f5f9; padding: 8px; border-radius: 6px; margin-bottom: 8px; border-left: 3px solid #0ea5e9;">
                <strong>Box {box['label']}</strong> ({box['layer_name']})<br>
                <span style="font-size: 14px; color: #64748b;">
                盒内均值 μ = {box['box_mean']:.4f}，AWGN σ = {box.get('awgn_sigma', 0):.4f}<br>
                泊松拟合 → λ = {box['box_lambda']:.2f}　泊松噪声区间 [{lo:.4f}, {hi:.4f}]
                </span>
            </div>
            """

        html += f"""
        <h4 style="color: #0284c7; margin-top: 16px;">聚合结果（中位数）</h4>
        <div style="background: #dcfce7; padding: 10px; border-radius: 6px; margin-top: 8px;">
            <strong>Poisson λ</strong> = {p['poisson_lambda']:.2f}<br>
            <strong>AWGN σ</strong> = {p['awgn_sigma']:.4f}
        </div>
        """

        return html

    def _get_noise_analysis_placeholder(self):
        return """
        <h4 style="color: #0284c7; margin-top: 10px;">计算流程</h4>
        <ol style="margin: 6px 0; padding-left: 20px; font-size: 14px; color: #64748b;">
        <li>Otsu 阈值检测 X 射线活跃区，排除未曝光背景</li>
        <li>按亮度分为低透射/高透射两层，每层选取方差最小的 4 个不重叠均匀区域盒</li>
        <li>对每个盒子：从局部方差最平坦区域估计 AWGN σ</li>
        <li>对每个盒子：将实测强度直方图拟合为高斯曲线（最小化 MSE）</li>
        <li>由拟合 σ 反推 Poisson λ = μ / (σ²_fit − σ²_awgn)</li>
        <li>取各盒子估计值的中位数，得到最终噪声参数</li>
        </ol>
        """

    # ========== 噪音提取方法 ==========

    def load_source_image(self):
        """加载单张 X 光源图像用于噪音提取。"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 X 光图像", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.tif);;所有文件 (*)"
        )

        if file_path:
            self.source_image_path = file_path

            img_data = np.fromfile(file_path, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)

            if img is not None:
                file_name = os.path.basename(file_path)
                shape = img.shape
                dtype = str(img.dtype)

                if img.ndim == 2:
                    h, w = shape
                else:
                    h, w, c = shape

                bit_depth = "8 位" if dtype == "uint8" else "16 位" if dtype == "uint16" else dtype
                info_text = f"文件名：{file_name} | 形状：({h}, {w}) | 数据类型：{dtype} | 位深度：{bit_depth}"
                self.source_info_label.setText(info_text)

            self.extract_btn.setEnabled(True)
            self._display_source_preview(file_path)

    def start_noise_extraction(self):
        """开始噪音提取。"""
        if not self.source_image_path:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return

        self.extraction_output_dir = os.path.join(os.path.dirname(__file__), 'noise_profile_output')
        extraction_method = 'homogeneous_region'

        self.step1_progress.setVisible(True)
        self.step1_progress.setValue(0)
        self.extract_btn.setEnabled(False)

        self.thread = NoiseExtractionThread(
            self.source_image_path,
            self.extraction_output_dir,
            extraction_method
        )
        self.thread.progress.connect(self.update_step1_progress)
        self.thread.finished.connect(self.noise_extraction_finished)
        self.thread.start()

    def update_step1_progress(self, value, message):
        """更新步骤 1 进度。"""
        self.step1_progress.setValue(value)

    def noise_extraction_finished(self, success, message):
        """噪音提取完成回调。"""
        self.step1_progress.setVisible(False)
        self.extract_btn.setEnabled(True)

        if success:
            self._load_noise_params()
            self._display_noise_boxes()

    def _load_noise_params(self):
        """加载提取的噪声参数。"""
        if not self.extraction_output_dir:
            self.extraction_output_dir = os.path.join(os.path.dirname(__file__), 'noise_profile_output')

        params_path = os.path.join(self.extraction_output_dir, 'noise_params.json')

        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                self.noise_params = json.load(f)

            self.extracted_params_text.setHtml(self._get_extracted_params_html())
            self.selected_boxes = self.noise_params.get('box_coords', [])
            self._draw_histograms()
            self.analysis_text.setHtml(self._get_noise_analysis_html())

    def _get_extracted_params_html(self):
        """生成噪声参数显示 HTML。"""
        if not self.noise_params:
            return "<p>暂无噪声参数</p>"

        p = self.noise_params

        html = f"""
        <h3 style="color: #0ea5e9; margin-bottom: 8px; font-size: 15px;">噪声模型</h3>
        <div style="background: #e0f2fe; padding: 8px; border-radius: 6px; margin-bottom: 12px; font-size: 14px;">
            <strong>Poisson(λ)</strong> + <strong>AWGN(σ)</strong> + <strong>Gaussian Blur(σ=1)</strong><br>
            <span style="font-size: 13px; color: #0369a1;">拟合方法：泊松 PMF ⊗ Gaussian 卷积拟合</span>
        </div>

        <h3 style="color: #0ea5e9; margin-bottom: 8px; font-size: 15px;">最终估计结果</h3>
        <div style="background: #f1f5f9; padding: 8px; border-radius: 6px; margin-bottom: 12px; font-size: 14px;">
            <strong>Poisson λ</strong> = {p['poisson_lambda']:.2f}<br>
            <strong>AWGN σ</strong> = {p['awgn_sigma']:.4f}<br>
            <strong>泊松噪声区间</strong> [{p.get('poisson_lambda_range', {}).get('min', 0):.2f}, {p.get('poisson_lambda_range', {}).get('max', 0):.2f}]
        </div>
        """

        return html

    def _draw_histograms(self):
        """绘制每个均匀区域窗口的强度分布直方图。"""
        if not MATPLOTLIB_AVAILABLE:
            self.histogram_label.setText("matplotlib 不可用，无法绘制直方图")
            return

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False

        box_data_list = self.noise_params.get('box_data_list', [])
        if not box_data_list:
            self.histogram_label.clear()
            return

        from matplotlib.lines import Line2D
        from io import BytesIO

        layer_colors = {'1': '#3498db', '2': '#e67e22'}
        layer_names  = {'1': '低透射区', '2': '高透射区'}
        n_pos = max(1, len(set(b.get('pos_in_layer', 0) for b in box_data_list)))
        pos_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'][:n_pos]
        pos_groups = [(i, i) for i in range(n_pos)]

        fig, axes = plt.subplots(1, n_pos, figsize=(7 * n_pos, 7))
        if n_pos == 1:
            axes = [axes]
        fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.30, wspace=0.35)

        for pos_idx, col in pos_groups:
            ax = axes[col]
            group_boxes = sorted(
                [b for b in box_data_list if b.get('pos_in_layer', 0) == pos_idx],
                key=lambda b: b['layer']
            )

            for box_data in group_boxes:
                layer = box_data['layer']
                color = layer_colors[layer]
                label = box_data['label']
                lam   = box_data.get('box_lambda', 0)
                sig   = box_data.get('awgn_sigma', 0)

                sig_hist = box_data.get('signal_hist', [])
                sig_bins = box_data.get('signal_bins', [])
                if sig_hist and sig_bins:
                    centers = [(sig_bins[i] + sig_bins[i+1]) / 2 for i in range(len(sig_hist))]
                    ax.plot(centers, sig_hist, color=color, linewidth=2.5,
                            label=f"Box {label} ({layer_names[layer]})  λ={lam:.1f}  σ={sig:.3f}")

                fit_hist = box_data.get('fitted_hist', [])
                fit_bins = box_data.get('fitted_bins', [])
                if fit_hist and fit_bins:
                    fcenters = [(fit_bins[i] + fit_bins[i+1]) / 2 for i in range(len(fit_hist))]
                    ax.plot(fcenters, fit_hist, color=color, linewidth=1.5,
                            linestyle='--', alpha=0.75)

            legend_handles = []
            for b in group_boxes:
                layer = b['layer']
                color = layer_colors[layer]
                lam   = b.get('box_lambda', 0)
                sig   = b.get('awgn_sigma', 0)
                legend_handles.append(
                    Line2D([0], [0], color=color, lw=2.5,
                           label=f"Box {b['label']} ({layer_names[layer]})  λ={lam:.1f}  σ={sig:.3f}")
                )
            legend_handles.append(
                Line2D([0], [0], color='gray', lw=1.5, linestyle='--',
                       label='Fitted (泊松×高斯拟合)')
            )
            ax.legend(handles=legend_handles, fontsize=8, framealpha=0.9,
                      loc='upper center', bbox_to_anchor=(0.5, -0.18),
                      ncol=1, borderaxespad=0)

            pos_letter = pos_labels[pos_idx] if pos_idx < len(pos_labels) else str(pos_idx)
            ax.set_title(f"位置 {pos_letter}", fontsize=11, fontweight='bold')
            ax.set_xlabel('Normalized Intensity', fontsize=9)
            ax.set_ylabel('Probability', fontsize=9)
            ax.grid(True, alpha=0.25, linestyle='--')

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue(), 'PNG')
        self.histogram_label.setPixmap(
            pixmap.scaled(self.histogram_label.size(), Qt.KeepAspectRatio,
                          Qt.SmoothTransformation))
        plt.close(fig)

    def _display_noise_boxes(self):
        """在源图像预览上叠加显示噪声计算选取的区域盒。"""
        if not self.source_image_path or not self.noise_params:
            return

        try:
            img_data = np.fromfile(self.source_image_path, dtype=np.uint8)
            if img_data is None or len(img_data) == 0:
                return

            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                return

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            box_data_list = self.noise_params.get('box_data_list', [])

            if not box_data_list:
                return

            img_h, img_w = rgb_img.shape[:2]

            for box_data in box_data_list:
                x1, y1 = box_data['x1'], box_data['y1']
                x2, y2 = box_data['x2'], box_data['y2']
                label = box_data['label']
                cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (255, 0, 0), 4)
                cv2.putText(rgb_img, label, (x1, y1 - 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 2.4, (255, 0, 0), 4)

            if not self.no_crop_checkbox.isChecked():
                img_h, img_w = rgb_img.shape[:2]
                xs = [b['x1'] for b in box_data_list] + [b['x2'] for b in box_data_list]
                ys = [b['y1'] for b in box_data_list] + [b['y2'] for b in box_data_list]
                bw = max(xs) - min(xs)
                bh = max(ys) - min(ys)
                pad = max(60, int(max(bw, bh) * 0.15))
                cx1 = max(0, min(xs) - pad)
                cy1 = max(0, min(ys) - pad)
                cx2 = min(img_w, max(xs) + pad)
                cy2 = min(img_h, max(ys) + pad)
                rgb_img = rgb_img[cy1:cy2, cx1:cx2]

            h, w = rgb_img.shape[:2]
            rgb_bytes = rgb_img.tobytes()
            q_img = QImage(rgb_bytes, w, h, w * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            label_rect = self.source_image_label.frameGeometry()
            avail_width = label_rect.width() - 60
            avail_height = label_rect.height() - 60

            if avail_width > 0 and avail_height > 0:
                scaled = pixmap.scaled(avail_width, avail_height,
                                       Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.source_image_label.setPixmap(scaled)
            else:
                self.source_image_label.setPixmap(pixmap)

            self.source_image_label.setText("")
            print(f"Successfully displayed {len(box_data_list)} noise estimation boxes")
        except Exception as e:
            print(f"Error displaying noise boxes: {e}")
            import traceback
            traceback.print_exc()

    def _display_source_preview(self, image_path):
        """显示源图像预览。"""
        try:
            img_data = np.fromfile(image_path, dtype=np.uint8)
            if img_data is None or len(img_data) == 0:
                self.source_image_label.setText("无法读取图像文件")
                return

            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                self.source_image_label.setText("无法解码图像")
                return

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if rgb_img.dtype == np.uint16:
                rgb_img = (rgb_img.astype(np.float64) / 65535.0 * 255).astype(np.uint8)
            elif rgb_img.dtype != np.uint8:
                rgb_img = ((rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min()) * 255).astype(np.uint8)

            h, w = rgb_img.shape[:2]
            rgb_bytes = rgb_img.tobytes()
            q_img = QImage(rgb_bytes, w, h, w * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)

            self.source_image_label.repaint()
            QApplication.processEvents()

            label_rect = self.source_image_label.frameGeometry()
            avail_width = label_rect.width() - 60
            avail_height = label_rect.height() - 60

            if avail_width > 0 and avail_height > 0:
                scaled = pixmap.scaled(avail_width, avail_height,
                                       Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.source_image_label.setPixmap(scaled)
            else:
                self.source_image_label.setPixmap(pixmap)

            self.source_image_label.setText("")
        except Exception as e:
            print(f"Display error: {e}")
            import traceback
            traceback.print_exc()
            self.source_image_label.setText(f"预览失败：{str(e)}")

    def apply_medical_style(self):
        """应用 Medical Minimalism 风格。"""
        from PyQt5.QtWidgets import QGroupBox, QComboBox, QSpinBox

        method_label = self.findChild(QLabel, "methodLabel")
        if method_label:
            method_label.setStyleSheet("""
                color: #64748b;
                font-size: 16px;
                padding: 0px;
                margin: 0px;
            """)

        for panel in self.findChildren(QFrame):
            if panel.objectName() in ["step1ControlPanel"]:
                panel.setStyleSheet("""
                    QFrame#step1ControlPanel {
                        background-color: #f8fafc;
                        border-radius: 12px;
                        border: 1px solid #e2e8f0;
                        padding: 20px;
                    }
                """)

        for group in self.findChildren(QGroupBox):
            if group.objectName() == "compactParamGroup":
                group.setStyleSheet("""
                    QGroupBox {
                        font-weight: 600;
                        color: #475569;
                        border: 1px solid #e2e8f0;
                        border-radius: 8px;
                        margin-top: 12px;
                        padding-top: 10px;
                        font-size: 16px;
                        max-height: 100px;
                    }
                    QGroupBox::title {
                        subcontrol-origin: margin;
                        left: 10px;
                        padding: 0 6px;
                        color: #0ea5e9;
                        font-weight: 600;
                        font-size: 16px;
                    }
                """)
            else:
                group.setStyleSheet("""
                    QGroupBox {
                        font-weight: 600;
                        color: #475569;
                        border: 1px solid #e2e8f0;
                        border-radius: 8px;
                        margin-top: 16px;
                        padding-top: 16px;
                        font-size: 16px;
                    }
                    QGroupBox::title {
                        subcontrol-origin: margin;
                        left: 12px;
                        padding: 0 8px;
                        color: #0ea5e9;
                        font-weight: 600;
                        font-size: 16px;
                    }
                """)

        for btn in self.findChildren(QPushButton):
            if "加载" in btn.text() or "选择" in btn.text():
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #f1f5f9;
                        color: #475569;
                        border: 1px solid #cbd5e1;
                        padding: 14px 28px;
                        border-radius: 8px;
                        font-weight: 600;
                        font-size: 16px;
                    }
                    QPushButton:hover {
                        background-color: #e2e8f0;
                        border-color: #0ea5e9;
                    }
                """)
            elif "开始" in btn.text() or btn.objectName() == "primaryBtn":
                btn.setStyleSheet("""
                    QPushButton#primaryBtn {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0ea5e9, stop:1 #0284c7);
                        color: white;
                        border: none;
                        padding: 16px 32px;
                        border-radius: 8px;
                        font-weight: 600;
                        font-size: 16px;
                    }
                    QPushButton#primaryBtn:hover {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #0369a1);
                    }
                    QPushButton#primaryBtn:disabled {
                        background: #cbd5e1;
                        color: #94a3b8;
                    }
                """)

        for progress in self.findChildren(QProgressBar):
            progress.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #e2e8f0;
                    border-radius: 8px;
                    text-align: center;
                    height: 36px;
                    font-weight: 600;
                    font-size: 15px;
                    background-color: #f8fafc;
                    color: #475569;
                }
                QProgressBar::chunk {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0ea5e9, stop:1 #0284c7);
                    border-radius: 7px;
                }
            """)

        for text in self.findChildren(QTextEdit):
            if text.isReadOnly():
                text.setStyleSheet("""
                    QTextEdit {
                        font-family: 'Consolas', 'Courier New', monospace;
                        font-size: 16px;
                        background-color: #f8fafc;
                        border: 1px solid #e2e8f0;
                        border-radius: 8px;
                        padding: 10px;
                        color: #475569;
                    }
                    QTextEdit:focus {
                        border-color: #0ea5e9;
                    }
                """)
