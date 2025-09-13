"""
Contour Context Loop Closure Detection - Correlation Implementation
GMM相关性计算和优化
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
import scipy.linalg as la

from contour_types import GMMOptConfig
from contour_manager import ContourManager


@dataclass
class GMMEllipse:
    """GMM椭圆"""
    cov: np.ndarray  # 2x2协方差矩阵
    mu: np.ndarray  # 2x1均值向量
    w: float  # 权重


class GMMPair:
    """GMM对"""

    def __init__(self, cm_src: ContourManager, cm_tgt: ContourManager,
                 config: GMMOptConfig, T_init: np.ndarray):
        """
        初始化GMM对

        Args:
            cm_src: 源轮廓管理器
            cm_tgt: 目标轮廓管理器
            config: GMM优化配置
            T_init: 初始变换矩阵 (3x3)
        """
        self.scale = config.cov_dilate_scale
        self.ellipses_src: List[List[GMMEllipse]] = []
        self.ellipses_tgt: List[List[GMMEllipse]] = []
        self.selected_pair_idx: List[List[Tuple[int, int]]] = []
        self.src_cell_cnts: List[int] = []
        self.tgt_cell_cnts: List[int] = []
        self.total_cells_src = 0
        self.total_cells_tgt = 0
        self.auto_corr_src = 0.0
        self.auto_corr_tgt = 0.0

        # 提取变换的旋转和平移部分
        R = T_init[:2, :2]
        t = T_init[:2, 2]

        # 收集椭圆数据
        max_majax_src = []
        max_majax_tgt = []

        for lev in config.levels:
            if lev >= len(cm_src.get_config().lv_grads):
                continue

            cnt_src_run = 0
            cnt_src_full = cm_src.get_lev_total_pix(lev)
            cnt_tgt_run = 0
            cnt_tgt_full = cm_tgt.get_lev_total_pix(lev)

            self.ellipses_src.append([])
            self.ellipses_tgt.append([])
            max_majax_src.append([])
            max_majax_tgt.append([])
            self.selected_pair_idx.append([])

            src_layer = cm_src.get_lev_contours(lev)
            tgt_layer = cm_tgt.get_lev_contours(lev)

            # 收集源椭圆
            for view_ptr in src_layer:
                if cnt_src_run / cnt_src_full >= config.min_area_perc:
                    break

                cov = view_ptr.get_manual_cov().astype(float)
                mu = view_ptr.pos_mean.astype(float)
                w = float(view_ptr.cell_cnt)

                self.ellipses_src[-1].append(GMMEllipse(cov, mu, w))
                max_majax_src[-1].append(np.sqrt(view_ptr.eig_vals[1]))
                cnt_src_run += view_ptr.cell_cnt

            # 收集目标椭圆
            for view_ptr in tgt_layer:
                if cnt_tgt_run / cnt_tgt_full >= config.min_area_perc:
                    break

                cov = view_ptr.get_manual_cov().astype(float)
                mu = view_ptr.pos_mean.astype(float)
                w = float(view_ptr.cell_cnt)

                self.ellipses_tgt[-1].append(GMMEllipse(cov, mu, w))
                max_majax_tgt[-1].append(np.sqrt(view_ptr.eig_vals[1]))
                cnt_tgt_run += view_ptr.cell_cnt

            self.src_cell_cnts.append(cnt_src_full)
            self.tgt_cell_cnts.append(cnt_tgt_full)
            self.total_cells_src += cnt_src_full
            self.total_cells_tgt += cnt_tgt_full

        # 预选择椭圆对（需要初始猜测）
        total_pairs = 0
        for li in range(len(self.ellipses_src)):
            for si in range(len(self.ellipses_src[li])):
                for ti in range(len(self.ellipses_tgt[li])):
                    # 变换源椭圆的均值
                    transformed_mu = R @ self.ellipses_src[li][si].mu + t
                    delta_mu = transformed_mu - self.ellipses_tgt[li][ti].mu

                    # 检查距离是否足够接近以进行相关
                    threshold_dist = 3.0 * (max_majax_src[li][si] + max_majax_tgt[li][ti])
                    if np.linalg.norm(delta_mu) < threshold_dist:
                        self.selected_pair_idx[li].append((si, ti))
                        total_pairs += 1

        if total_pairs > 50:  # 只在椭圆对很多时才输出
            print(f"Large GMM correlation: {total_pairs} pairs")

        # 计算自相关
        self._calc_auto_correlation()

    def _calc_auto_correlation(self):
        """计算自相关"""
        for li in range(len(self.ellipses_src)):
            # 源的自相关
            for i in range(len(self.ellipses_src[li])):
                for j in range(len(self.ellipses_src[li])):
                    new_cov = self.scale * (self.ellipses_src[li][i].cov +
                                            self.ellipses_src[li][j].cov)
                    new_mu = self.ellipses_src[li][i].mu - self.ellipses_src[li][j].mu

                    det_cov = np.linalg.det(new_cov)
                    if det_cov > 1e-10:
                        inv_cov = np.linalg.inv(new_cov)
                        quad_form = new_mu.T @ inv_cov @ new_mu
                        self.auto_corr_src += (self.ellipses_src[li][i].w *
                                               self.ellipses_src[li][j].w /
                                               np.sqrt(det_cov) *
                                               np.exp(-0.5 * quad_form))

            # 目标的自相关
            for i in range(len(self.ellipses_tgt[li])):
                for j in range(len(self.ellipses_tgt[li])):
                    new_cov = self.scale * (self.ellipses_tgt[li][i].cov +
                                            self.ellipses_tgt[li][j].cov)
                    new_mu = self.ellipses_tgt[li][i].mu - self.ellipses_tgt[li][j].mu

                    det_cov = np.linalg.det(new_cov)
                    if det_cov > 1e-10:
                        inv_cov = np.linalg.inv(new_cov)
                        quad_form = new_mu.T @ inv_cov @ new_mu
                        self.auto_corr_tgt += (self.ellipses_tgt[li][i].w *
                                               self.ellipses_tgt[li][j].w /
                                               np.sqrt(det_cov) *
                                               np.exp(-0.5 * quad_form))

    def evaluate(self, parameters: np.ndarray) -> float:
        """
        评估目标函数

        Args:
            parameters: [x, y, theta] 变换参数

        Returns:
            目标函数值（负相关）
        """
        x, y, theta = parameters

        # 构造旋转矩阵和平移向量
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        t = np.array([x, y])

        cost = 0.0

        for li in range(len(self.selected_pair_idx)):
            for si, ti in self.selected_pair_idx[li]:
                # 计算变换后的协方差和均值差
                transformed_cov = R @ self.ellipses_src[li][si].cov @ R.T
                new_cov = self.scale * (transformed_cov + self.ellipses_tgt[li][ti].cov)
                transformed_mu = R @ self.ellipses_src[li][si].mu + t
                new_mu = transformed_mu - self.ellipses_tgt[li][ti].mu

                det_cov = np.linalg.det(new_cov)
                if det_cov > 1e-10:
                    inv_cov = np.linalg.inv(new_cov)
                    quad_form = new_mu.T @ inv_cov @ new_mu

                    cost += -(self.ellipses_tgt[li][ti].w *
                              self.ellipses_src[li][si].w /
                              np.sqrt(det_cov) *
                              np.exp(-0.5 * quad_form))

        return cost


class ConstellCorrelation:
    """星座相关类"""

    def __init__(self, cfg: Optional[GMMOptConfig] = None):
        """
        初始化星座相关

        Args:
            cfg: GMM优化配置
        """
        self.cfg = cfg if cfg else GMMOptConfig()
        self.problem_ptr: Optional[GMMPair] = None
        self.auto_corr_src = 0.0
        self.auto_corr_tgt = 0.0
        self.T_best = np.eye(3)

    def init_problem(self, cm_src: ContourManager, cm_tgt: ContourManager,
                     T_init: np.ndarray) -> float:
        """
        初始化问题

        Args:
            cm_src: 源轮廓管理器
            cm_tgt: 目标轮廓管理器
            T_init: 初始变换矩阵

        Returns:
            初始相关分数
        """
        self.T_best = T_init.copy()
        self.problem_ptr = GMMPair(cm_src, cm_tgt, self.cfg, T_init)

        self.auto_corr_src = self.problem_ptr.auto_corr_src
        self.auto_corr_tgt = self.problem_ptr.auto_corr_tgt

        return self.try_problem(T_init)

    def try_problem(self, T_try: np.ndarray) -> float:
        """
        尝试特定变换下的相关性

        Args:
            T_try: 尝试的变换矩阵

        Returns:
            归一化相关分数
        """
        if self.problem_ptr is None:
            return 0.0

        # 从变换矩阵提取参数
        x, y = T_try[:2, 2]
        theta = np.arctan2(T_try[1, 0], T_try[0, 0])
        parameters = np.array([x, y, theta])

        cost = self.problem_ptr.evaluate(parameters)

        # 归一化
        if self.auto_corr_src > 0 and self.auto_corr_tgt > 0:
            return -cost / np.sqrt(self.auto_corr_src * self.auto_corr_tgt)
        else:
            return 0.0

    def calc_correlation(self) -> Tuple[float, np.ndarray]:
        """
        计算相关性并优化

        Returns:
            (相关分数, 优化后的变换矩阵)
        """
        if self.problem_ptr is None:
            return 0.0, self.T_best

        # 从最佳变换矩阵提取初始参数
        x_init, y_init = self.T_best[:2, 2]
        theta_init = np.arctan2(self.T_best[1, 0], self.T_best[0, 0])

        initial_params = np.array([x_init, y_init, theta_init])

        # 优化
        result = minimize(
            fun=self.problem_ptr.evaluate,
            x0=initial_params,
            method='BFGS',
            options={'maxiter': 10, 'disp': False}
        )

        if result.success:
            optimized_params = result.x
        else:
            optimized_params = initial_params

        # 构造优化后的变换矩阵
        x_opt, y_opt, theta_opt = optimized_params

        self.T_best = np.eye(3)
        self.T_best[:2, :2] = np.array([[np.cos(theta_opt), -np.sin(theta_opt)],
                                        [np.sin(theta_opt), np.cos(theta_opt)]])
        self.T_best[:2, 2] = np.array([x_opt, y_opt])

        # 计算最终相关分数
        final_cost = -result.fun if result.success else -self.problem_ptr.evaluate(optimized_params)
        correlation = final_cost / np.sqrt(self.auto_corr_src * self.auto_corr_tgt) if (
                    self.auto_corr_src > 0 and self.auto_corr_tgt > 0) else 0.0

        print(f"Correlation: {correlation:.6f}")

        return correlation, self.T_best

    @staticmethod
    def get_est_sens_tf(T_delta: np.ndarray, bev_config) -> np.ndarray:
        """获取估计的传感器变换"""
        # 忽略非正方形分辨率
        assert bev_config.reso_row == bev_config.reso_col

        # 传感器在BEV原点坐标系中的变换
        T_so_ssen = np.eye(3)
        T_so_ssen[:2, 2] = np.array([bev_config.n_row / 2 - 0.5,
                                     bev_config.n_col / 2 - 0.5])
        T_to_tsen = T_so_ssen.copy()

        # 计算传感器坐标系中的变换 - 修正矩阵乘法顺序
        T_tsen_ssen2_est = np.linalg.inv(T_to_tsen) @ T_delta @ T_so_ssen

        # 缩放平移部分
        T_tsen_ssen2_est[:2, 2] *= bev_config.reso_row

        return T_tsen_ssen2_est

    @staticmethod
    def eval_metric_est(T_delta: np.ndarray, gt_src_3d: np.ndarray,
                        gt_tgt_3d: np.ndarray, bev_config) -> np.ndarray:
        """
        评估度量估计性能

        Args:
            T_delta: 估计的变换
            gt_src_3d: 源的3D真值姿态 (4x4)
            gt_tgt_3d: 目标的3D真值姿态 (4x4)
            bev_config: BEV配置

        Returns:
            误差变换矩阵
        """
        # 获取估计的传感器变换
        T_tsen_ssen2_est = ConstellCorrelation.get_est_sens_tf(T_delta, bev_config)

        # 计算真值的传感器变换
        T_tsen_ssen3 = np.linalg.inv(gt_tgt_3d) @ gt_src_3d

        # 投影到2D
        T_tsen_ssen2_gt = np.eye(3)

        # 计算旋转（使z轴对齐）
        z0 = np.array([0, 0, 1])
        z1 = T_tsen_ssen3[:3, 2]
        ax = np.cross(z0, z1)
        ax_norm = np.linalg.norm(ax)

        if ax_norm > 1e-6:
            ax = ax / ax_norm
            ang = np.arccos(np.clip(np.dot(z0, z1), -1, 1))
            # 罗德里格旋转公式
            K = np.array([[0, -ax[2], ax[1]],
                          [ax[2], 0, -ax[0]],
                          [-ax[1], ax[0], 0]])
            d_rot = np.eye(3) + np.sin(-ang) * K + (1 - np.cos(-ang)) * K @ K
        else:
            d_rot = np.eye(3)

        R_rectified = d_rot @ T_tsen_ssen3[:3, :3]

        # 提取2D旋转和平移
        theta_gt = np.arctan2(R_rectified[1, 0], R_rectified[0, 0])
        T_tsen_ssen2_gt[:2, :2] = np.array([[np.cos(theta_gt), -np.sin(theta_gt)],
                                            [np.sin(theta_gt), np.cos(theta_gt)]])
        T_tsen_ssen2_gt[:2, 2] = T_tsen_ssen3[:3, 2][:2]

        # 计算误差
        T_gt_est = np.linalg.inv(T_tsen_ssen2_gt) @ T_tsen_ssen2_est

        return T_gt_est