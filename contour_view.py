"""
Contour Context Loop Closure Detection - Contour View Implementation
轮廓视图实现
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import cv2

from contour_types import (
    ContourViewStatConfig, ContourSimThresConfig, RunningStatRecorder,
    diff_perc, diff_delt
)


class ContourView:
    """轮廓视图类"""

    def __init__(self, level: int, poi_r: int, poi_c: int):
        """
        初始化轮廓视图

        Args:
            level: 轮廓层级
            poi_r: 属于此轮廓的点的行坐标
            poi_c: 属于此轮廓的点的列坐标
        """
        self.level = level
        self.poi = np.array([poi_r, poi_c])

        # 统计摘要
        self.cell_cnt = 0
        self.pos_mean = np.zeros(2, dtype=np.float32)
        self.pos_cov = np.zeros((2, 2), dtype=np.float32)
        self.eig_vals = np.zeros(2, dtype=np.float32)
        self.eig_vecs = np.zeros((2, 2), dtype=np.float32)
        self.eccen = 0.0  # 偏心率，0表示圆形
        self.vol3_mean = 0.0
        self.com = np.zeros(2, dtype=np.float32)  # 质心
        self.ecc_feat = False  # 偏心率特征
        self.com_feat = False  # 质心特征

    def calc_stat_vals(self, rec: RunningStatRecorder, cfg: ContourViewStatConfig):
        """
        从运行数据计算统计值

        Args:
            rec: 运行统计记录器
            cfg: 轮廓视图统计配置
        """
        self.cell_cnt = rec.cell_cnt
        self.pos_mean = (rec.cell_pos_sum / rec.cell_cnt).astype(np.float32)

        self.vol3_mean = rec.cell_vol3 / rec.cell_cnt
        self.com = (rec.cell_vol3_torq / rec.cell_vol3).astype(np.float32)

        # 计算偏心率
        if rec.cell_cnt < cfg.min_cell_cov:
            self.pos_cov = np.eye(2) * cfg.point_sigma * cfg.point_sigma
            self.eig_vals = np.array([cfg.point_sigma, cfg.point_sigma])
            self.eig_vecs = np.eye(2)
            self.ecc_feat = False
            self.com_feat = False
        else:
            # 计算协方差矩阵
            mean_outer = np.outer(self.pos_mean, self.pos_mean)
            self.pos_cov = ((rec.cell_pos_tss.astype(np.float32) -
                             mean_outer * rec.cell_cnt) / (rec.cell_cnt - 1))

            # 特征分解
            eig_vals, eig_vecs = np.linalg.eigh(self.pos_cov)
            # 确保特征值为正
            eig_vals = np.maximum(eig_vals, cfg.point_sigma)

            # 按升序排列
            idx = np.argsort(eig_vals)
            self.eig_vals = eig_vals[idx].astype(np.float32)
            self.eig_vecs = eig_vecs[:, idx].astype(np.float32)

            # 计算偏心率
            self.eccen = np.sqrt(self.eig_vals[1] ** 2 - self.eig_vals[0] ** 2) / self.eig_vals[1]

            self.ecc_feat = self._eccentricity_salient(cfg)
            self.com_feat = self._center_of_mass_salient(cfg)

    def _eccentricity_salient(self, cfg: ContourViewStatConfig) -> bool:
        """检查偏心率是否显著"""
        return (self.cell_cnt > 5 and
                diff_perc(self.eig_vals[0], self.eig_vals[1], 0.2) and
                self.eig_vals[1] > 2.5)

    def _center_of_mass_salient(self, cfg: ContourViewStatConfig) -> bool:
        """检查质心是否显著"""
        return np.linalg.norm(self.com - self.pos_mean) > cfg.com_bias_thres

    def get_manual_cov(self) -> np.ndarray:
        """获取手动计算的协方差矩阵"""
        return self.eig_vecs @ np.diag(self.eig_vals) @ self.eig_vecs.T

    @staticmethod
    def check_sim(cont_src: 'ContourView', cont_tgt: 'ContourView',
                  simthres: ContourSimThresConfig) -> bool:
        """
        检查两个轮廓的相似性

        Args:
            cont_src: 源轮廓
            cont_tgt: 目标轮廓
            simthres: 相似性阈值配置

        Returns:
            是否相似
        """

        # 1. 面积检查
        if (diff_perc(cont_src.cell_cnt, cont_tgt.cell_cnt, simthres.tp_cell_cnt) and
                diff_delt(cont_src.cell_cnt, cont_tgt.cell_cnt, simthres.ta_cell_cnt)):

            return False

        # 2. 大特征值检查
        if (max(cont_src.eig_vals[1], cont_tgt.eig_vals[1]) > 2.0 and
                diff_perc(np.sqrt(cont_src.eig_vals[1]), np.sqrt(cont_tgt.eig_vals[1]),
                          simthres.tp_eigval)):
            return False

        # 3. 小特征值检查
        if (max(cont_src.eig_vals[0], cont_tgt.eig_vals[0]) > 2.0 and
                diff_perc(np.sqrt(cont_src.eig_vals[0]), np.sqrt(cont_tgt.eig_vals[0]),
                          simthres.tp_eigval)):
            return False

        # 4. 平均高度检查
        if (max(cont_src.cell_cnt, cont_tgt.cell_cnt) > 15 and
                diff_delt(cont_src.vol3_mean, cont_tgt.vol3_mean, simthres.ta_h_bar)):
            return False

        # 5. 质心半径检查
        com_r1 = np.linalg.norm(cont_src.com - cont_src.pos_mean)
        com_r2 = np.linalg.norm(cont_tgt.com - cont_tgt.pos_mean)
        if (diff_delt(com_r1, com_r2, simthres.ta_rcom) and
                diff_perc(com_r1, com_r2, simthres.tp_rcom)):
            return False

        return True

    @staticmethod
    def add_contour_res(cont1: 'ContourView', cont2: 'ContourView',
                        cfg: ContourViewStatConfig) -> 'ContourView':
        """
        合并两个轮廓（仅统计部分有用）

        Args:
            cont1: 轮廓1
            cont2: 轮廓2
            cfg: 配置

        Returns:
            合并后的轮廓
        """
        assert cont1.level == cont2.level

        # 重建统计记录器并合并
        media = RunningStatRecorder()
        media.cell_cnt = cont1.cell_cnt + cont2.cell_cnt
        media.cell_pos_sum = (cont1.cell_cnt * cont1.pos_mean +
                              cont2.cell_cnt * cont2.pos_mean).astype(float)
        media.cell_vol3 = (cont1.cell_cnt * cont1.vol3_mean +
                           cont2.cell_cnt * cont2.vol3_mean)
        media.cell_vol3_torq = ((cont1.com * (cont1.cell_cnt * cont1.vol3_mean) +
                                 cont2.com * (cont2.cell_cnt * cont2.vol3_mean))
                                .astype(float))

        # 重建协方差
        cov1_scaled = cont1.pos_cov * (cont1.cell_cnt - 1)
        mean1_outer = cont1.cell_cnt * np.outer(cont1.pos_mean, cont1.pos_mean)
        cov2_scaled = cont2.pos_cov * (cont2.cell_cnt - 1)
        mean2_outer = cont2.cell_cnt * np.outer(cont2.pos_mean, cont2.pos_mean)

        media.cell_pos_tss = (cov1_scaled + mean1_outer +
                              cov2_scaled + mean2_outer).astype(float)

        # 创建结果轮廓
        res = ContourView(cont1.level, cont1.poi[0], cont1.poi[1])
        res.calc_stat_vals(media, cfg)

        return res


def extract_contours_from_bev(bev: np.ndarray, height_threshold: float,
                              config: ContourViewStatConfig) -> List[ContourView]:
    """
    从BEV图像中提取轮廓

    Args:
        bev: BEV图像
        height_threshold: 高度阈值
        config: 配置

    Returns:
        轮廓列表
    """
    # 二值化
    mask = (bev > height_threshold).astype(np.uint8) * 255

    # 连通组件分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)

    contours = []

    # 处理每个连通组件（跳过背景）
    for n in range(1, num_labels):
        area = stats[n, cv2.CC_STAT_AREA]
        if area < config.min_cell_cov:
            continue

        # 获取边界框
        x, y, w, h = stats[n, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT + 4]

        # 计算统计
        rec = RunningStatRecorder()
        poi_r, poi_c = -1, -1

        for i in range(y, y + h):
            for j in range(x, x + w):
                if bev[i, j] > height_threshold:
                    rec.running_stats(i, j, bev[i, j])
                    poi_r, poi_c = i, j

        if poi_r >= 0:
            # 创建轮廓视图
            level = int(height_threshold)  # 简化的层级分配
            contour = ContourView(level, poi_r, poi_c)
            contour.calc_stat_vals(rec, config)
            contours.append(contour)

    # 按面积排序（从大到小）
    contours.sort(key=lambda c: c.cell_cnt, reverse=True)

    return contours