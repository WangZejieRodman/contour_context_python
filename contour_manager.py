"""
Contour Context Loop Closure Detection - Contour Manager
轮廓管理器实现 - 修复版本
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import os

from contour_types import (
    ContourManagerConfig, ContourViewStatConfig, ContourSimThresConfig,
    ConstellationPair, BCI, RelativePoint, RunningStatRecorder,
    RET_KEY_DIM, DIST_BIN_LAYERS, NUM_BIN_KEY_LAYER, BITS_PER_LAYER,
    gauss_pdf, clamp_angle
)
from contour_view import ContourView


class ContourManager:
    """轮廓管理器类"""

    def __init__(self, config: ContourManagerConfig, int_id: int):
        """
        初始化轮廓管理器

        Args:
            config: 配置
            int_id: 整数ID
        """
        self.cfg = config
        self.view_stat_cfg = ContourViewStatConfig()
        self.int_id = int_id
        self.str_id = ""

        # 验证配置
        assert config.n_col % 2 == 0
        assert config.n_row % 2 == 0
        assert len(config.lv_grads) > 0

        # 坐标范围
        self.x_min = -(config.n_row // 2) * config.reso_row
        self.x_max = -self.x_min
        self.y_min = -(config.n_col // 2) * config.reso_col
        self.y_max = -self.y_min

        # 数据存储
        self.bev = None
        self.cont_views: List[List[ContourView]] = [[] for _ in range(len(config.lv_grads))]
        self.cont_perc: List[List[float]] = [[] for _ in range(len(config.lv_grads))]
        self.layer_cell_cnt: List[int] = [0] * len(config.lv_grads)
        self.layer_keys: List[List[np.ndarray]] = [[] for _ in range(len(config.lv_grads))]
        self.layer_key_bcis: List[List[BCI]] = [[] for _ in range(len(config.lv_grads))]

        # BEV像素信息
        self.bev_pixfs: List[Tuple[int, Tuple[float, float, float]]] = []
        self.max_bin_val = -float('inf')
        self.min_bin_val = float('inf')

        # 初始化BEV
        self._init_bev()

    def _init_bev(self):
        """初始化BEV图像"""
        self.bev = np.full((self.cfg.n_row, self.cfg.n_col), -1000.0, dtype=np.float32)

    def hash_point_to_image(self, pt: np.ndarray) -> Tuple[int, int]:
        """
        将点映射到图像坐标

        Args:
            pt: 点坐标 [x, y, z]

        Returns:
            (row, col) 或 (-1, -1) 如果点在范围外
        """
        padding = 1e-2
        x, y = pt[0], pt[1]

        # 检查范围
        if (x < self.x_min + padding or x > self.x_max - padding or
                y < self.y_min + padding or y > self.y_max - padding or
                (y * y + x * x) < self.cfg.blind_sq):
            return -1, -1

        row = int(np.floor(x / self.cfg.reso_row)) + self.cfg.n_row // 2
        col = int(np.floor(y / self.cfg.reso_col)) + self.cfg.n_col // 2

        # 验证范围
        if not (0 <= row < self.cfg.n_row and 0 <= col < self.cfg.n_col):
            return -1, -1

        return row, col

    def point_to_cont_row_col(self, p_in_l: np.ndarray) -> np.ndarray:
        """
        将激光雷达坐标系中的点转换到连续图像坐标系

        Args:
            p_in_l: 激光雷达坐标系中的点 [x, y]

        Returns:
            连续的行列坐标
        """
        continuous_rc = np.array([
            p_in_l[0] / self.cfg.reso_row + self.cfg.n_row / 2 - 0.5,
            p_in_l[1] / self.cfg.reso_col + self.cfg.n_col / 2 - 0.5
        ], dtype=np.float32)
        return continuous_rc

    def make_bev(self, point_cloud: np.ndarray, str_id: str = ""):
        """
        从点云生成BEV图像

        Args:
            point_cloud: 点云数组，形状为 [N, 3] 或 [N, 4]
            str_id: 字符串ID
        """
        """从点云生成BEV图像"""
        print(f"[BEV_DEBUG] {str_id}: pointcloud shape={point_cloud.shape}")
        print(f"[BEV_DEBUG] {str_id}: pointcloud hash={hash(point_cloud.tobytes())}")
        print(f"[BEV_DEBUG] {str_id}: first 3 points=\n{point_cloud[:3]}")

        assert point_cloud.shape[0] > 10, "点云数量太少"

        self.str_id = str_id if str_id else f"scan_{self.int_id}"

        # 清空之前的数据
        self.bev_pixfs.clear()
        self._init_bev()

        tmp_pillars = {}

        # 处理每个点
        for pt in point_cloud:
            row, col = self.hash_point_to_image(pt)
            if row >= 0:
                height = self.cfg.lidar_height + pt[2]

                # 更新最大高度
                if self.bev[row, col] < height:
                    self.bev[row, col] = height
                    # 计算连续坐标
                    coor_f = self.point_to_cont_row_col(pt[:2])
                    hash_key = row * self.cfg.n_col + col
                    tmp_pillars[hash_key] = (coor_f[0], coor_f[1], height)

                # 更新范围
                self.max_bin_val = max(self.max_bin_val, height)
                self.min_bin_val = min(self.min_bin_val, height)

        # 转换为列表格式
        self.bev_pixfs = [(k, v) for k, v in tmp_pillars.items()]
        self.bev_pixfs.sort(key=lambda x: x[0])  # 按hash键排序

        print(f"Max/Min bin height: {self.max_bin_val:.3f} {self.min_bin_val:.3f}")
        print(f"Continuous Pos size: {len(self.bev_pixfs)}")

    def make_contours_recursive(self):
        """递归生成轮廓"""
        full_roi = (0, 0, self.cfg.n_col, self.cfg.n_row)
        mask = np.ones((1, 1), dtype=np.uint8)

        self._make_contour_recursive_helper(full_roi, mask, 0, None)

        # 对每层的轮廓按面积排序并计算百分比
        for ll in range(len(self.cont_views)):

            print(f"[CONTOUR_DEBUG] {self.str_id} Level {ll}: {len(self.cont_views[ll])} contours before sort")

            self.cont_views[ll].sort(key=lambda x: x.cell_cnt, reverse=True)

            # 打印前几个轮廓的信息
            for i, cont in enumerate(self.cont_views[ll][:3]):  # 只看前3个
                print(f"[CONTOUR_DEBUG] {self.str_id} L{ll}S{i}: "
                      f"cells={cont.cell_cnt}, eig_vals=[{cont.eig_vals[0]:.6f}, {cont.eig_vals[1]:.6f}]")

            # 计算层级总像素数
            self.layer_cell_cnt[ll] = sum(cont.cell_cnt for cont in self.cont_views[ll])

            # 计算每个轮廓的面积百分比
            self.cont_perc[ll] = []
            for cont in self.cont_views[ll]:
                if self.layer_cell_cnt[ll] > 0:
                    perc = cont.cell_cnt / self.layer_cell_cnt[ll]
                else:
                    perc = 0.0
                self.cont_perc[ll].append(perc)

        # 生成检索键
        self._make_retrieval_keys()

    def _make_contour_recursive_helper(self, cc_roi: Tuple[int, int, int, int],
                                       cc_mask: np.ndarray, level: int, parent):
        """
        递归轮廓生成辅助函数

        Args:
            cc_roi: 区域 (x, y, width, height)
            cc_mask: 掩码
            level: 当前层级
            parent: 父轮廓
        """
        if level >= len(self.cfg.lv_grads):
            return

        h_min = self.cfg.lv_grads[level]
        x, y, w, h = cc_roi

        # 提取ROI
        bev_roi = self.bev[y:y + h, x:x + w]

        # 阈值化
        thres_roi = (bev_roi > h_min).astype(np.uint8) * 255

        # 如果不是第一层，应用父层掩码
        if level > 0:
            thres_roi = cv2.bitwise_and(thres_roi, thres_roi, mask=cc_mask)

        # 连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            thres_roi, connectivity=8)

        # 处理每个连通组件
        for n in range(1, num_labels):  # 跳过背景
            if stats[n, cv2.CC_STAT_AREA] < self.cfg.min_cont_cell_cnt:
                continue

            # 获取组件的边界框（相对于ROI）
            comp_x, comp_y, comp_w, comp_h = stats[n, :4]

            # 转换为全局坐标
            global_roi = (comp_x + x, comp_y + y, comp_w, comp_h)
            local_roi = (comp_x, comp_y, comp_w, comp_h)

            # 创建组件掩码
            mask_n = (labels[comp_y:comp_y + comp_h, comp_x:comp_x + comp_w] == n).astype(np.uint8)

            # 计算统计
            rec = RunningStatRecorder()
            poi_r, poi_c = -1, -1

            for i in range(comp_h):
                for j in range(comp_w):
                    if mask_n[i, j]:
                        global_r = i + global_roi[1]
                        global_c = j + global_roi[0]
                        poi_r, poi_c = global_r, global_c

                        # 查找连续坐标
                        q_hash = global_r * self.cfg.n_col + global_c
                        pixf = self._search_pixf(q_hash)
                        if pixf:
                            rec.running_stats(pixf[0], pixf[1], self.bev[global_r, global_c])

            if poi_r >= 0:
                # 创建轮廓视图
                contour = ContourView(level, poi_r, poi_c)
                contour.calc_stat_vals(rec, self.view_stat_cfg)
                self.cont_views[level].append(contour)

                # 递归处理下一层
                self._make_contour_recursive_helper(global_roi, mask_n, level + 1, contour)

    def _search_pixf(self, q_hash: int) -> Optional[Tuple[float, float, float]]:
        """搜索像素浮点数据"""
        # 二分搜索
        left, right = 0, len(self.bev_pixfs) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.bev_pixfs[mid][0] == q_hash:
                return self.bev_pixfs[mid][1]
            elif self.bev_pixfs[mid][0] < q_hash:
                left = mid + 1
            else:
                right = mid - 1
        return None

    def _make_retrieval_keys(self):
        """生成检索键"""
        roi_radius_padded = int(np.ceil(self.cfg.roi_radius + 1))

        for ll in range(len(self.cfg.lv_grads)):
            accumulate_cell_cnt = 0

            for seq in range(self.cfg.piv_firsts):
                key = np.zeros(RET_KEY_DIM, dtype=np.float32)
                bci = BCI(seq, ll)

                if seq < len(self.cont_views[ll]):
                    accumulate_cell_cnt += self.cont_views[ll][seq].cell_cnt

                if (seq < len(self.cont_views[ll]) and
                        self.cont_views[ll][seq].cell_cnt >= self.cfg.min_cont_key_cnt):
                    v_cen = self.cont_views[ll][seq].pos_mean
                    r_cen, c_cen = int(v_cen[0]), int(v_cen[1])

                    # 定义搜索区域
                    r_min = max(0, r_cen - roi_radius_padded)
                    r_max = min(self.cfg.n_row - 1, r_cen + roi_radius_padded)
                    c_min = max(0, c_cen - roi_radius_padded)
                    c_max = min(self.cfg.n_col - 1, c_cen + roi_radius_padded)

                    # 生成环形特征
                    key = self._generate_ring_features(v_cen, r_min, r_max, c_min, c_max,
                                                       accumulate_cell_cnt, ll, seq)  # ✅ 传递层级和序列

                    # 生成二进制星座标识
                    self._generate_bci(bci, ll, seq, v_cen)

                self.layer_key_bcis[ll].append(bci)
                self.layer_keys[ll].append(key)

    def _generate_ring_features(self, v_cen: np.ndarray, r_min: int, r_max: int,
                                c_min: int, c_max: int, accumulate_cell_cnt: int,
                                current_level: int, current_seq: int) -> np.ndarray:
        """生成环形特征"""
        key = np.zeros(RET_KEY_DIM, dtype=np.float32)

        # ✅ 修复：使用当前层级的轮廓
        if (current_level < len(self.cont_views) and
                current_seq < len(self.cont_views[current_level])):
            cont = self.cont_views[current_level][current_seq]

            # ✅ 添加详细调试输出
            print(f"[KEY_DEBUG] {self.str_id} L{current_level}S{current_seq}: "
                  f"eig_vals=[{cont.eig_vals[0]:.6f}, {cont.eig_vals[1]:.6f}], "
                  f"cell_cnt={cont.cell_cnt}")

            key[0] = np.sqrt(cont.eig_vals[1] * cont.cell_cnt)  # 最大特征值 * 计数
            key[1] = np.sqrt(cont.eig_vals[0] * cont.cell_cnt)  # 最小特征值 * 计数

            print(f"[KEY_DEBUG] {self.str_id} L{current_level}S{current_seq}: "
                  f"key[0]={key[0]:.6f}, key[1]={key[1]:.6f}")

        key[2] = np.sqrt(accumulate_cell_cnt)

        # 环形分布特征
        num_bins = RET_KEY_DIM - 3
        bin_len = self.cfg.roi_radius / num_bins
        ring_bins = np.zeros(num_bins)

        div_per_bin = 5
        discrete_divs = np.zeros(num_bins * div_per_bin)
        div_len = self.cfg.roi_radius / (num_bins * div_per_bin)
        cnt_point = 0

        # 遍历ROI区域
        for rr in range(r_min, r_max + 1):
            for cc in range(c_min, c_max + 1):
                if self.bev[rr, cc] < self.cfg.lv_grads[current_level]:
                    continue

                # 查找连续坐标
                q_hash = rr * self.cfg.n_col + cc
                pixf = self._search_pixf(q_hash)
                if not pixf:
                    continue

                pos = np.array([pixf[0], pixf[1]])
                dist = np.linalg.norm(pos - v_cen)

                if dist < self.cfg.roi_radius - 1e-2 and self.bev[rr, cc] > self.cfg.lv_grads[current_level]:
                    # ✅ 计算超过当前层级的更高层级数量
                    higher_cnt = 0
                    for ele in range(current_level, len(self.cfg.lv_grads)):
                        if self.bev[rr, cc] > self.cfg.lv_grads[ele]:
                            higher_cnt += 1

                    cnt_point += 1

                    # 使用高斯分布分配到bins
                    for div_idx in range(num_bins * div_per_bin):
                        center = div_idx * div_len + 0.5 * div_len
                        discrete_divs[div_idx] += higher_cnt * gauss_pdf(center, dist, 1.0)

        # 合并bins
        for b in range(num_bins):
            for d in range(div_per_bin):
                ring_bins[b] += discrete_divs[b * div_per_bin + d]
            if cnt_point > 0:
                ring_bins[b] *= bin_len / np.sqrt(cnt_point)

        # 填充键的环形部分
        key[3:3 + num_bins] = ring_bins

        print(f"[KEY_DEBUG] {self.str_id} L{current_level}S{current_seq}: "
              f"final_key[0]={key[0]:.6f}")

        return key

    def _generate_bci(self, bci: BCI, ll: int, seq: int, v_cen: np.ndarray):
        """生成二进制星座标识"""
        for bl in range(NUM_BIN_KEY_LAYER):
            bit_offset = bl * BITS_PER_LAYER
            layer_idx = DIST_BIN_LAYERS[bl]

            # 添加边界检查
            if layer_idx >= len(self.cont_views):
                print(f"Warning: layer_idx {layer_idx} >= len(cont_views) {len(self.cont_views)}")
                continue

            for j in range(min(self.cfg.dist_firsts, len(self.cont_views[layer_idx]))):
                if ll != layer_idx or j != seq:
                    # 计算相对位置
                    vec_cc = self.cont_views[layer_idx][j].pos_mean - v_cen
                    tmp_dist = np.linalg.norm(vec_cc)

                    # 距离范围检查
                    min_dist = 5.43
                    max_dist = (BITS_PER_LAYER - 1) * 1.01 + min_dist

                    if tmp_dist <= min_dist or tmp_dist > max_dist - 1e-3:
                        continue

                    tmp_orie = np.arctan2(vec_cc[1], vec_cc[0])
                    dist_idx = min(int(np.floor((tmp_dist - min_dist) / 1.01)), BITS_PER_LAYER - 1)
                    dist_idx += bit_offset

                    if dist_idx < BITS_PER_LAYER * NUM_BIN_KEY_LAYER:
                        bci.dist_bin[dist_idx] = True
                        bci.nei_pts.append(RelativePoint(layer_idx, j, dist_idx, tmp_dist, tmp_orie))

        # 排序并建立索引段
        if bci.nei_pts:
            bci.nei_pts.sort(key=lambda p: p.bit_pos)

            bci.nei_idx_segs = [0]
            for p1 in range(len(bci.nei_pts)):
                if bci.nei_pts[bci.nei_idx_segs[-1]].bit_pos != bci.nei_pts[p1].bit_pos:
                    bci.nei_idx_segs.append(p1)
            bci.nei_idx_segs.append(len(bci.nei_pts))

    # Getter方法
    def get_lev_retrieval_key(self, level: int) -> List[np.ndarray]:
        """获取指定层级的检索键"""
        return self.layer_keys[level]

    def get_retrieval_key(self, level: int, seq: int) -> np.ndarray:
        """获取指定层级和序列的检索键"""
        return self.layer_keys[level][seq]

    def get_lev_contours(self, level: int) -> List[ContourView]:
        """获取指定层级的轮廓"""
        return self.cont_views[level]

    def get_lev_total_pix(self, level: int) -> int:
        """获取指定层级的总像素数"""
        return self.layer_cell_cnt[level]

    def get_lev_bci(self, level: int) -> List[BCI]:
        """获取指定层级的BCI"""
        return self.layer_key_bcis[level]

    def get_bci(self, level: int, seq: int) -> BCI:
        """获取指定层级和序列的BCI"""
        return self.layer_key_bcis[level][seq]

    def get_str_id(self) -> str:
        """获取字符串ID"""
        return self.str_id

    def get_int_id(self) -> int:
        """获取整数ID"""
        return self.int_id

    def get_config(self) -> ContourManagerConfig:
        """获取配置"""
        return self.cfg

    def get_area_perc(self, level: int, seq: int) -> float:
        """获取面积百分比"""
        return self.cont_perc[level][seq]

    def get_bev_image(self) -> np.ndarray:
        """获取BEV图像"""
        return self.bev.copy()

    def get_contour_image(self, level: int) -> np.ndarray:
        """获取指定层级的轮廓图像"""
        if self.bev is None:
            return np.zeros((self.cfg.n_row, self.cfg.n_col), dtype=np.uint8)

        mask = (self.bev > self.cfg.lv_grads[level]).astype(np.uint8) * 255
        return mask

    def clear_image(self):
        """清理图像以节省内存"""
        self.bev = None

    def resume_image(self):
        """从像素数据恢复图像"""
        self._init_bev()
        for hash_key, (row_f, col_f, elev) in self.bev_pixfs:
            rr = hash_key // self.cfg.n_col
            cc = hash_key % self.cfg.n_col
            if rr < self.cfg.n_row and cc < self.cfg.n_col:
                self.bev[rr, cc] = elev

    @staticmethod
    def check_constell_corresp_sim(src: 'ContourManager', tgt: 'ContourManager',
                                   cstl_in: List[ConstellationPair],
                                   lb: 'ScorePairwiseSim',
                                   cont_sim: ContourSimThresConfig) -> Tuple[
        'ScorePairwiseSim', List[ConstellationPair], List[float]]:
        """
        检查星座对应相似性

        Args:
            src: 源轮廓管理器
            tgt: 目标轮廓管理器
            cstl_in: 输入星座对列表
            lb: 下界阈值
            cont_sim: 轮廓相似性配置

        Returns:
            (分数, 过滤后的星座对, 面积百分比)
        """
        from contour_types import ScorePairwiseSim

        ret = ScorePairwiseSim()
        cstl_out = []
        area_perc = []

        # 检查个体相似性
        for pr in cstl_in:
            if ContourView.check_sim(src.cont_views[pr.level][pr.seq_src],
                                     tgt.cont_views[pr.level][pr.seq_tgt], cont_sim):
                cstl_out.append(pr)

        ret.i_indiv_sim = len(cstl_out)
        if ret.i_indiv_sim < lb.i_indiv_sim:
            return ret, cstl_out, area_perc

        # 检查方向一致性
        if len(cstl_out) > 1:
            # 计算主轴方向
            shaft_src = np.array([0.0, 0.0])
            shaft_tgt = np.array([0.0, 0.0])
            max_norm = 0.0

            for i in range(1, min(len(cstl_out), 10)):
                for j in range(i):
                    curr_shaft_src = (src.cont_views[cstl_out[i].level][cstl_out[i].seq_src].pos_mean -
                                      src.cont_views[cstl_out[j].level][cstl_out[j].seq_src].pos_mean)
                    curr_norm = np.linalg.norm(curr_shaft_src)

                    if curr_norm > max_norm:
                        max_norm = curr_norm
                        shaft_src = curr_shaft_src / curr_norm
                        shaft_tgt = ((tgt.cont_views[cstl_out[i].level][cstl_out[i].seq_tgt].pos_mean -
                                      tgt.cont_views[cstl_out[j].level][cstl_out[j].seq_tgt].pos_mean) / curr_norm)

            # 过滤方向不一致的对
            num_sim = len(cstl_out)
            i = 0
            while i < num_sim:
                sc1 = src.cont_views[cstl_out[i].level][cstl_out[i].seq_src]
                tc1 = tgt.cont_views[cstl_out[i].level][cstl_out[i].seq_tgt]

                if sc1.ecc_feat and tc1.ecc_feat:
                    theta_s = np.arccos(np.clip(np.dot(shaft_src, sc1.eig_vecs[:, 1]), -1, 1))
                    theta_t = np.arccos(np.clip(np.dot(shaft_tgt, tc1.eig_vecs[:, 1]), -1, 1))

                    from contour_types import diff_delt
                    if (diff_delt(theta_s, theta_t, np.pi / 6) and
                            diff_delt(np.pi - theta_s, theta_t, np.pi / 6)):
                        # 移除此对
                        cstl_out[i], cstl_out[num_sim - 1] = cstl_out[num_sim - 1], cstl_out[i]
                        num_sim -= 1
                        continue
                i += 1

            cstl_out = cstl_out[:num_sim]

        ret.i_orie_sim = len(cstl_out)
        if ret.i_orie_sim < lb.i_orie_sim:
            return ret, cstl_out, area_perc

        # 计算面积百分比
        for pair in cstl_out:
            perc = 0.5 * (src.cont_perc[pair.level][pair.seq_src] +
                          tgt.cont_perc[pair.level][pair.seq_tgt])
            area_perc.append(perc)

        return ret, cstl_out, area_perc

    @staticmethod
    def get_tf_from_constell(src: 'ContourManager', tgt: 'ContourManager',
                             cstl_pairs: List[ConstellationPair]) -> np.ndarray:
        """
        从星座计算变换矩阵

        Args:
            src: 源轮廓管理器
            tgt: 目标轮廓管理器
            cstl_pairs: 星座对列表

        Returns:
            2D同构变换矩阵 (3x3)
        """
        num_elem = len(cstl_pairs)
        if num_elem < 3:
            print(f"警告：对应点不足({num_elem}个)，返回单位变换")
            return np.eye(3)

        # 收集对应点
        pointset1 = np.zeros((2, num_elem))  # src
        pointset2 = np.zeros((2, num_elem))  # tgt

        for i, pair in enumerate(cstl_pairs):
            pointset1[:, i] = src.cont_views[pair.level][pair.seq_src].pos_mean
            pointset2[:, i] = tgt.cont_views[pair.level][pair.seq_tgt].pos_mean

        # 使用Umeyama算法计算变换
        T_delta = umeyama_2d(pointset1, pointset2)

        return T_delta


def umeyama_2d(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """
    2D Umeyama算法计算相似变换
    """
    assert src_points.shape == dst_points.shape
    assert src_points.shape[0] == 2

    n = src_points.shape[1]
    if n < 2:
        return np.eye(3)  # 🔧 处理边界情况

    # 计算质心
    mu_src = np.mean(src_points, axis=1, keepdims=True)
    mu_dst = np.mean(dst_points, axis=1, keepdims=True)

    # 中心化
    src_centered = src_points - mu_src
    dst_centered = dst_points - mu_dst

    # 计算协方差矩阵
    C = src_centered @ dst_centered.T / n

    # SVD分解
    U, S, Vt = np.linalg.svd(C)

    # 计算旋转矩阵
    R = Vt.T @ U.T

    # 确保是旋转矩阵（行列式为正）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 🔧 添加数值稳定性检查
    if np.abs(np.linalg.det(R) - 1.0) > 1e-6:
        print(f"Warning: Rotation matrix determinant = {np.linalg.det(R)}")

    # 计算平移
    t = mu_dst - R @ mu_src

    # 构造齐次变换矩阵
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2:3] = t

    return T