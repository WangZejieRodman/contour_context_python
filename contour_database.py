"""
Contour Context Loop Closure Detection - Contour Database
轮廓数据库实现，包含KD树和候选匹配
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Set
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors
import time
import copy

from contour_types import (
    ContourDBConfig, CandidateScoreEnsemble, ConstellationPair,
    ScoreConstellSim, ScorePairwiseSim, ScorePostProc,
    BCI, DistSimPair, RelativePoint, clamp_angle,
    BITS_PER_LAYER, NUM_BIN_KEY_LAYER, DIST_BIN_LAYERS, LAYER_AREA_WEIGHTS
)
from contour_manager import ContourManager
from correlation import ConstellCorrelation, GMMOptConfig


@dataclass
class IndexOfKey:
    """键的索引"""
    gidx: int  # 全局索引
    level: int  # 层级
    seq: int   # 序列


class TreeBucket:
    """树桶类 - 管理单个KD树和缓冲区"""

    def __init__(self, config, beg: float, end: float):
        """
        初始化树桶

        Args:
            config: 树桶配置
            beg: 桶开始值
            end: 桶结束值
        """
        self.cfg = config
        self.buc_beg = beg
        self.buc_end = end
        self.data_tree: List[np.ndarray] = []
        self.tree_ptr: Optional[NearestNeighbors] = None
        self.buffer: List[Tuple[np.ndarray, float, IndexOfKey]] = []  # (key, timestamp, index)
        self.gkidx_tree: List[IndexOfKey] = []

        # 最大距离常量
        self.MAX_DIST_SQ = 1e6

    def get_tree_size(self) -> int:
        """获取树大小"""
        assert len(self.data_tree) == len(self.gkidx_tree)
        return len(self.data_tree)

    def push_buffer(self, tree_key: np.ndarray, ts: float, iok: IndexOfKey):
        """推送到缓冲区"""
        self.buffer.append((tree_key.copy(), ts, iok))

    def need_pop_buffer(self, curr_ts: float) -> bool:
        """检查是否需要弹出缓冲区"""
        ts_overflow = curr_ts - self.cfg.max_elapse
        if not self.buffer or self.buffer[0][1] > ts_overflow:
            return False
        return True

    def rebuild_tree(self):
        """重建KD树"""
        if len(self.data_tree) > 0:
            data_matrix = np.array(self.data_tree)
            self.tree_ptr = NearestNeighbors(
                n_neighbors=min(50, len(self.data_tree)),
                algorithm='kd_tree',
                leaf_size=10
            )
            self.tree_ptr.fit(data_matrix)
        else:
            self.tree_ptr = None

    def pop_buffer_max(self, curr_ts: float):
        """从缓冲区弹出到树中并重建树"""
        ts_cutoff = curr_ts - self.cfg.min_elapse
        gap = 0

        # 找到需要移动的元素数量
        for i, (_, ts, _) in enumerate(self.buffer):
            if ts >= ts_cutoff:
                break
            gap += 1

        if gap > 0:
            # 移动数据到树中
            sz0 = len(self.data_tree)
            self.data_tree.extend([self.buffer[i][0] for i in range(gap)])
            self.gkidx_tree.extend([self.buffer[i][2] for i in range(gap)])

            # 移除已处理的缓冲区项
            self.buffer = self.buffer[gap:]

            # 重建树
            self.rebuild_tree()

    def knn_search(self, num_res: int, q_key: np.ndarray, max_dist_sq: float) -> Tuple[List[IndexOfKey], List[float]]:
        """KNN搜索"""
        ret_idx = []
        out_dist_sq = [self.MAX_DIST_SQ] * num_res

        if self.tree_ptr is None or len(self.data_tree) == 0:
            return ret_idx, out_dist_sq[:0]  # 返回空列表

        # 执行搜索
        k = min(num_res, len(self.data_tree))
        try:
            distances, indices = self.tree_ptr.kneighbors([q_key], n_neighbors=k)

            # 过滤距离并构建结果
            for i in range(k):
                dist_sq = distances[0][i] ** 2
                if dist_sq < max_dist_sq:
                    ret_idx.append(self.gkidx_tree[indices[0][i]])
                    if i < num_res:
                        out_dist_sq[i] = dist_sq
                else:
                    break
        except Exception as e:
            print(f"KNN search error: {e}")
            return [], []

        return ret_idx, out_dist_sq[:len(ret_idx)]

    def range_search(self, max_dist_sq: float, q_key: np.ndarray) -> Tuple[List[IndexOfKey], List[float]]:
        """范围搜索"""
        ret_idx = []
        out_dist_sq = []

        if self.tree_ptr is None or len(self.data_tree) == 0:
            return ret_idx, out_dist_sq

        try:
            # sklearn的radius_neighbors返回的是距离，不是距离的平方
            max_dist = np.sqrt(max_dist_sq)
            indices, distances = self.tree_ptr.radius_neighbors([q_key], radius=max_dist)

            if len(indices[0]) > 0:
                for i, idx in enumerate(indices[0]):
                    ret_idx.append(self.gkidx_tree[idx])
                    out_dist_sq.append(distances[0][i] ** 2)
        except Exception as e:
            print(f"Range search error: {e}")

        return ret_idx, out_dist_sq


class LayerDB:
    """层数据库 - 管理一层的多个树桶"""

    MIN_ELEM_SPLIT = 100
    IMBA_DIFF_RATIO = 0.2
    MAX_NUM_BUCKETS = 6
    BUCKET_CHANN = 0  # 用作桶的检索键的第0维
    MAX_BUCKET_VAL = 1000.0

    def __init__(self, tb_cfg):
        self.buckets: List[TreeBucket] = []
        self.bucket_ranges: List[float] = []

        # ✅ 修复：初始时就分配合理的桶范围
        self.bucket_ranges = [0.0] * (self.MAX_NUM_BUCKETS + 1)

        # 根据检索键的典型范围来分配桶
        # 从调试输出看，检索键的第0维通常在0-50之间
        key_range_min = 0.0
        key_range_max = 50.0

        # 等分桶范围
        for i in range(self.MAX_NUM_BUCKETS + 1):
            if i == 0:
                self.bucket_ranges[i] = key_range_min
            elif i == self.MAX_NUM_BUCKETS:
                self.bucket_ranges[i] = key_range_max
            else:
                self.bucket_ranges[i] = key_range_min + (key_range_max - key_range_min) * i / self.MAX_NUM_BUCKETS

        # 创建桶
        for i in range(self.MAX_NUM_BUCKETS):
            self.buckets.append(TreeBucket(tb_cfg, self.bucket_ranges[i], self.bucket_ranges[i + 1]))

    def push_buffer(self, layer_key: np.ndarray, ts: float, scan_key_gidx: IndexOfKey):
        """推送到合适的桶缓冲区"""
        key_val = layer_key[self.BUCKET_CHANN]

        # 找到合适的桶
        for i in range(self.MAX_NUM_BUCKETS):
            if (self.bucket_ranges[i] <= key_val < self.bucket_ranges[i + 1]):
                if np.sum(layer_key) != 0:  # 非零键才添加
                    self.buckets[i].push_buffer(layer_key, ts, scan_key_gidx)
                return

    def rebuild(self, idx_t1: int, curr_ts: float):
        """重建指定的相邻桶对"""
        if idx_t1 >= len(self.buckets) - 1:
            return

        tr1, tr2 = self.buckets[idx_t1], self.buckets[idx_t1 + 1]

        # 检查是否需要弹出缓冲区
        pb1 = tr1.need_pop_buffer(curr_ts)
        pb2 = tr2.need_pop_buffer(curr_ts)

        if not pb1 and not pb2:
            return  # 当我们弹出缓冲区时才重建

        # 获取树大小
        sz1 = tr1.get_tree_size()
        sz2 = tr2.get_tree_size()

        # 计算大小差异比例
        max_size = max(sz1, sz2)
        if max_size == 0:
            diff_ratio = 0
        else:
            diff_ratio = abs(sz1 - sz2) / max_size

        # 简单情况：只需要弹出缓冲区，不需要平衡
        if pb1 and not pb2 and (diff_ratio < self.IMBA_DIFF_RATIO or max_size < self.MIN_ELEM_SPLIT):
            tr1.pop_buffer_max(curr_ts)
            return

        if not pb1 and pb2 and (diff_ratio < self.IMBA_DIFF_RATIO or max_size < self.MIN_ELEM_SPLIT):
            tr2.pop_buffer_max(curr_ts)
            return

        # 需要平衡的情况
        if diff_ratio >= self.IMBA_DIFF_RATIO and max_size >= self.MIN_ELEM_SPLIT:
            print(" (rebalancing...)")
            success = self._rebalance_buckets(tr1, tr2, idx_t1, curr_ts, sz1, sz2)
            if not success:
                # 平衡失败，只弹出缓冲区
                if pb1:
                    tr1.pop_buffer_max(curr_ts)
                if pb2:
                    tr2.pop_buffer_max(curr_ts)
        else:
            # 不需要平衡，只弹出缓冲区
            if pb1:
                tr1.pop_buffer_max(curr_ts)
            if pb2:
                tr2.pop_buffer_max(curr_ts)

    def _rebalance_buckets(self, tr1: TreeBucket, tr2: TreeBucket, idx_t1: int,
                           curr_ts: float, sz1: int, sz2: int) -> bool:
        """重新平衡两个桶"""
        try:
            if sz1 > sz2:
                return self._move_data_from_bucket1_to_bucket2(tr1, tr2, idx_t1, curr_ts, sz1, sz2)
            else:
                return self._move_data_from_bucket2_to_bucket1(tr1, tr2, idx_t1, curr_ts, sz1, sz2)
        except Exception as e:
            print(f"Rebalancing failed: {e}")
            return False

    def _move_data_from_bucket1_to_bucket2(self, tr1: TreeBucket, tr2: TreeBucket,
                                           idx_t1: int, curr_ts: float, sz1: int, sz2: int) -> bool:
        """从桶1移动数据到桶2"""
        if sz1 == 0:
            return False

        # 计算要移动的数据量
        to_move = (sz1 - sz2) // 2
        to_move = max(1, min(to_move, sz1 - 1))  # 至少移动1个，最多保留1个

        # 获取所有数据并按桶通道值排序
        all_data = []
        all_indices = []

        for i, key in enumerate(tr1.data_tree):
            all_data.append((key[self.BUCKET_CHANN], key, tr1.gkidx_tree[i]))

        # 按桶通道值排序
        all_data.sort(key=lambda x: x[0])

        if len(all_data) < to_move:
            return False

        # 找到分割点
        split_idx = len(all_data) - to_move
        split_val = all_data[split_idx][0]

        # 移动数据到桶2
        moved_keys = []
        moved_indices = []
        remaining_keys = []
        remaining_indices = []

        for val, key, idx in all_data:
            if val >= split_val and len(moved_keys) < to_move:
                moved_keys.append(key)
                moved_indices.append(idx)
            else:
                remaining_keys.append(key)
                remaining_indices.append(idx)

        # 更新桶1
        tr1.data_tree = remaining_keys
        tr1.gkidx_tree = remaining_indices

        # 更新桶2
        tr2.data_tree.extend(moved_keys)
        tr2.gkidx_tree.extend(moved_indices)

        # 更新桶范围
        old_range = tr1.buc_end
        tr1.buc_end = tr2.buc_beg = split_val
        self.bucket_ranges[idx_t1 + 1] = split_val

        # 处理缓冲区
        self._redistribute_buffer_data(tr1, tr2, split_val)

        # 重建树
        tr1.pop_buffer_max(curr_ts)
        tr2.pop_buffer_max(curr_ts)

        return True

    def _move_data_from_bucket2_to_bucket1(self, tr1: TreeBucket, tr2: TreeBucket,
                                           idx_t1: int, curr_ts: float, sz1: int, sz2: int) -> bool:
        """从桶2移动数据到桶1"""
        if sz2 == 0:
            return False

        # 计算要移动的数据量
        to_move = (sz2 - sz1) // 2
        to_move = max(1, min(to_move, sz2 - 1))

        # 获取桶2的所有数据并排序
        all_data = []
        for i, key in enumerate(tr2.data_tree):
            all_data.append((key[self.BUCKET_CHANN], key, tr2.gkidx_tree[i]))

        all_data.sort(key=lambda x: x[0])

        if len(all_data) < to_move:
            return False

        # 找到分割点（移动最小的数据到桶1）
        split_val = all_data[to_move - 1][0]

        # 移动数据到桶1
        moved_keys = []
        moved_indices = []
        remaining_keys = []
        remaining_indices = []

        for val, key, idx in all_data:
            if val <= split_val and len(moved_keys) < to_move:
                moved_keys.append(key)
                moved_indices.append(idx)
            else:
                remaining_keys.append(key)
                remaining_indices.append(idx)

        # 更新桶1
        tr1.data_tree.extend(moved_keys)
        tr1.gkidx_tree.extend(moved_indices)

        # 更新桶2
        tr2.data_tree = remaining_keys
        tr2.gkidx_tree = remaining_indices

        # 更新桶范围
        if remaining_keys:
            new_split = min(key[self.BUCKET_CHANN] for key in remaining_keys)
        else:
            new_split = split_val + 0.1

        tr1.buc_end = tr2.buc_beg = new_split
        self.bucket_ranges[idx_t1 + 1] = new_split

        # 处理缓冲区
        self._redistribute_buffer_data(tr1, tr2, new_split)

        # 重建树
        tr1.pop_buffer_max(curr_ts)
        tr2.pop_buffer_max(curr_ts)

        return True

    def _redistribute_buffer_data(self, tr1: TreeBucket, tr2: TreeBucket, split_val: float):
        """重新分配缓冲区数据"""
        # 重新分配tr1的缓冲区
        tr1_new_buffer = []
        tr2_additional_buffer = []

        for key, ts, idx in tr1.buffer:
            if key[self.BUCKET_CHANN] < split_val:
                tr1_new_buffer.append((key, ts, idx))
            else:
                tr2_additional_buffer.append((key, ts, idx))

        # 重新分配tr2的缓冲区
        tr2_new_buffer = []
        tr1_additional_buffer = []

        for key, ts, idx in tr2.buffer:
            if key[self.BUCKET_CHANN] >= split_val:
                tr2_new_buffer.append((key, ts, idx))
            else:
                tr1_additional_buffer.append((key, ts, idx))

        # 更新缓冲区
        tr1.buffer = tr1_new_buffer + tr1_additional_buffer
        tr2.buffer = tr2_new_buffer + tr2_additional_buffer

        # 按时间戳排序
        tr1.buffer.sort(key=lambda x: x[1])
        tr2.buffer.sort(key=lambda x: x[1])

    def layer_knn_search(self, q_key: np.ndarray, k_top: int, max_dist_sq: float) -> List[Tuple[IndexOfKey, float]]:
        """层KNN搜索"""
        # 找到中间桶
        key_val = q_key[self.BUCKET_CHANN]
        mid_bucket = 0

        for i in range(self.MAX_NUM_BUCKETS):
            if (self.bucket_ranges[i] <= key_val < self.bucket_ranges[i + 1]):
                mid_bucket = i
                break

        res_pairs = []
        max_dist_sq_run = max_dist_sq

        # 按距离顺序搜索桶
        for i in range(self.MAX_NUM_BUCKETS):
            bucket_idx = -1

            if i == 0:
                bucket_idx = mid_bucket
            elif mid_bucket - i >= 0:
                # 检查距离约束
                dist_to_bucket = abs(key_val - self.bucket_ranges[mid_bucket - i + 1])
                if dist_to_bucket * dist_to_bucket > max_dist_sq_run:
                    continue
                bucket_idx = mid_bucket - i
            elif mid_bucket + i < self.MAX_NUM_BUCKETS:
                # 检查距离约束
                dist_to_bucket = abs(key_val - self.bucket_ranges[mid_bucket + i])
                if dist_to_bucket * dist_to_bucket > max_dist_sq_run:
                    continue
                bucket_idx = mid_bucket + i

            if bucket_idx >= 0 and bucket_idx < len(self.buckets):
                tmp_gidx, tmp_dists_sq = self.buckets[bucket_idx].knn_search(k_top, q_key, max_dist_sq_run)

                for gidx, dist_sq in zip(tmp_gidx, tmp_dists_sq):
                    if dist_sq < max_dist_sq_run:
                        res_pairs.append((gidx, dist_sq))

                # 排序并限制数量
                res_pairs.sort(key=lambda x: x[1])
                if len(res_pairs) >= k_top:
                    res_pairs = res_pairs[:k_top]
                    if res_pairs:
                        max_dist_sq_run = res_pairs[-1][1]

        return res_pairs

    def layer_range_search(self, q_key: np.ndarray, max_dist_sq: float) -> List[Tuple[IndexOfKey, float]]:
        """层范围搜索"""
        res_pairs = []

        for i in range(self.MAX_NUM_BUCKETS):
            tmp_gidx, tmp_dists_sq = self.buckets[i].range_search(max_dist_sq, q_key)

            for gidx, dist_sq in zip(tmp_gidx, tmp_dists_sq):
                res_pairs.append((gidx, dist_sq))

        return res_pairs


class CandidateAnchorProp:
    """候选锚点提议"""

    def __init__(self):
        self.constell: Dict[ConstellationPair, float] = {}  # 星座匹配：百分比分数
        self.T_delta = np.eye(3)  # 区分不同提议的关键特征
        self.correlation = 0.0
        self.vote_cnt = 0  # 投票给此TF的匹配轮廓数量
        self.area_perc = 0.0  # 所有层级中使用轮廓的面积百分比加权和


class CandidatePoseData:
    """候选姿态数据"""

    def __init__(self, cm_cand: ContourManager):
        self.cm_cand = cm_cand
        self.corr_est: Optional[ConstellCorrelation] = None
        self.anch_props: List[CandidateAnchorProp] = []

    def add_proposal(self, T_prop: np.ndarray, sim_pairs: List[ConstellationPair],
                    sim_area_perc: List[float]):
        """添加锚点提议，合并相似的提议"""
        # 如果对应点不足3个，直接返回不添加这个提议
        if len(sim_pairs) <= 2:
            print(f"警告：sim_pairs不足3个({len(sim_pairs)}个)，跳过此提议")
            return
        assert len(sim_pairs) == len(sim_area_perc)

        # 检查是否与现有提议相似（硬编码阈值：2.0m, 0.3rad）
        for i, prop in enumerate(self.anch_props):
            delta_T = np.linalg.inv(T_prop) @ prop.T_delta
            trans_diff = np.linalg.norm(delta_T[:2, 2])
            rot_diff = abs(np.arctan2(delta_T[1, 0], delta_T[0, 0]))

            if trans_diff < 2.0 and rot_diff < 0.3:
                # 合并到现有提议
                for j, pair in enumerate(sim_pairs):
                    prop.constell[pair] = sim_area_perc[j]  # 覆盖或添加

                old_vote_cnt = prop.vote_cnt
                prop.vote_cnt += len(sim_pairs)

                # 混合变换参数（简单加权平均）
                w1, w2 = old_vote_cnt, len(sim_pairs)
                if w1 + w2 > 0:
                    trans_bl = (prop.T_delta[:2, 2] * w1 + T_prop[:2, 2] * w2) / (w1 + w2)

                    ang1 = np.arctan2(prop.T_delta[1, 0], prop.T_delta[0, 0])
                    ang2 = np.arctan2(T_prop[1, 0], T_prop[0, 0])

                    # 处理角度差
                    diff = ang2 - ang1
                    if diff < 0:
                        diff += 2 * np.pi
                    if diff > np.pi:
                        diff -= 2 * np.pi
                    ang_bl = diff * w2 / (w1 + w2) + ang1

                    # 更新变换矩阵
                    prop.T_delta = np.eye(3)
                    prop.T_delta[:2, :2] = np.array([[np.cos(ang_bl), -np.sin(ang_bl)],
                                                    [np.sin(ang_bl), np.cos(ang_bl)]])
                    prop.T_delta[:2, 2] = trans_bl

                return  # 贪心策略，找到第一个就返回

        # 限制提议数量
        if len(self.anch_props) > 3:
            return

        # 创建新提议
        new_prop = CandidateAnchorProp()
        new_prop.T_delta = T_prop.copy()
        for j, pair in enumerate(sim_pairs):
            new_prop.constell[pair] = sim_area_perc[j]
        new_prop.vote_cnt = len(sim_pairs)

        self.anch_props.append(new_prop)


class CandidateManager:
    """候选管理器 - 处理候选的检查、过滤和优化"""

    def __init__(self, cm_tgt: ContourManager, sim_lb: CandidateScoreEnsemble,
                 sim_ub: CandidateScoreEnsemble):
        """
        初始化候选管理器

        Args:
            cm_tgt: 目标轮廓管理器
            sim_lb: 相似性下界
            sim_ub: 相似性上界
        """
        self.cm_tgt = cm_tgt
        self.sim_var = copy.deepcopy(sim_lb)  # 动态下界（会随着检测过程调整）
        self.sim_ub = sim_ub  # 上界

        # 数据结构
        self.cand_id_pos_pair: Dict[int, int] = {}  # 候选ID到位置的映射
        self.candidates: List[CandidatePoseData] = []

        # 统计记录
        self.flow_valve = 0  # 避免反向工作流
        self.cand_aft_check1 = 0  # 第一轮检查后的候选数
        self.cand_aft_check2 = 0  # 第二轮检查后的候选数
        self.cand_aft_check3 = 0  # 第三轮检查后的候选数

        # 动态阈值控制开关（对应C++的DYNAMIC_THRES宏）
        self.enable_dynamic_thres = False

        # 阈值调整限制参数
        self.max_thres_increase_rate = 2.0  # 最大增长倍数
        self.min_thres_change = 1  # 最小变化量
        self.max_single_adjustment = 3  # 单次最大调整量

        # 验证阈值配置
        assert sim_lb.sim_constell.strict_smaller(sim_ub.sim_constell)
        assert sim_lb.sim_pair.strict_smaller(sim_ub.sim_pair)
        assert sim_lb.sim_post.strict_smaller(sim_ub.sim_post)

    def check_cand_with_hint(self, cm_cand: ContourManager, anchor_pair: ConstellationPair,
                             cont_sim) -> CandidateScoreEnsemble:
        """
        使用提示检查候选

        这是候选管理器的主要功能之一，通过多层级检查验证候选的有效性

        Args:
            cm_cand: 候选轮廓管理器
            anchor_pair: 锚点对提示
            cont_sim: 轮廓相似性配置

        Returns:
            候选分数集合
        """
        assert self.flow_valve == 0, "工作流状态错误"

        cand_id = cm_cand.get_int_id()
        ret_score = CandidateScoreEnsemble()

        # 检查1/4: 锚点相似性
        anchor_sim = self._check_cont_pair_sim(cm_cand, self.cm_tgt, anchor_pair, cont_sim)

        if not anchor_sim:
            return ret_score

        self.cand_aft_check1 += 1

        if self.enable_dynamic_thres:
            print("Before check, curr bar:")
            self._print_threshold_status()

        # 检查2/4: 纯星座检查
        tmp_pairs1 = []
        ret_constell_sim = self._check_constell_sim(
            cm_cand.get_bci(anchor_pair.level, anchor_pair.seq_src),
            self.cm_tgt.get_bci(anchor_pair.level, anchor_pair.seq_tgt),
            self.sim_var.sim_constell, tmp_pairs1)

        ret_score.sim_constell = ret_constell_sim
        if ret_constell_sim.overall() < self.sim_var.sim_constell.overall():
            return ret_score

        self.cand_aft_check2 += 1

        # 检查3/4: 个体相似性检查
        tmp_pairs2 = []
        tmp_area_perc = []
        ret_pairwise_sim, tmp_pairs2, tmp_area_perc = self._check_constell_corresp_sim(
            cm_cand, self.cm_tgt, tmp_pairs1, self.sim_var.sim_pair, cont_sim)

        ret_score.sim_pair = ret_pairwise_sim
        if ret_pairwise_sim.overall() < self.sim_var.sim_pair.overall():
            return ret_score

        self.cand_aft_check3 += 1

        # 获取变换矩阵
        T_pass = self._get_tf_from_constell(cm_cand, self.cm_tgt, tmp_pairs2)

        # 动态阈值更新（仅在启用时）
        if self.enable_dynamic_thres:
            self._update_dynamic_thresholds_safe(ret_pairwise_sim.cnt())

            print("Cand passed. New dynamic bar:")
            self._print_threshold_status()

        # 添加到候选列表或更新现有候选
        self._add_or_update_candidate(cand_id, cm_cand, T_pass, tmp_pairs2, tmp_area_perc)

        return ret_score

    def _update_dynamic_thresholds_safe(self, cnt_curr_valid: int):
        """
        安全的动态阈值更新机制，防止过度调整

        Args:
            cnt_curr_valid: 当前有效的对数量
        """
        if not self.enable_dynamic_thres:
            return

        # 保存当前阈值用于回滚
        old_constell = copy.deepcopy(self.sim_var.sim_constell)
        old_pair = copy.deepcopy(self.sim_var.sim_pair)

        # 计算新的星座相似性阈值（保守调整）
        new_const_lb = ScoreConstellSim()

        # 使用渐进式调整而非直接设置为当前值
        current_ovlp_sum = self.sim_var.sim_constell.i_ovlp_sum
        current_ovlp_max = self.sim_var.sim_constell.i_ovlp_max_one
        current_ang_rng = self.sim_var.sim_constell.i_in_ang_rng

        # 限制单次调整幅度
        max_increase = min(self.max_single_adjustment,
                           max(self.min_thres_change,
                               int(current_ovlp_sum * 0.5)))  # 最多增加50%

        new_const_lb.i_ovlp_sum = min(cnt_curr_valid,
                                      current_ovlp_sum + max_increase)
        new_const_lb.i_ovlp_max_one = min(cnt_curr_valid,
                                          current_ovlp_max + max_increase)
        new_const_lb.i_in_ang_rng = min(cnt_curr_valid,
                                        current_ang_rng + max_increase)

        # 应用阈值约束
        self._align_lb_constell_safe(new_const_lb)
        self._align_ub_constell_safe()

        # 计算新的成对相似性阈值（保守调整）
        new_pair_lb = ScorePairwiseSim()

        current_indiv = self.sim_var.sim_pair.i_indiv_sim
        current_orie = self.sim_var.sim_pair.i_orie_sim

        new_pair_lb.i_indiv_sim = min(cnt_curr_valid,
                                      current_indiv + max_increase)
        new_pair_lb.i_orie_sim = min(cnt_curr_valid,
                                     current_orie + max_increase)

        self._align_lb_pair_safe(new_pair_lb)
        self._align_ub_pair_safe()

        # 验证调整结果，如果超出合理范围则回滚
        if self._is_threshold_reasonable():
            pass  # 阈值合理，保持当前值
        else:
            # 回滚到之前的值
            self.sim_var.sim_constell = old_constell
            self.sim_var.sim_pair = old_pair
            print("Threshold adjustment rolled back due to unreasonable values")

    def _is_threshold_reasonable(self) -> bool:
        """检查阈值是否在合理范围内"""
        # 星座阈值不应超过20（经验值）
        if (self.sim_var.sim_constell.i_ovlp_sum > 20 or
                self.sim_var.sim_constell.i_ovlp_max_one > 20 or
                self.sim_var.sim_constell.i_in_ang_rng > 20):
            return False

        # 成对阈值不应超过20
        if (self.sim_var.sim_pair.i_indiv_sim > 20 or
                self.sim_var.sim_pair.i_orie_sim > 20):
            return False

        # 阈值不应低于初始下界的一半
        if (self.sim_var.sim_constell.i_in_ang_rng < 2 or
                self.sim_var.sim_pair.i_orie_sim < 2):
            return False

        return True

    def _align_lb_constell_safe(self, bar: ScoreConstellSim):
        """安全地对齐星座下界，防止过度增长"""
        # 限制增长幅度
        max_ovlp_sum = min(bar.i_ovlp_sum,
                           int(self.sim_var.sim_constell.i_ovlp_sum * self.max_thres_increase_rate))
        max_ovlp_max = min(bar.i_ovlp_max_one,
                           int(self.sim_var.sim_constell.i_ovlp_max_one * self.max_thres_increase_rate))
        max_ang_rng = min(bar.i_in_ang_rng,
                          int(self.sim_var.sim_constell.i_in_ang_rng * self.max_thres_increase_rate))

        self.sim_var.sim_constell.i_ovlp_sum = max(self.sim_var.sim_constell.i_ovlp_sum,
                                                   max_ovlp_sum)
        self.sim_var.sim_constell.i_ovlp_max_one = max(self.sim_var.sim_constell.i_ovlp_max_one,
                                                       max_ovlp_max)
        self.sim_var.sim_constell.i_in_ang_rng = max(self.sim_var.sim_constell.i_in_ang_rng,
                                                     max_ang_rng)

    def _align_ub_constell_safe(self):
        """安全地对齐星座上界"""
        self.sim_var.sim_constell.i_ovlp_sum = min(self.sim_var.sim_constell.i_ovlp_sum,
                                                   self.sim_ub.sim_constell.i_ovlp_sum)
        self.sim_var.sim_constell.i_ovlp_max_one = min(self.sim_var.sim_constell.i_ovlp_max_one,
                                                       self.sim_ub.sim_constell.i_ovlp_max_one)
        self.sim_var.sim_constell.i_in_ang_rng = min(self.sim_var.sim_constell.i_in_ang_rng,
                                                     self.sim_ub.sim_constell.i_in_ang_rng)

    def _align_lb_pair_safe(self, bar: ScorePairwiseSim):
        """安全地对齐成对下界"""
        max_indiv = min(bar.i_indiv_sim,
                        int(self.sim_var.sim_pair.i_indiv_sim * self.max_thres_increase_rate))
        max_orie = min(bar.i_orie_sim,
                       int(self.sim_var.sim_pair.i_orie_sim * self.max_thres_increase_rate))

        self.sim_var.sim_pair.i_indiv_sim = max(self.sim_var.sim_pair.i_indiv_sim, max_indiv)
        self.sim_var.sim_pair.i_orie_sim = max(self.sim_var.sim_pair.i_orie_sim, max_orie)

    def _align_ub_pair_safe(self):
        """安全地对齐成对上界"""
        self.sim_var.sim_pair.i_indiv_sim = min(self.sim_var.sim_pair.i_indiv_sim,
                                                self.sim_ub.sim_pair.i_indiv_sim)
        self.sim_var.sim_pair.i_orie_sim = min(self.sim_var.sim_pair.i_orie_sim,
                                               self.sim_ub.sim_pair.i_orie_sim)

    def _check_cont_pair_sim(self, src: ContourManager, tgt: ContourManager,
                             cstl: ConstellationPair, cont_sim) -> bool:
        """检查轮廓对相似性"""
        from contour_view import ContourView
        return ContourView.check_sim(
            src.cont_views[cstl.level][cstl.seq_src],
            tgt.cont_views[cstl.level][cstl.seq_tgt],
            cont_sim)

    def _check_constell_sim(self, src: BCI, tgt: BCI, lb: ScoreConstellSim,
                            constell_res: List[ConstellationPair]) -> ScoreConstellSim:
        """检查星座相似性"""
        return BCI.check_constell_sim(src, tgt, lb, constell_res)

    def _check_constell_corresp_sim(self, src: ContourManager, tgt: ContourManager,
                                    cstl_in: List[ConstellationPair],
                                    lb: ScorePairwiseSim, cont_sim) -> Tuple[
        ScorePairwiseSim, List[ConstellationPair], List[float]]:
        """检查星座对应相似性"""
        return ContourManager.check_constell_corresp_sim(src, tgt, cstl_in, lb, cont_sim)

    def _get_tf_from_constell(self, src: ContourManager, tgt: ContourManager,
                              cstl_pairs: List[ConstellationPair]) -> np.ndarray:
        """从星座计算变换矩阵"""
        return ContourManager.get_tf_from_constell(src, tgt, cstl_pairs)

    def _print_threshold_status(self):
        """打印当前阈值状态"""
        print(
            f"Constell: {self.sim_var.sim_constell.i_ovlp_sum}, {self.sim_var.sim_constell.i_ovlp_max_one}, {self.sim_var.sim_constell.i_in_ang_rng}")
        print(f"Pair: {self.sim_var.sim_pair.i_indiv_sim}, {self.sim_var.sim_pair.i_orie_sim}")

    def _add_or_update_candidate(self, cand_id: int, cm_cand: ContourManager,
                                 T_pass: np.ndarray, tmp_pairs2: List[ConstellationPair],
                                 tmp_area_perc: List[float]):
        """添加或更新候选"""
        cand_it = self.cand_id_pos_pair.get(cand_id)
        if cand_it is not None:
            # 候选姿态已存在，添加提议
            self.candidates[cand_it].add_proposal(T_pass, tmp_pairs2, tmp_area_perc)
        else:
            # 添加新候选
            new_cand = CandidatePoseData(cm_cand)
            new_cand.add_proposal(T_pass, tmp_pairs2, tmp_area_perc)
            self.cand_id_pos_pair[cand_id] = len(self.candidates)
            self.candidates.append(new_cand)

    def tidy_up_candidates(self):
        """整理候选 - 预先计算相关性并过滤候选"""
        # 在开始处理前过滤掉空候选
        valid_candidates_with_props = [
            candidate for candidate in self.candidates
            if len(candidate.anch_props) > 0
        ]

        empty_candidates = len(self.candidates) - len(valid_candidates_with_props)
        if empty_candidates > 0:
            print(f"过滤掉 {empty_candidates} 个空候选")

        self.candidates = valid_candidates_with_props

        if not self.candidates:
            print("所有候选都被过滤，没有有效候选")
            return

        assert self.flow_valve < 1
        self.flow_valve += 1

        gmm_config = GMMOptConfig()
        print(f"Tidy up pose {len(self.candidates)} candidates.")

        cnt_to_rm = 0
        valid_candidates = []

        # 分析每个姿态的锚点对
        for candidate in self.candidates:
            assert len(candidate.anch_props) > 0

            # 找到最佳的T_init（基于投票）
            idx_sel = 0
            for i in range(len(candidate.anch_props)):
                # 计算使用区域百分比
                lev_perc = [0] * len(self.cm_tgt.get_config().lv_grads)
                for pr, area_perc in candidate.anch_props[i].constell.items():
                    lev_perc[pr.level] += area_perc

                perc = 0
                for j in range(NUM_BIN_KEY_LAYER):
                    perc += LAYER_AREA_WEIGHTS[j] * lev_perc[DIST_BIN_LAYERS[j]]

                candidate.anch_props[i].area_perc = perc

                if candidate.anch_props[i].vote_cnt > candidate.anch_props[idx_sel].vote_cnt:
                    idx_sel = i

            # 将最佳提议放到第一位
            if idx_sel != 0:
                candidate.anch_props[0], candidate.anch_props[idx_sel] = \
                    candidate.anch_props[idx_sel], candidate.anch_props[0]

            # 总体测试1：面积百分比分数
            print(
                f"Cand id:{candidate.cm_cand.get_int_id()}, @max# {candidate.anch_props[0].vote_cnt} votes, area perc: {candidate.anch_props[0].area_perc}")
            if candidate.anch_props[0].area_perc < self.sim_var.sim_post.area_perc:
                print(
                    f"Low area skipped: {candidate.anch_props[0].area_perc:.6f} < {self.sim_var.sim_post.area_perc:.6f}")
                cnt_to_rm += 1
                continue

            # 总体测试2：距离审查
            est_sens_tf = ConstellCorrelation.get_est_sens_tf(candidate.anch_props[0].T_delta,
                                                              self.cm_tgt.get_config())
            neg_est_trans_norm2d = -np.linalg.norm(est_sens_tf[:2, 2])

            # ===== 添加详细调试输出 =====
            print(f"[DEBUG] Distance calculation for candidate {candidate.cm_cand.get_int_id()}:")
            print(f"  T_delta:\n{candidate.anch_props[0].T_delta}")
            print(f"  est_sens_tf:\n{est_sens_tf}")
            print(f"  estimated 2D distance: {np.linalg.norm(est_sens_tf[:2, 2]):.2f} meters")
            print(f"  neg_est_trans_norm2d: {neg_est_trans_norm2d:.2f}")
            print(f"  threshold: {self.sim_var.sim_post.neg_est_dist:.2f}")
            # ===== 调试输出结束 =====

            # 临时注释掉距离检查
            if neg_est_trans_norm2d < self.sim_var.sim_post.neg_est_dist:
                print(f"Low dist skipped: {neg_est_trans_norm2d:.6f} < {self.sim_var.sim_post.neg_est_dist:.6f}")
                cnt_to_rm += 1
                continue

            # 总体测试3：相关性分数
            corr_est = ConstellCorrelation(gmm_config)
            corr_score_init = corr_est.init_problem(candidate.cm_cand, self.cm_tgt,
                                                    candidate.anch_props[0].T_delta)

            print(f"       :{candidate.cm_cand.get_int_id()}, init corr: {corr_score_init:.6f}")

            if corr_score_init < self.sim_var.sim_post.correlation:
                print(f"Low corr skipped: {corr_score_init:.6f} < {self.sim_var.sim_post.correlation:.6f}")
                cnt_to_rm += 1
                continue

            # 通过测试，谨慎更新阈值变量
            if self.enable_dynamic_thres:
                new_post_lb = ScorePostProc()
                new_post_lb.correlation = corr_score_init
                new_post_lb.area_perc = candidate.anch_props[0].area_perc
                new_post_lb.neg_est_dist = neg_est_trans_norm2d
                self._align_lb_post_safe(new_post_lb)
                self._align_ub_post_safe()

            candidate.anch_props[0].correlation = corr_score_init
            candidate.corr_est = corr_est
            valid_candidates.append(candidate)

        # 更新候选列表
        self.candidates = valid_candidates
        print(f"Tidy up pose remaining: {len(self.candidates)}")

    def _align_lb_post_safe(self, bar: ScorePostProc):
        """安全地对齐后处理下界"""
        # 限制相关性阈值的增长
        max_corr_increase = 0.1  # 每次最多增加0.1
        new_correlation = min(bar.correlation,
                              self.sim_var.sim_post.correlation + max_corr_increase)

        self.sim_var.sim_post.correlation = max(self.sim_var.sim_post.correlation,
                                                new_correlation)
        self.sim_var.sim_post.area_perc = max(self.sim_var.sim_post.area_perc,
                                              bar.area_perc)
        self.sim_var.sim_post.neg_est_dist = max(self.sim_var.sim_post.neg_est_dist,
                                                 bar.neg_est_dist)

    def _align_ub_post_safe(self):
        """安全地对齐后处理上界"""
        self.sim_var.sim_post.correlation = min(self.sim_var.sim_post.correlation,
                                                self.sim_ub.sim_post.correlation)
        self.sim_var.sim_post.area_perc = min(self.sim_var.sim_post.area_perc,
                                              self.sim_ub.sim_post.area_perc)
        self.sim_var.sim_post.neg_est_dist = min(self.sim_var.sim_post.neg_est_dist,
                                                 self.sim_ub.sim_post.neg_est_dist)

    def fine_optimize(self, max_fine_opt: int) -> Tuple[List[ContourManager], List[float], List[np.ndarray]]:
        """精细优化 - 选择有希望的姿态候选并优化精确姿态估计"""
        assert self.flow_valve < 2
        self.flow_valve += 1

        res_cand = []
        res_corr = []
        res_T = []

        if not self.candidates:
            return res_cand, res_corr, res_T

        # 按相关性排序
        self.candidates.sort(key=lambda d: d.anch_props[0].correlation, reverse=True)

        pre_sel_size = min(max_fine_opt, len(self.candidates))
        for i in range(pre_sel_size):
            correlation, T_best = self.candidates[i].corr_est.calc_correlation()
            self.candidates[i].anch_props[0].correlation = correlation
            self.candidates[i].anch_props[0].T_delta = T_best

        # 再次排序
        self.candidates[:pre_sel_size] = sorted(
            self.candidates[:pre_sel_size],
            key=lambda d: d.anch_props[0].correlation,
            reverse=True
        )

        ret_size = 1  # 返回前1个
        for i in range(ret_size):
            res_cand.append(self.candidates[i].cm_cand)
            res_corr.append(self.candidates[i].anch_props[0].correlation)
            res_T.append(self.candidates[i].anch_props[0].T_delta)
            print(f"Fine optimization: {ret_size} candidates selected")

        return res_cand, res_corr, res_T

    def set_dynamic_threshold_enabled(self, enabled: bool):
        """设置动态阈值是否启用"""
        self.enable_dynamic_thres = enabled
        if not enabled:
            print("Dynamic threshold adjustment disabled")
        else:
            print("Dynamic threshold adjustment enabled")


class ContourDB:
    """轮廓数据库 - 管理整个轮廓数据库以进行位置重识别"""

    def __init__(self, config: ContourDBConfig):
        """
        初始化轮廓数据库

        Args:
            config: 数据库配置
        """
        self.cfg = config
        self.layer_db: List[LayerDB] = []
        self.all_bevs: List[ContourManager] = []

        # 为每个查询层级创建层数据库
        for level in config.q_levels:
            self.layer_db.append(LayerDB(config.tb_cfg))

        assert len(config.q_levels) > 0

    def query_ranged_knn(self, q_ptr: ContourManager,
                        thres_lb: CandidateScoreEnsemble,
                        thres_ub: CandidateScoreEnsemble) -> Tuple[List[ContourManager], List[float], List[np.ndarray]]:
        """
        范围KNN查询

        Args:
            q_ptr: 查询轮廓管理器
            thres_lb: 下界阈值
            thres_ub: 上界阈值

        Returns:
            (候选轮廓管理器列表, 相关性分数列表, 变换矩阵列表)
        """
        cand_ptrs = []
        cand_corr = []
        cand_tf = []

        t1 = t2 = t3 = t4 = t5 = 0.0
        start_time = time.time()

        cand_mng = CandidateManager(q_ptr, thres_lb, thres_ub)

        # 为每个层级进行搜索
        for ll in range(len(self.cfg.q_levels)):
            q_bcis = q_ptr.get_lev_bci(self.cfg.q_levels[ll])
            q_keys = q_ptr.get_lev_retrieval_key(self.cfg.q_levels[ll])

            for seq in range(len(q_keys)):
                if np.sum(q_keys[seq]) != 0:
                    print(f"[QUERY] {q_ptr.get_str_id()} L{self.cfg.q_levels[ll]}S{seq}: "
                          f"query_key[0]={q_keys[seq][0]:.6f}")

            assert len(q_bcis) == len(q_keys)

            for seq in range(len(q_bcis)):
                if np.sum(q_keys[seq]) != 0:
                    # 1. 查询
                    start_time = time.time()
                    tmp_res = []

                    # 计算最大查询距离
                    key_bounds = np.zeros((3, 2))
                    key_bounds[0, 0] = q_keys[seq][0] * 0.8
                    key_bounds[0, 1] = q_keys[seq][0] / 0.8
                    key_bounds[1, 0] = q_keys[seq][1] * 0.8
                    key_bounds[1, 1] = q_keys[seq][1] / 0.8
                    key_bounds[2, 0] = q_keys[seq][2] * 0.8 * 0.75
                    key_bounds[2, 1] = q_keys[seq][2] / (0.8 * 0.75)

                    dist_ub = max(
                        max((q_keys[seq][0] - key_bounds[0, 0]) ** 2,
                            (q_keys[seq][0] - key_bounds[0, 1]) ** 2),
                        max((q_keys[seq][1] - key_bounds[1, 0]) ** 2,
                            (q_keys[seq][1] - key_bounds[1, 1]) ** 2),
                        max((q_keys[seq][2] - key_bounds[2, 0]) ** 2,
                            (q_keys[seq][2] - key_bounds[2, 1]) ** 2)
                    )

                    tmp_res = self.layer_db[ll].layer_knn_search(q_keys[seq], self.cfg.nnk, dist_ub)
                    t1 += time.time() - start_time

                    print(f"Dist ub: {dist_ub:.6f}")
                    print(f"L:{self.cfg.q_levels[ll]} S:{seq}. Found in range: {len(tmp_res)}")

                    # 2. 检查
                    start_time = time.time()
                    for sear_res in tmp_res:
                        cnt_chk_pass = cand_mng.check_cand_with_hint(
                            self.all_bevs[sear_res[0].gidx],
                            ConstellationPair(self.cfg.q_levels[ll], sear_res[0].seq, seq),
                            self.cfg.cont_sim_cfg)
                    t2 += time.time() - start_time

        # 找到最佳候选并进行精细调整
        start_time = time.time()
        cand_mng.tidy_up_candidates()
        res_cand_ptr, res_corr, res_T = cand_mng.fine_optimize(self.cfg.max_fine_opt)
        t5 += time.time() - start_time

        num_best_cands = len(res_cand_ptr)
        if num_best_cands:
            print(f"After check 1: {cand_mng.cand_aft_check1}")
            print(f"After check 2: {cand_mng.cand_aft_check2}")
            print(f"After check 3: {cand_mng.cand_aft_check3}")
        else:
            print("No candidates are valid after checks.")

        for i in range(num_best_cands):
            cand_ptrs.append(res_cand_ptr[i])
            cand_corr.append(res_corr[i])
            cand_tf.append(res_T[i])

        if num_best_cands > 0:
            print(f"After check 1: {cand_mng.cand_aft_check1}")
            print(f"After check 2: {cand_mng.cand_aft_check2}")
            print(f"After check 3: {cand_mng.cand_aft_check3}")

        return cand_ptrs, cand_corr, cand_tf

    def add_scan(self, added: ContourManager, curr_timestamp: float):
        """添加扫描到数据库"""
        print(f"[DB_ADD] Adding scan: {added.get_str_id()}")

        for ll in range(len(self.cfg.q_levels)):
            seq = 0
            keys = added.get_lev_retrieval_key(self.cfg.q_levels[ll])
            for permu_key in keys:
                if np.sum(permu_key) != 0:
                    # ✅ 记录添加到数据库的键
                    print(f"[DB_ADD] {added.get_str_id()} L{self.cfg.q_levels[ll]}S{seq}: "
                          f"key[0]={permu_key[0]:.6f}")

                    self.layer_db[ll].push_buffer(permu_key, curr_timestamp,
                                                  IndexOfKey(len(self.all_bevs), self.cfg.q_levels[ll], seq))
                seq += 1

        self.all_bevs.append(added)

    def push_and_balance(self, seed: int, curr_timestamp: float):
        """
        推送数据并维持平衡

        Args:
            seed: 种子值
            curr_timestamp: 当前时间戳
        """
        idx_t1 = abs(seed) % (2 * (LayerDB.MAX_NUM_BUCKETS - 2))
        if idx_t1 > (LayerDB.MAX_NUM_BUCKETS - 2):
            idx_t1 = 2 * (LayerDB.MAX_NUM_BUCKETS - 2) - idx_t1

        print(f"Balancing bucket {idx_t1} and {idx_t1 + 1}")

        print("Tree size of each bucket:")
        for ll in range(len(self.cfg.q_levels)):
            print(f"q_levels_[{ll}]: ", end="")
            self.layer_db[ll].rebuild(idx_t1, curr_timestamp)
            for i in range(LayerDB.MAX_NUM_BUCKETS):
                print(f"{self.layer_db[ll].buckets[i].get_tree_size():5d}", end="")
            print()