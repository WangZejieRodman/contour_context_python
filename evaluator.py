"""
Contour Context Loop Closure Detection - Evaluator
评估器实现，用于数据加载和性能评估
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import os
import time

from contour_types import (
    ContourManagerConfig, CandidateScoreEnsemble, PredictionOutcome
)
from contour_manager import ContourManager
from correlation import ConstellCorrelation


@dataclass
class SimpleRMSE:
    """简单RMSE计算器"""
    sum_sqs: float = 0.0
    sum_abs: float = 0.0
    cnt_sqs: int = 0

    def add_one_err(self, d: np.ndarray):
        """添加一个误差"""
        self.cnt_sqs += 1
        tmp = np.sum(d * d)
        self.sum_sqs += tmp
        self.sum_abs += np.sqrt(tmp)

    def get_rmse(self) -> float:
        """获取RMSE"""
        return np.sqrt(self.sum_sqs / self.cnt_sqs) if self.cnt_sqs > 0 else -1

    def get_mean(self) -> float:
        """获取平均误差"""
        return self.sum_abs / self.cnt_sqs if self.cnt_sqs > 0 else -1


@dataclass
class PredictionResult:
    """预测结果"""
    id_src: int = -1
    id_tgt: int = -1
    tfpn: PredictionOutcome = PredictionOutcome.TN
    est_err: np.ndarray = None
    correlation: float = 0.0

    def __post_init__(self):
        if self.est_err is None:
            self.est_err = np.zeros(3)


@dataclass
class LaserScanInfo:
    """激光扫描信息"""
    has_gt_positive_lc: bool = False
    sens_pose: np.ndarray = None  # 4x4变换矩阵
    seq: int = 0
    ts: float = 0.0
    fpath: str = ""

    def __post_init__(self):
        if self.sens_pose is None:
            self.sens_pose = np.eye(4)


class ContLCDEvaluator:
    """轮廓回环检测评估器"""

    def __init__(self, fpath_pose: str, fpath_laser: str, sim_thres: float):
        """
        初始化评估器

        Args:
            fpath_pose: 姿态文件路径
            fpath_laser: 激光文件路径
            sim_thres: 相似性阈值
        """
        self.laser_info: List[LaserScanInfo] = []
        self.assigned_seqs: List[int] = []

        # 参数
        self.ts_diff_tol = 10e-3  # 10ms
        self.min_time_excl = 15.0  # 排除15s
        self.sim_thres = sim_thres

        # 记录变量
        self.p_lidar_curr = -1

        # 基准记录器
        self.tp_trans_rmse = SimpleRMSE()
        self.all_trans_rmse = SimpleRMSE()
        self.tp_rot_rmse = SimpleRMSE()
        self.all_rot_rmse = SimpleRMSE()
        self.pred_records: List[PredictionResult] = []

        # 加载数据
        self._load_data(fpath_pose, fpath_laser)

    def _load_data(self, fpath_pose: str, fpath_laser: str):
        """加载数据文件"""
        # 1. 读取真值姿态
        gt_tss = []
        gt_poses = []

        with open(fpath_pose, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 13:
                    # 时间戳和12元素姿态
                    ts = float(parts[0])
                    pose_data = [float(x) for x in parts[1:13]]

                    # 重建4x4变换矩阵
                    pose_matrix = np.eye(4)
                    # 旋转矩阵 (行主序)
                    pose_matrix[:3, :3] = np.array(pose_data[:9]).reshape(3, 3)
                    # 平移向量
                    pose_matrix[:3, 3] = pose_data[9:12]

                    gt_tss.append(ts)
                    gt_poses.append(pose_matrix)

        print(f"Added {len(gt_poses)} stamped gt poses.")

        # 按时间戳排序
        sort_indices = np.argsort(gt_tss)
        gt_tss = [gt_tss[i] for i in sort_indices]
        gt_poses = [gt_poses[i] for i in sort_indices]

        # 2. 读取激光扫描信息
        lidar_ts = []
        assigned_seq = []
        bin_paths = []

        with open(fpath_laser, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    ts = float(parts[0])
                    seq = int(parts[1])
                    bin_path = parts[2]

                    lidar_ts.append(ts)
                    assigned_seq.append(seq)
                    bin_paths.append(bin_path)

        print(f"Added {len(bin_paths)} laser bin paths.")

        # 3. 关联激光扫描和真值姿态
        cnt_valid_scans = 0
        for i in range(len(lidar_ts)):
            gt_idx = self._lookup_nn(lidar_ts[i], gt_tss, self.ts_diff_tol)
            if gt_idx < 0:
                continue

            scan_info = LaserScanInfo()
            scan_info.sens_pose = gt_poses[gt_idx]
            scan_info.fpath = bin_paths[i]
            scan_info.ts = lidar_ts[i]
            scan_info.seq = assigned_seq[i]

            cnt_valid_scans += 1
            self.laser_info.append(scan_info)
            self.assigned_seqs.append(assigned_seq[i])

        print(f"Found {cnt_valid_scans} laser scans with gt poses.")

        # 4. 验证排序
        for i in range(1, len(self.laser_info)):
            assert self.laser_info[i - 1].seq < self.laser_info[i].seq
            assert self.laser_info[i - 1].ts < self.laser_info[i].ts

        print("Ordering check passed")

        # 5. 添加真值回环信息
        cnt_gt_lc_p = 0
        cnt_gt_lc = 0

        for i, scan_fast in enumerate(self.laser_info):
            for j, scan_slow in enumerate(self.laser_info):
                if scan_fast.ts < scan_slow.ts + self.min_time_excl:
                    break

                # 计算距离
                trans_fast = scan_fast.sens_pose[:3, 3]
                trans_slow = scan_slow.sens_pose[:3, 3]
                dist = np.linalg.norm(trans_fast - trans_slow)

                if dist < 5.0:
                    if not scan_fast.has_gt_positive_lc:
                        scan_fast.has_gt_positive_lc = True
                        cnt_gt_lc_p += 1
                    cnt_gt_lc += 1

        print(f"Found {cnt_gt_lc_p} poses with {cnt_gt_lc} gt loops.")

    def _lookup_nn(self, q_val: float, sorted_vec: List[float], tol: float) -> int:
        """查找最近邻"""
        if not sorted_vec:
            return -1

        # 二分查找
        left, right = 0, len(sorted_vec) - 1
        best_idx = -1
        best_diff = float('inf')

        while left <= right:
            mid = (left + right) // 2
            diff = abs(sorted_vec[mid] - q_val)

            if diff < best_diff:
                best_diff = diff
                best_idx = mid

            if sorted_vec[mid] < q_val:
                left = mid + 1
            else:
                right = mid - 1

        return best_idx if best_diff <= tol else -1

    def load_new_scan(self) -> bool:
        """加载新扫描"""
        self.p_lidar_curr += 1

        if self.p_lidar_curr >= len(self.laser_info):
            print(f"\ncurrent addr {self.p_lidar_curr} exceeds boundary")
            return False

        scan_info = self.laser_info[self.p_lidar_curr]
        print(f"\n===\nloaded scan addr {self.p_lidar_curr}, seq: {scan_info.seq}, fpath: {scan_info.fpath}")
        return True

    def get_curr_scan_info(self) -> LaserScanInfo:
        """获取当前扫描信息"""
        assert 0 <= self.p_lidar_curr < len(self.laser_info)
        return self.laser_info[self.p_lidar_curr]

    def get_curr_contour_manager(self, config: ContourManagerConfig) -> ContourManager:
        """获取当前轮廓管理器"""
        scan_info = self.laser_info[self.p_lidar_curr]

        # 创建轮廓管理器
        cmng = ContourManager(config, scan_info.seq)

        # 读取点云数据
        point_cloud = self._read_point_cloud(scan_info.fpath)

        # 生成字符串ID
        str_id = f"assigned_id_{scan_info.seq:08d}"

        # 处理点云
        cmng.make_bev(point_cloud, str_id)
        cmng.make_contours_recursive()

        return cmng

    def _read_point_cloud(self, fpath: str) -> np.ndarray:
        """读取KITTI格式的点云二进制文件"""
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Point cloud file not found: {fpath}")

        # 读取二进制文件
        points = np.fromfile(fpath, dtype=np.float32)
        points = points.reshape(-1, 4)  # KITTI格式是(x, y, z, intensity)

        return points[:, :3]  # 只返回xyz坐标

    def add_prediction(self, q_mng: ContourManager, est_corr: float,
                       cand_mng: Optional[ContourManager] = None,
                       T_est_delta_2d: Optional[np.ndarray] = None) -> PredictionResult:
        """
        添加预测结果

        Args:
            q_mng: 查询轮廓管理器
            est_corr: 估计相关性
            cand_mng: 候选轮廓管理器
            T_est_delta_2d: 估计的2D变换

        Returns:
            预测结果
        """
        if T_est_delta_2d is None:
            T_est_delta_2d = np.eye(3)

        id_tgt = q_mng.get_int_id()
        addr_tgt = self._lookup_nn(id_tgt, self.assigned_seqs, 0)
        assert addr_tgt >= 0

        curr_res = PredictionResult()
        curr_res.id_tgt = id_tgt
        curr_res.correlation = est_corr

        if cand_mng is not None:
            # 预测为正
            id_src = cand_mng.get_int_id()
            addr_src = self._lookup_nn(id_src, self.assigned_seqs, 0)
            assert addr_src >= 0

            curr_res.id_src = id_src

            # 计算误差
            gen_bev_config = q_mng.get_config()
            tf_err = ConstellCorrelation.eval_metric_est(
                T_est_delta_2d,
                self.laser_info[addr_src].sens_pose,
                self.laser_info[addr_tgt].sens_pose,
                gen_bev_config)

            # 计算距离
            est_trans_norm2d = np.linalg.norm(
                ConstellCorrelation.get_est_sens_tf(T_est_delta_2d, gen_bev_config)[:2, 2])
            gt_trans_norm3d = np.linalg.norm(
                self.laser_info[addr_src].sens_pose[:3, 3] -
                self.laser_info[addr_tgt].sens_pose[:3, 3])

            print(f" Dist: Est2d: {est_trans_norm2d:.2f}; GT3d: {gt_trans_norm3d:.2f}")

            # 误差向量
            err_vec = np.array([
                tf_err[0, 2],  # dx
                tf_err[1, 2],  # dy
                np.arctan2(tf_err[1, 0], tf_err[0, 0])  # dtheta
            ])

            print(f" Error: dx={err_vec[0]:.6f}, dy={err_vec[1]:.6f}, dtheta={err_vec[2]:.6f}")

            curr_res.est_err = err_vec

            # 确定TFPN
            if est_corr >= self.sim_thres:
                if (self.laser_info[addr_tgt].has_gt_positive_lc and gt_trans_norm3d < 5.0):
                    curr_res.tfpn = PredictionOutcome.TP
                    self.tp_trans_rmse.add_one_err(err_vec[:2])
                    self.tp_rot_rmse.add_one_err(err_vec[2:3])
                else:
                    curr_res.tfpn = PredictionOutcome.FP
            else:
                if self.laser_info[addr_tgt].has_gt_positive_lc:
                    curr_res.tfpn = PredictionOutcome.FN
                else:
                    curr_res.tfpn = PredictionOutcome.TN

            self.all_trans_rmse.add_one_err(err_vec[:2])
            self.all_rot_rmse.add_one_err(err_vec[2:3])

        else:
            # 预测为负
            if self.laser_info[addr_tgt].has_gt_positive_lc:
                curr_res.tfpn = PredictionOutcome.FN
            else:
                curr_res.tfpn = PredictionOutcome.TN

        self.pred_records.append(curr_res)
        return curr_res

    def save_prediction_results(self, sav_path: str):
        """保存预测结果"""
        with open(sav_path, 'w') as f:
            for rec in self.pred_records:
                addr_tgt = self._lookup_nn(rec.id_tgt, self.assigned_seqs, 0)
                assert addr_tgt >= 0

                # 写入TFPN
                f.write(f"{rec.tfpn.value}\t")

                # 写入ID对
                if rec.id_src < 0:
                    f.write(f"{rec.id_tgt}-x\t")
                    str_rep_src = "x"
                else:
                    addr_src = self._lookup_nn(rec.id_src, self.assigned_seqs, 0)
                    assert addr_src >= 0
                    f.write(f"{rec.id_tgt}-{rec.id_src}\t")
                    str_rep_src = self.laser_info[addr_src].fpath

                # 写入相关性和误差
                f.write(f"{rec.correlation:.6f}\t{rec.est_err[0]:.6f}\t{rec.est_err[1]:.6f}\t{rec.est_err[2]:.6f}\t")

                # 写入路径（缩短版本）
                str_rep_tgt = self.laser_info[addr_tgt].fpath
                str_max_len = 32

                beg_tgt = max(0, len(str_rep_tgt) - str_max_len)
                beg_src = max(0, len(str_rep_src) - str_max_len)

                f.write(f"{str_rep_tgt[beg_tgt:]}\t{str_rep_src[beg_src:]}\n")

        print("结果文件中:")
        print(f"TP is {PredictionOutcome.TP.value}")
        print(f"FP is {PredictionOutcome.FP.value}")
        print(f"TN is {PredictionOutcome.TN.value}")
        print(f"FN is {PredictionOutcome.FN.value}")
        print("Outcome saved successfully.")

    # Getter方法
    def get_tp_mean_trans(self) -> float:
        """获取TP平均平移误差"""
        return self.tp_trans_rmse.get_mean()

    def get_tp_mean_rot(self) -> float:
        """获取TP平均旋转误差"""
        return self.tp_rot_rmse.get_mean()

    def get_tp_rmse_trans(self) -> float:
        """获取TP平移RMSE"""
        return self.tp_trans_rmse.get_rmse()

    def get_tp_rmse_rot(self) -> float:
        """获取TP旋转RMSE"""
        return self.tp_rot_rmse.get_rmse()

    @staticmethod
    def load_check_thres(fpath: str) -> Tuple[CandidateScoreEnsemble, CandidateScoreEnsemble]:
        """从文件加载检查阈值"""
        from contour_types import ScoreConstellSim, ScorePairwiseSim, ScorePostProc

        thres_lb = CandidateScoreEnsemble()
        thres_ub = CandidateScoreEnsemble()

        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) < 3:
                    continue

                pname = parts[0]
                lb_val = float(parts[1])
                ub_val = float(parts[2])

                if pname == "i_ovlp_sum":
                    thres_lb.sim_constell.i_ovlp_sum = int(lb_val)
                    thres_ub.sim_constell.i_ovlp_sum = int(ub_val)
                elif pname == "i_ovlp_max_one":
                    thres_lb.sim_constell.i_ovlp_max_one = int(lb_val)
                    thres_ub.sim_constell.i_ovlp_max_one = int(ub_val)
                elif pname == "i_in_ang_rng":
                    thres_lb.sim_constell.i_in_ang_rng = int(lb_val)
                    thres_ub.sim_constell.i_in_ang_rng = int(ub_val)
                elif pname == "i_indiv_sim":
                    thres_lb.sim_pair.i_indiv_sim = int(lb_val)
                    thres_ub.sim_pair.i_indiv_sim = int(ub_val)
                elif pname == "i_orie_sim":
                    thres_lb.sim_pair.i_orie_sim = int(lb_val)
                    thres_ub.sim_pair.i_orie_sim = int(ub_val)
                elif pname == "correlation":
                    thres_lb.sim_post.correlation = lb_val
                    thres_ub.sim_post.correlation = ub_val
                elif pname == "area_perc":
                    thres_lb.sim_post.area_perc = lb_val
                    thres_ub.sim_post.area_perc = ub_val
                elif pname == "neg_est_dist":
                    thres_lb.sim_post.neg_est_dist = lb_val
                    thres_ub.sim_post.neg_est_dist = ub_val

        return thres_lb, thres_ub