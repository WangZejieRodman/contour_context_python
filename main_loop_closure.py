"""
Contour Context Loop Closure Detection - Main Loop Closure Detection Pipeline
主回环检测流程 - Python版本 (修改版，支持PyCharm直接运行)
"""

import numpy as np
import os
import time
import yaml
from typing import List, Tuple, Dict, Optional, Any

from contour_types import (
    ContourManagerConfig, ContourDBConfig, CandidateScoreEnsemble,
    PredictionOutcome, load_config_from_yaml
)
from contour_manager import ContourManager
from contour_database import ContourDB
from evaluator import ContLCDEvaluator
from correlation import ConstellCorrelation


class LoopClosureDetector:
    """回环检测器主类"""

    def __init__(self, config_path: str):
        """
        初始化回环检测器

        Args:
            config_path: 配置文件路径
        """
        self.config = load_config_from_yaml(config_path)

        # 初始化组件
        self.contour_db: Optional[ContourDB] = None
        self.evaluator: Optional[ContLCDEvaluator] = None
        self.cm_config = ContourManagerConfig()
        self.db_config = ContourDBConfig()
        self.thres_lb = CandidateScoreEnsemble()
        self.thres_ub = CandidateScoreEnsemble()

        # 统计变量
        self.cnt_tp = 0
        self.cnt_fn = 0
        self.cnt_fp = 0
        self.ts_beg = -1

        # 加载配置
        self._load_config()

        print("Loop Closure Detector initialized successfully!")

    def _load_config(self):
        """从配置文件加载参数"""
        config = self.config

        # 基本文件路径
        fpath_sens_gt_pose = config['fpath_sens_gt_pose']
        fpath_lidar_bins = config['fpath_lidar_bins']
        fpath_outcome_sav = config['fpath_outcome_sav']
        corr_thres = config['correlation_thres']

        # 初始化评估器
        self.evaluator = ContLCDEvaluator(fpath_sens_gt_pose, fpath_lidar_bins, corr_thres)

        # 数据库配置
        db_cfg = config['ContourDBConfig']
        self.db_config.nnk = db_cfg['nnk_']
        self.db_config.max_fine_opt = db_cfg['max_fine_opt_']
        self.db_config.q_levels = db_cfg['q_levels_']

        # 树桶配置
        tb_cfg = db_cfg['TreeBucketConfig']
        self.db_config.tb_cfg.max_elapse = tb_cfg['max_elapse_']
        self.db_config.tb_cfg.min_elapse = tb_cfg['min_elapse_']

        # 轮廓相似性配置
        cs_cfg = db_cfg['ContourSimThresConfig']
        self.db_config.cont_sim_cfg.ta_cell_cnt = cs_cfg['ta_cell_cnt']
        self.db_config.cont_sim_cfg.tp_cell_cnt = cs_cfg['tp_cell_cnt']
        self.db_config.cont_sim_cfg.tp_eigval = cs_cfg['tp_eigval']
        self.db_config.cont_sim_cfg.ta_h_bar = cs_cfg['ta_h_bar']
        self.db_config.cont_sim_cfg.ta_rcom = cs_cfg['ta_rcom']
        self.db_config.cont_sim_cfg.tp_rcom = cs_cfg['tp_rcom']
        # ===== 添加调试输出 =====
        print(f"[DEBUG] Contour similarity thresholds:")
        print(f"  ta_cell_cnt (absolute): {self.db_config.cont_sim_cfg.ta_cell_cnt}")
        print(f"  tp_cell_cnt (percentage): {self.db_config.cont_sim_cfg.tp_cell_cnt}")
        print(f"  ta_h_bar: {self.db_config.cont_sim_cfg.ta_h_bar}")
        # ===== 调试输出结束 =====

        # 初始化数据库
        self.contour_db = ContourDB(self.db_config)

        # 阈值配置
        thres_cfg = config['thres_lb_']
        self.thres_lb.sim_constell.i_ovlp_sum = thres_cfg['i_ovlp_sum']
        self.thres_lb.sim_constell.i_ovlp_max_one = thres_cfg['i_ovlp_max_one']
        self.thres_lb.sim_constell.i_in_ang_rng = thres_cfg['i_in_ang_rng']
        self.thres_lb.sim_pair.i_indiv_sim = thres_cfg['i_indiv_sim']
        self.thres_lb.sim_pair.i_orie_sim = thres_cfg['i_orie_sim']
        self.thres_lb.sim_post.correlation = thres_cfg['correlation']
        self.thres_lb.sim_post.area_perc = thres_cfg['area_perc']
        self.thres_lb.sim_post.neg_est_dist = thres_cfg['neg_est_dist']

        thres_cfg = config['thres_ub_']
        self.thres_ub.sim_constell.i_ovlp_sum = thres_cfg['i_ovlp_sum']
        self.thres_ub.sim_constell.i_ovlp_max_one = thres_cfg['i_ovlp_max_one']
        self.thres_ub.sim_constell.i_in_ang_rng = thres_cfg['i_in_ang_rng']
        self.thres_ub.sim_pair.i_indiv_sim = thres_cfg['i_indiv_sim']
        self.thres_ub.sim_pair.i_orie_sim = thres_cfg['i_orie_sim']
        self.thres_ub.sim_post.correlation = thres_cfg['correlation']
        self.thres_ub.sim_post.area_perc = thres_cfg['area_perc']
        self.thres_ub.sim_post.neg_est_dist = thres_cfg['neg_est_dist']

        # 轮廓管理器配置
        cm_cfg = config['ContourManagerConfig']
        self.cm_config.lv_grads = cm_cfg['lv_grads_']
        self.cm_config.reso_row = cm_cfg['reso_row_']
        self.cm_config.reso_col = cm_cfg['reso_col_']
        self.cm_config.n_row = cm_cfg['n_row_']
        self.cm_config.n_col = cm_cfg['n_col_']
        self.cm_config.lidar_height = cm_cfg['lidar_height_']
        self.cm_config.blind_sq = cm_cfg['blind_sq_']
        self.cm_config.min_cont_key_cnt = cm_cfg['min_cont_key_cnt_']
        self.cm_config.min_cont_cell_cnt = cm_cfg['min_cont_cell_cnt_']
        self.cm_config.piv_firsts = cm_cfg['piv_firsts_']
        self.cm_config.dist_firsts = cm_cfg['dist_firsts_']
        self.cm_config.roi_radius = cm_cfg['roi_radius_']

        self.outcome_save_path = fpath_outcome_sav

        # ===== 添加调试输出 =====
        print(f"[DEBUG] Loaded thresholds:")
        print(
            f"  Lower bounds - Constell: {self.thres_lb.sim_constell.i_in_ang_rng}, Pair: {self.thres_lb.sim_pair.i_orie_sim}")
        print(
            f"  Upper bounds - Constell: {self.thres_ub.sim_constell.i_in_ang_rng}, Pair: {self.thres_ub.sim_pair.i_orie_sim}")
        print(f"  Correlation threshold: {corr_thres}")
        # ===== 调试输出结束 =====

    def process_single_scan(self, outer_cnt: int) -> int:
        """
        处理单个扫描

        Args:
            outer_cnt: 外部计数器

        Returns:
            0: 正常, 1: 加载失败, -1: 完成
        """
        # 1. 加载新扫描
        if not self.evaluator.load_new_scan():
            print("Load new scan failed.")
            return 1

        start_time = time.time()

        # 2. 初始化当前扫描
        ptr_cm_tgt = self.evaluator.get_curr_contour_manager(self.cm_config)
        laser_info_tgt = self.evaluator.get_curr_scan_info()

        print(f"\n===\nLoaded: assigned seq: {laser_info_tgt.seq}, bin path: {laser_info_tgt.fpath}")

        # 2.1 准备和显示信息：真值姿态
        ts_curr = laser_info_tgt.ts
        if self.ts_beg < 0:
            self.ts_beg = ts_curr

        T_gt_curr = laser_info_tgt.sens_pose

        # 显示轮廓统计
        total_contours = sum(len(ptr_cm_tgt.get_lev_contours(i)) for i in range(len(self.cm_config.lv_grads)))
        print(f"Generated {total_contours} total contours across all levels")
        for level in range(min(3, len(self.cm_config.lv_grads))):  # 只显示前3层
            contours = ptr_cm_tgt.get_lev_contours(level)
            keys = ptr_cm_tgt.get_lev_retrieval_key(level)
            valid_keys = sum(1 for key in keys if np.sum(key) != 0)
            print(f"  Level {level}: {len(contours)} contours, {valid_keys}/{len(keys)} valid keys")

        print(f"Time makebev: {time.time() - start_time:.5f}")

        # 2.2 清理图像以节省内存
        ptr_cm_tgt.clear_image()

        # 3. 查询
        start_time = time.time()
        ptr_cands, cand_corr, bev_tfs = self.contour_db.query_ranged_knn(
            ptr_cm_tgt, self.thres_lb, self.thres_ub)
        query_time = time.time() - start_time

        print(f"{len(ptr_cands)} Candidates in {query_time:.5f}s:")

        # 3.1 处理查询结果
        assert len(ptr_cands) < 2, "当前实现最多支持1个候选"

        if not ptr_cands:
            pred_res = self.evaluator.add_prediction(ptr_cm_tgt, 0.0)
        else:
            pred_res = self.evaluator.add_prediction(
                ptr_cm_tgt, cand_corr[0], ptr_cands[0], bev_tfs[0])
            print(f"  Best candidate: ID {ptr_cands[0].get_int_id()}, correlation: {cand_corr[0]:.6f}")

        # 3.2 统计结果
        if pred_res.tfpn == PredictionOutcome.TP:
            print("Prediction outcome: TP ✓")
            self.cnt_tp += 1
        elif pred_res.tfpn == PredictionOutcome.FP:
            print("Prediction outcome: FP ✗")
            self.cnt_fp += 1
        elif pred_res.tfpn == PredictionOutcome.TN:
            print("Prediction outcome: TN")
        elif pred_res.tfpn == PredictionOutcome.FN:
            print("Prediction outcome: FN ✗")
            self.cnt_fn += 1

        # 3.3 显示累计统计
        print(
            f"TP Error mean: t:{self.evaluator.get_tp_mean_trans():.4f} m, r:{self.evaluator.get_tp_mean_rot():.4f} rad")
        print(
            f"TP Error rmse: t:{self.evaluator.get_tp_rmse_trans():.4f} m, r:{self.evaluator.get_tp_rmse_rot():.4f} rad")
        print(f"Accumulated tp poses: {self.cnt_tp}")
        print(f"Accumulated fn poses: {self.cnt_fn}")
        print(f"Accumulated fp poses: {self.cnt_fp}")

        # 4. 更新数据库
        start_time = time.time()

        # 4.1 添加扫描
        self.contour_db.add_scan(ptr_cm_tgt, laser_info_tgt.ts)

        # 4.2 平衡树
        self.contour_db.push_and_balance(laser_info_tgt.seq, laser_info_tgt.ts)
        rebalance_time = time.time() - start_time

        print(f"Rebalance tree cost: {rebalance_time:.5f}")

        # 显示数据库大小
        total_db_size = len(self.contour_db.all_bevs)
        print(f"Database size: {total_db_size} scans")

        return 0

    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        运行完整的回环检测流程

        Returns:
            包含统计结果的字典
        """
        print("Starting loop closure detection pipeline...")

        cnt = 0
        start_time = time.time()

        while True:
            ret_code = self.process_single_scan(cnt)

            if ret_code == 1:
                print("No more scans to process. Pipeline completed!")
                break  # 改为正常退出而不是重试
            elif ret_code == -1:
                print("Pipeline completed!")
                break

            cnt += 1

            # 每10个扫描显示一次进度
            if cnt % 10 == 0:
                elapsed = time.time() - start_time
                print(f"\nProcessed {cnt} scans in {elapsed:.2f}s (avg: {elapsed / cnt:.3f}s per scan)")

        # 保存预测结果
        self.evaluator.save_prediction_results(self.outcome_save_path)

        # 计算最终统计
        total_time = time.time() - start_time
        results = {
            'total_scans': cnt,
            'total_time': total_time,
            'avg_time_per_scan': total_time / cnt if cnt > 0 else 0,
            'tp_count': self.cnt_tp,
            'fp_count': self.cnt_fp,
            'fn_count': self.cnt_fn,
            'tp_mean_trans_error': self.evaluator.get_tp_mean_trans(),
            'tp_mean_rot_error': self.evaluator.get_tp_mean_rot(),
            'tp_rmse_trans_error': self.evaluator.get_tp_rmse_trans(),
            'tp_rmse_rot_error': self.evaluator.get_tp_rmse_rot(),
        }

        # 计算精度和召回率
        if self.cnt_tp + self.cnt_fp > 0:
            results['precision'] = self.cnt_tp / (self.cnt_tp + self.cnt_fp)
        else:
            results['precision'] = 0.0

        if self.cnt_tp + self.cnt_fn > 0:
            results['recall'] = self.cnt_tp / (self.cnt_tp + self.cnt_fn)
        else:
            results['recall'] = 0.0

        if results['precision'] + results['recall'] > 0:
            results['f1_score'] = 2 * results['precision'] * results['recall'] / (
                        results['precision'] + results['recall'])
        else:
            results['f1_score'] = 0.0

        return results

    def print_results(self, results: Dict[str, Any]):
        """打印结果统计"""
        print("\n" + "=" * 60)
        print("LOOP CLOSURE DETECTION RESULTS")
        print("=" * 60)

        print(f"Total scans processed: {results['total_scans']}")
        print(f"Total processing time: {results['total_time']:.2f}s")
        print(f"Average time per scan: {results['avg_time_per_scan']:.3f}s")

        print(f"\nDetection Results:")
        print(f"  True Positives (TP):  {results['tp_count']}")
        print(f"  False Positives (FP): {results['fp_count']}")
        print(f"  False Negatives (FN): {results['fn_count']}")

        print(f"\nPerformance Metrics:")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1_score']:.4f}")

        print(f"\nTP Error Statistics:")
        print(f"  Mean Translation Error: {results['tp_mean_trans_error']:.4f} m")
        print(f"  Mean Rotation Error:    {results['tp_mean_rot_error']:.4f} rad")
        print(f"  RMSE Translation Error: {results['tp_rmse_trans_error']:.4f} m")
        print(f"  RMSE Rotation Error:    {results['tp_rmse_rot_error']:.4f} rad")

        print("=" * 60)


def create_config_file():
    """创建配置文件"""
    config = {
        # 数据文件路径
        'fpath_sens_gt_pose': "/home/wzj/pan1/contour_context_ws/src/contour-context/sample_data/ts-sens_pose-kitti00.txt",
        'fpath_lidar_bins': "/home/wzj/pan1/contour_context_ws/src/contour-context/sample_data/ts-lidar_bins-kitti00.txt",
        'fpath_outcome_sav': "/home/wzj/pan1/contour_context_py/outcome-kitti00.txt",

        # 相关性阈值
        'correlation_thres': 0.64928,

        # 轮廓数据库配置
        'ContourDBConfig': {
            'nnk_': 50,
            'max_fine_opt_': 10,
            'q_levels_': [1, 2, 3],

            # 树桶配置
            'TreeBucketConfig': {
                'max_elapse_': 25.0,
                'min_elapse_': 15.0
            },

            # 轮廓相似性阈值配置
            'ContourSimThresConfig': {
                'ta_cell_cnt': 6.0,
                'tp_cell_cnt': 0.2,
                'tp_eigval': 0.2,
                'ta_h_bar': 0.3,
                'ta_rcom': 0.4,
                'tp_rcom': 0.25
            }
        },

        # 轮廓管理器配置
        'ContourManagerConfig': {
            'lv_grads_': [1.5, 2, 2.5, 3, 3.5, 4],
            'reso_row_': 1.0,
            'reso_col_': 1.0,
            'n_row_': 150,
            'n_col_': 150,
            'lidar_height_': 2.0,
            'blind_sq_': 9.0,
            'min_cont_key_cnt_': 9,
            'min_cont_cell_cnt_': 3,
            'piv_firsts_': 6,
            'dist_firsts_': 10,
            'roi_radius_': 10.0
        },

        # 下界阈值 - 修正距离下界
        'thres_lb_': {
            'i_ovlp_sum': 3,
            'i_ovlp_max_one': 3,
            'i_in_ang_rng': 3,
            'i_indiv_sim': 3,
            'i_orie_sim': 4,
            'correlation': 0.3,        # 恢复原值
            'area_perc': 0.03,         # 恢复原值
            'neg_est_dist': -5.01      # 恢复原值
        },

        # 上界阈值 - 需要更大的余量
        'thres_ub_': {
            'i_ovlp_sum': 6,
            'i_ovlp_max_one': 6,
            'i_in_ang_rng': 6,
            'i_indiv_sim': 6,
            'i_orie_sim': 6,
            'correlation': 0.75,
            'area_perc': 0.15,
            'neg_est_dist': -5.0
        }
    }

    return config


def main():
    """主函数 - 修改为在PyCharm中直接运行"""

    # 硬编码配置，无需命令行参数
    config_dict = create_config_file()

    # 创建临时配置文件
    config_path = "temp_config.yaml"
    output_dir = "./results"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)


    print(f"Output directory: {output_dir}")

    try:
        # 初始化检测器
        detector = LoopClosureDetector(config_path)

        # 运行完整流程
        results = detector.run_full_pipeline()

        # 打印结果
        detector.print_results(results)

        # 保存结果到JSON文件
        import json
        results_path = os.path.join(output_dir, 'loop_closure_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_path}")

        # 清理临时配置文件
        if os.path.exists(config_path):
            os.remove(config_path)

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())