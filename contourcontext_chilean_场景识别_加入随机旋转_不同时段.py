#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ContourContext Chilean场景识别评估
基于contour_context改编，用于Chilean数据集的跨时间段场景识别
评估任务：历史地图(100-101) vs 当前观测(180-181)
"""

import numpy as np
import os
import sys
import pickle
import time
from typing import List, Dict, Tuple, Optional
import logging

# 导入contour context相关模块
from contour_types import (
    ContourManagerConfig, ContourDBConfig, CandidateScoreEnsemble,
    PredictionOutcome, load_config_from_yaml
)
from contour_manager import ContourManager
from contour_database import ContourDB


class ChileanContourContextEvaluator:
    """Chilean数据集上的ContourContext场景识别评估器"""

    def __init__(self, dataset_folder: str,
                 database_file: str = 'chilean_NoRot_NoScale_5cm_evaluation_database_100_101.pickle',
                 query_file: str = 'chilean_NoRot_NoScale_5cm_evaluation_query_180_181.pickle',
                 log_file: str = 'contourcontext_chilean_不同时段_log.txt'):
        self.dataset_folder = dataset_folder
        self.database_file = database_file
        self.query_file = query_file

        # 设置随机种子以确保可重复的实验结果
        np.random.seed(42)

        # 设置日志记录
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # 加载数据集
        self.database_sets = self.load_sets_dict(self.database_file)
        self.query_sets = self.load_sets_dict(self.query_file)

        # ContourContext配置
        self.cm_config = self.create_contour_manager_config_for_chilean()
        self.db_config = self.create_contour_db_config_for_chilean()
        self.thres_lb, self.thres_ub = self.create_detection_thresholds()

        # 为每个数据库时间段创建ContourDB
        self.contour_dbs = {}
        for i in range(len(self.database_sets)):
            self.contour_dbs[i] = ContourDB(self.db_config)

        self.logger.info(f"初始化完成：{len(self.database_sets)}个数据库时间段，{len(self.query_sets)}个查询时间段")
        self.logger.info(f"评估任务：历史地图(100-101) vs 当前观测(180-181)")
        self.processed_cloud_count = 0

    def create_contour_manager_config_for_chilean(self) -> ContourManagerConfig:
        """为Chilean数据集创建轮廓管理器配置"""
        config = ContourManagerConfig()

        # 基础参数调整适应Chilean地下矿井环境
        config.lv_grads = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # 降低高度层级以适应地下环境
        config.reso_row = 0.8  # 提高分辨率
        config.reso_col = 0.8
        config.n_row = 200  # 增加网格大小以捕获更大区域
        config.n_col = 200
        config.lidar_height = 1.5  # 调整激光雷达高度
        config.blind_sq = 4.0  # 减小盲区
        config.min_cont_key_cnt = 6  # 降低最小轮廓关键点数量
        config.min_cont_cell_cnt = 2
        config.piv_firsts = 8  # 增加主要轮廓数量
        config.dist_firsts = 12
        config.roi_radius = 12.0  # 增加兴趣区域半径

        return config

    def create_contour_db_config_for_chilean(self) -> ContourDBConfig:
        """为Chilean数据集创建数据库配置"""
        config = ContourDBConfig()

        # 调整搜索参数适应地下环境
        config.nnk = 80  # 增加邻近搜索数量
        config.max_fine_opt = 15
        config.q_levels = [0, 1, 2, 3]  # 使用更多层级

        # 树桶配置
        config.tb_cfg.max_elapse = 30.0
        config.tb_cfg.min_elapse = 10.0

        # 轮廓相似性配置（放宽以适应环境变化）
        config.cont_sim_cfg.ta_cell_cnt = 15.0
        config.cont_sim_cfg.tp_cell_cnt = 0.6
        config.cont_sim_cfg.tp_eigval = 0.6
        config.cont_sim_cfg.ta_h_bar = 0.8
        config.cont_sim_cfg.ta_rcom = 1.5
        config.cont_sim_cfg.tp_rcom = 0.7

        return config

    def create_detection_thresholds(self) -> Tuple[CandidateScoreEnsemble, CandidateScoreEnsemble]:
        """创建检测阈值"""
        from contour_types import ScoreConstellSim, ScorePairwiseSim, ScorePostProc

        # 下界阈值（较宽松以适应时间跨度大的数据）
        thres_lb = CandidateScoreEnsemble()
        thres_lb.sim_constell.i_ovlp_sum = 2
        thres_lb.sim_constell.i_ovlp_max_one = 2
        thres_lb.sim_constell.i_in_ang_rng = 3
        thres_lb.sim_pair.i_indiv_sim = 3
        thres_lb.sim_pair.i_orie_sim = 3
        thres_lb.sim_post.correlation = 0.2
        thres_lb.sim_post.area_perc = 0.02
        thres_lb.sim_post.neg_est_dist = -8.0

        # 上界阈值
        thres_ub = CandidateScoreEnsemble()
        thres_ub.sim_constell.i_ovlp_sum = 10
        thres_ub.sim_constell.i_ovlp_max_one = 10
        thres_ub.sim_constell.i_in_ang_rng = 10
        thres_ub.sim_pair.i_indiv_sim = 10
        thres_ub.sim_pair.i_orie_sim = 10
        thres_ub.sim_post.correlation = 0.9
        thres_ub.sim_post.area_perc = 0.3
        thres_ub.sim_post.neg_est_dist = -3.0

        return thres_lb, thres_ub

    def load_sets_dict(self, filename: str) -> List[Dict]:
        """加载数据集字典"""
        try:
            with open(filename, 'rb') as handle:
                sets = pickle.load(handle)
                self.logger.info(f"加载 {filename}: {len(sets)} 个时间段")
                return sets
        except Exception as e:
            self.logger.error(f"加载 {filename} 失败: {e}")
            return []

    def load_chilean_pointcloud(self, filename: str) -> np.ndarray:
        """加载Chilean点云数据"""
        try:
            full_path = os.path.join(self.dataset_folder, filename)

            # 读取二进制数据
            pc = np.fromfile(full_path, dtype=np.float64)

            # 检查数据长度是否是3的倍数
            if len(pc) % 3 != 0:
                self.logger.warning(f"Chilean点云数据长度不是3的倍数: {len(pc)}")
                return np.array([])

            # reshape为 [N, 3] 格式
            num_points = len(pc) // 3
            pc = pc.reshape(num_points, 3)

            return pc

        except Exception as e:
            self.logger.error(f"加载Chilean点云 {filename} 失败: {e}")
            return np.array([])

    def apply_random_rotation(self, pointcloud: np.ndarray) -> np.ndarray:
        """对点云应用随机旋转（绕x、y、z轴）"""
        if len(pointcloud) == 0:
            return pointcloud

        # 生成随机旋转角度（弧度）
        angle_x = np.random.uniform(-np.pi, np.pi)
        angle_y = np.random.uniform(-np.pi, np.pi)
        angle_z = np.random.uniform(-np.pi, np.pi)

        # 创建旋转矩阵
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])

        R_y = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])

        R_z = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1]
        ])

        # 组合旋转矩阵（先绕z轴，再绕y轴，最后绕x轴）
        R = R_x @ R_y @ R_z

        # 应用旋转到点云的xyz坐标
        rotated_pointcloud = pointcloud.copy()
        rotated_pointcloud[:, :3] = (R @ pointcloud[:, :3].T).T

        return rotated_pointcloud

    def save_pointcloud_to_txt(self, pointcloud: np.ndarray, filename: str):
        """将点云保存为txt文件"""
        try:
            save_dir = "不旋转点云&随机旋转点云"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            filepath = os.path.join(save_dir, filename)

            # 保存点云数据 (x, y, z)
            np.savetxt(filepath, pointcloud, fmt='%.6f', delimiter=' ',
                       header='x y z', comments='')

            self.logger.info(f"点云已保存到: {filepath}")
            self.logger.info(f"点云信息: {len(pointcloud)} 个点")

        except Exception as e:
            self.logger.error(f"保存点云失败: {e}")

    def build_database(self, set_idx: int) -> bool:
        """为指定时间段构建数据库"""
        session_id = 100 + set_idx  # 数据库session从100开始
        self.logger.info(f"构建数据库时间段 {set_idx} (session {session_id})...")

        database_set = self.database_sets[set_idx]
        contour_db = self.contour_dbs[set_idx]

        processed_count = 0
        total_count = len(database_set)

        for key in sorted(database_set.keys()):
            item = database_set[key]
            filename = item['query']

            # 加载点云
            pointcloud = self.load_chilean_pointcloud(filename)
            if len(pointcloud) == 0:
                continue

            # 如果是第10个点云，保存原始点云
            if self.processed_cloud_count == 9:
                self.save_pointcloud_to_txt(pointcloud, "tenth_pointcloud_original.txt")
                self.logger.info(f"已保存第10个原始点云")

            # 应用随机旋转
            pointcloud = self.apply_random_rotation(pointcloud)

            # 如果是第10个点云，保存旋转后的点云
            if self.processed_cloud_count == 9:
                self.save_pointcloud_to_txt(pointcloud, "tenth_pointcloud_rotated.txt")
                self.logger.info(f"已保存第10个旋转后点云")

            self.processed_cloud_count += 1

            # 创建轮廓管理器并处理
            start_time = time.time()
            cm = ContourManager(self.cm_config, key)

            # 生成BEV和轮廓
            cm.make_bev(pointcloud, f"db_{set_idx}_{key}")
            cm.make_contours_recursive()

            desc_time = (time.time() - start_time) * 1000

            # 统计轮廓信息
            total_contours = sum(len(cm.get_lev_contours(i)) for i in range(len(self.cm_config.lv_grads)))
            valid_keys = sum(1 for level in range(len(self.cm_config.lv_grads))
                             for key_arr in cm.get_lev_retrieval_key(level)
                             if np.sum(key_arr) != 0)

            self.logger.info(f"[DB {set_idx}] Frame {key}: "
                             f"contours: {total_contours}, "
                             f"valid_keys: {valid_keys}, "
                             f"times: {desc_time:.1f}ms")

            # 添加到数据库
            start_time = time.time()
            contour_db.add_scan(cm, float(processed_count))  # 使用处理计数作为时间戳
            update_time = (time.time() - start_time) * 1000

            processed_count += 1

            if processed_count % 50 == 0:
                self.logger.info(f"  已处理: {processed_count}/{total_count}")

        self.logger.info(f"数据库时间段 {set_idx} (session {session_id}) 构建完成: {processed_count}/{total_count}")
        return processed_count > 0

    def query_in_database(self, query_set_idx: int, database_set_idx: int,
                          k: int = 25) -> Tuple[List, List, float]:
        """在指定数据库时间段中查询"""
        query_set = self.query_sets[query_set_idx]
        contour_db = self.contour_dbs[database_set_idx]

        recall = [0] * k
        top1_similarity_score = []
        one_percent_retrieved = 0
        threshold = max(int(round(len(self.database_sets[database_set_idx]) / 100.0)), 1)
        num_evaluated = 0

        # 失败案例统计
        failed_queries = []
        failure_statistics = {
            'no_ground_truth': 0,
            'pointcloud_load_failed': 0,
            'no_retrieval_results': 0,
            'no_valid_matches': 0,
            'total_queries': 0
        }

        for query_key in sorted(query_set.keys()):
            query_item = query_set[query_key]
            failure_statistics['total_queries'] += 1

            # 检查ground truth
            if database_set_idx not in query_item:
                failed_queries.append({
                    'query_key': query_key,
                    'filename': query_item['query'],
                    'failure_reason': 'no_ground_truth',
                    'details': f'No ground truth for database {database_set_idx}'
                })
                failure_statistics['no_ground_truth'] += 1
                continue

            true_neighbors = query_item[database_set_idx]
            if len(true_neighbors) == 0:
                failed_queries.append({
                    'query_key': query_key,
                    'filename': query_item['query'],
                    'failure_reason': 'empty_ground_truth',
                    'details': 'Ground truth list is empty'
                })
                failure_statistics['no_ground_truth'] += 1
                continue

            num_evaluated += 1

            # 加载查询点云
            filename = query_item['query']
            pointcloud = self.load_chilean_pointcloud(filename)
            if len(pointcloud) == 0:
                failed_queries.append({
                    'query_key': query_key,
                    'filename': filename,
                    'failure_reason': 'pointcloud_load_failed',
                    'details': 'Failed to load or empty pointcloud'
                })
                failure_statistics['pointcloud_load_failed'] += 1
                continue

            # 应用随机旋转（与数据库保持一致）
            pointcloud = self.apply_random_rotation(pointcloud)

            # 创建查询轮廓管理器
            try:
                query_cm = ContourManager(self.cm_config, query_key)
                query_cm.make_bev(pointcloud, f"query_{query_set_idx}_{query_key}")
                query_cm.make_contours_recursive()

                # 在数据库中检索
                start_time = time.time()
                cand_ptrs, cand_corr, cand_tf = contour_db.query_ranged_knn(
                    query_cm, self.thres_lb, self.thres_ub)
                query_time = time.time() - start_time

                if len(cand_ptrs) == 0:
                    failed_queries.append({
                        'query_key': query_key,
                        'filename': filename,
                        'failure_reason': 'no_retrieval_results',
                        'details': 'No retrieval results returned',
                        'ground_truth': true_neighbors
                    })
                    failure_statistics['no_retrieval_results'] += 1
                    continue

                # 提取索引（使用候选轮廓管理器的ID）
                indices = [cand.get_int_id() for cand in cand_ptrs]

                self.logger.info(f"Query {query_key}: found {len(indices)} candidates, "
                                 f"query_time: {query_time * 1000:.1f}ms")

                # 检查是否有有效匹配
                has_valid_match = False
                for j, idx in enumerate(indices):
                    if idx in true_neighbors:
                        has_valid_match = True
                        if j == 0:
                            similarity = cand_corr[0] if cand_corr else 0.0
                            top1_similarity_score.append(similarity)
                        for k_idx in range(j, len(recall)):
                            recall[k_idx] += 1
                        break

                if not has_valid_match:
                    failed_queries.append({
                        'query_key': query_key,
                        'filename': filename,
                        'failure_reason': 'no_valid_matches',
                        'details': f'Retrieved {len(indices)} results but none in ground truth',
                        'ground_truth': true_neighbors,
                        'retrieved_indices': indices,
                        'similarities': cand_corr if cand_corr else []
                    })
                    failure_statistics['no_valid_matches'] += 1

                # 计算top 1% recall
                top_percent_indices = indices[:threshold]
                if len(set(top_percent_indices).intersection(set(true_neighbors))) > 0:
                    one_percent_retrieved += 1

            except Exception as e:
                self.logger.error(f"查询处理错误 {query_key}: {e}")
                failed_queries.append({
                    'query_key': query_key,
                    'filename': filename,
                    'failure_reason': 'processing_error',
                    'details': f'Processing error: {str(e)}'
                })
                failure_statistics['pointcloud_load_failed'] += 1
                continue

        # 记录失败统计
        if failed_queries:
            self.logger.info(f"\n=== 失败查询分析 (DB{database_set_idx} <- Query{query_set_idx}) ===")
            self.logger.info(f"总查询数: {failure_statistics['total_queries']}")
            self.logger.info(f"成功评估数: {num_evaluated}")
            self.logger.info(f"失败统计:")
            for reason, count in failure_statistics.items():
                if reason != 'total_queries' and count > 0:
                    self.logger.info(f"  {reason}: {count}")

        if num_evaluated > 0:
            recall = [(r / num_evaluated) * 100 for r in recall]
            one_percent_recall = (one_percent_retrieved / num_evaluated) * 100
        else:
            recall = [0] * k
            one_percent_recall = 0

        return recall, top1_similarity_score, one_percent_recall

    def evaluate(self) -> float:
        """执行完整评估"""
        self.logger.info("开始Chilean数据集ContourContext评估...")

        # 构建所有数据库
        for i in range(len(self.database_sets)):
            if not self.build_database(i):
                self.logger.error(f"数据库时间段 {i} 构建失败")
                return 0.0

        self.logger.info("\n开始跨时间段评估...")

        recall = np.zeros(25)
        count = 0
        similarity = []
        one_percent_recall = []

        # 跨时间段评估
        for m in range(len(self.database_sets)):  # 数据库时间段 (100-101)
            for n in range(len(self.query_sets)):  # 查询时间段 (180-181)
                db_session_id = 100 + m
                query_session_id = 180 + n
                self.logger.info(f"评估：查询时间段{query_session_id} -> 数据库时间段{db_session_id}")

                pair_recall, pair_similarity, pair_opr = self.query_in_database(n, m)
                recall += np.array(pair_recall)
                count += 1
                one_percent_recall.append(pair_opr)

                for x in pair_similarity:
                    similarity.append(x)

                self.logger.info(f"  Recall@1: {pair_recall[0]:.2f}%, Top1%: {pair_opr:.2f}%")

        # 计算平均结果
        if count > 0:
            ave_recall = recall / count
            average_similarity = np.mean(similarity) if similarity else 0
            ave_one_percent_recall = np.mean(one_percent_recall)
        else:
            ave_recall = np.zeros(25)
            average_similarity = 0
            ave_one_percent_recall = 0

        # 输出结果
        self.logger.info(f"\n=== Chilean数据集 ContourContext 评估结果 ===")
        self.logger.info(f"数据库时间段: 100-101 (历史地图)")
        self.logger.info(f"查询时间段: 180-181 (当前观测)")
        self.logger.info(f"Average Recall @1: {ave_recall[0]:.2f}%")
        self.logger.info(f"Average Recall @5: {ave_recall[4]:.2f}%")
        self.logger.info(f"Average Recall @10: {ave_recall[9]:.2f}%")
        self.logger.info(f"Average Recall @25: {ave_recall[24]:.2f}%")
        self.logger.info(f"Average Similarity: {average_similarity:.4f}")
        self.logger.info(f"Average Top 1% Recall: {ave_one_percent_recall:.2f}%")

        # 保存详细结果
        self.save_results(ave_recall, average_similarity, ave_one_percent_recall)

        return ave_one_percent_recall

    def save_results(self, ave_recall: np.ndarray, average_similarity: float,
                     ave_one_percent_recall: float):
        """保存评估结果"""
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        output_file = os.path.join(results_dir, "contourcontext_results_chilean_不同时段_session.txt")

        with open(output_file, "w") as f:
            f.write("Chilean Dataset ContourContext Evaluation Results\n")
            f.write("=" * 60 + "\n\n")
            f.write("Dataset Info:\n")
            f.write("Database: Sessions 100-101 (Historical Map)\n")
            f.write("Query: Sessions 180-181 (Current Observations)\n\n")
            f.write("Average Recall @N:\n")
            f.write(str(ave_recall) + "\n\n")
            f.write("Average Similarity:\n")
            f.write(str(average_similarity) + "\n\n")
            f.write("Average Top 1% Recall:\n")
            f.write(str(ave_one_percent_recall) + "\n")

        self.logger.info(f"结果已保存到: {output_file}")


def main():
    """主函数"""
    # ========== 配置参数 - 在PyCharm中直接修改这里 ==========
    DATASET_FOLDER = '/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times'  # 修改为你的Chilean数据集路径
    DATABASE_FILE = 'chilean_NoRot_NoScale_5cm_evaluation_database_100_101.pickle'  # 数据库pickle文件
    QUERY_FILE = 'chilean_NoRot_NoScale_5cm_evaluation_query_180_181.pickle'  # 查询pickle文件
    LOG_FILE = 'contourcontext_chilean_不同时段_log.txt'  # 日志文件
    # =====================================================

    # 如果仍然希望支持命令行参数，可以取消注释以下代码
    import argparse
    parser = argparse.ArgumentParser(description='Chilean数据集上的ContourContext评估')
    parser.add_argument('--dataset_folder', type=str, default=DATASET_FOLDER,
                        help='Chilean数据集文件夹路径')
    parser.add_argument('--database_file', type=str, default=DATABASE_FILE,
                        help='数据库pickle文件名')
    parser.add_argument('--query_file', type=str, default=QUERY_FILE,
                        help='查询pickle文件名')
    parser.add_argument('--log_file', type=str, default=LOG_FILE,
                        help='日志文件名')

    args = parser.parse_args()

    # 使用配置参数
    dataset_folder = args.dataset_folder
    database_file = args.database_file
    query_file = args.query_file
    log_file = args.log_file

    # 检查必要文件
    if not os.path.exists(database_file):
        print(f"错误：找不到数据库文件 {database_file}")
        print("请先运行 generate_test_sets_chilean_NoRot_period.py 生成评估数据")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"寻找文件: {os.path.abspath(database_file)}")
        return

    if not os.path.exists(query_file):
        print(f"错误：找不到查询文件 {query_file}")
        print("请先运行 generate_test_sets_chilean_NoRot_period.py 生成评估数据")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"寻找文件: {os.path.abspath(query_file)}")
        return

    if not os.path.exists(dataset_folder):
        print(f"错误：找不到数据集文件夹 {dataset_folder}")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"寻找路径: {os.path.abspath(dataset_folder)}")
        return

    print(f"数据集路径: {dataset_folder}")
    print(f"数据库文件: {database_file}")
    print(f"查询文件: {query_file}")
    print(f"日志文件: {log_file}")
    print(f"当前工作目录: {os.getcwd()}")
    print("-" * 60)

    # 创建评估器并运行评估
    evaluator = ChileanContourContextEvaluator(
        dataset_folder=dataset_folder,
        database_file=database_file,
        query_file=query_file,
        log_file=log_file
    )

    # 执行评估
    start_time = time.time()
    top1_recall = evaluator.evaluate()
    end_time = time.time()

    print(f"\n评估完成!")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"最终Top 1% Recall: {top1_recall:.2f}%")
    print(f"详细日志已保存到: {log_file}")


if __name__ == "__main__":
    main()