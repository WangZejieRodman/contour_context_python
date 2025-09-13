#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contour Context Chilean场景识别评估
基于main_loop_closure.py改编，用于Chilean数据集的跨时间段场景识别
评估任务：历史地图(100-100) vs 当前观测(100-100)
"""

import numpy as np
import os
import sys
import pickle
import time
from typing import List, Dict, Tuple, Optional
import logging

# 导入Contour Context相关模块
from contour_types import (
    ContourManagerConfig, ContourDBConfig, CandidateScoreEnsemble,
    ContourSimThresConfig, TreeBucketConfig
)
from contour_manager import ContourManager
from contour_database import ContourDB


class ChileanContourEvaluator:
    """Chilean数据集上的Contour Context场景识别评估器"""

    def __init__(self, dataset_folder: str,
                 database_file: str = 'chilean_NoRot_NoScale_5cm_evaluation_database_100_100.pickle',
                 query_file: str = 'chilean_NoRot_NoScale_5cm_evaluation_query_100_100.pickle',
                 log_file: str = 'contour_chilean_不同时段_log.txt'):
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

        # 创建配置
        self.cm_config = self.create_contour_manager_config()
        self.db_config = self.create_contour_db_config()
        self.thres_lb, self.thres_ub = self.create_similarity_thresholds()

        # 创建数据库
        self.contour_db = ContourDB(self.db_config)

        # 存储轮廓管理器用于重叠计算
        self.database_cms = []

        self.logger.info(f"初始化完成：{len(self.database_sets)}个数据库时间段，{len(self.query_sets)}个查询时间段")
        self.logger.info(f"评估任务：历史地图(100-100) vs 当前观测(100-100)")
        self.processed_cloud_count = 0

    def create_contour_manager_config(self) -> ContourManagerConfig:
        """为Chilean数据集创建轮廓管理器配置"""
        config = ContourManagerConfig()

        # 根据Chilean地下矿井环境调整参数
        config.lv_grads = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  # 高度阈值
        config.reso_row = 0.5  # 提高分辨率
        config.reso_col = 0.5
        config.n_row = 200  # 增大网格以适应地下环境
        config.n_col = 200
        config.lidar_height = 1.0  # 地下环境激光雷达高度较低
        config.blind_sq = 4.0  # 减小盲区
        config.min_cont_key_cnt = 6  # 降低最小轮廓键数量
        config.min_cont_cell_cnt = 2  # 降低最小轮廓像素数量
        config.piv_firsts = 8  # 增加关键轮廓数量
        config.dist_firsts = 12
        config.roi_radius = 15.0  # 增大感兴趣区域半径

        return config

    def create_contour_db_config(self) -> ContourDBConfig:
        """创建轮廓数据库配置"""
        config = ContourDBConfig()

        config.nnk = 30  # 降低KNN搜索数量以提高召回率
        config.max_fine_opt = 5  # 减少精细优化候选数
        config.q_levels = [1, 2, 3, 4]  # 使用更多层级

        # 树桶配置
        tb_cfg = TreeBucketConfig()
        tb_cfg.max_elapse = 30.0
        tb_cfg.min_elapse = 10.0
        config.tb_cfg = tb_cfg

        # 轮廓相似性配置 - 放宽阈值以适应地下环境
        sim_cfg = ContourSimThresConfig()
        sim_cfg.ta_cell_cnt = 15.0  # 放宽绝对面积差异
        sim_cfg.tp_cell_cnt = 0.6  # 放宽相对面积差异
        sim_cfg.tp_eigval = 0.6  # 放宽特征值差异
        sim_cfg.ta_h_bar = 1.2  # 放宽高度差异
        sim_cfg.ta_rcom = 1.5  # 放宽质心半径差异
        sim_cfg.tp_rcom = 0.7  # 放宽质心半径相对差异
        config.cont_sim_cfg = sim_cfg

        return config

    def create_similarity_thresholds(self) -> Tuple[CandidateScoreEnsemble, CandidateScoreEnsemble]:
        """创建检测阈值"""
        # 下界阈值（较宽松以适应时间跨度大的数据）
        thres_lb = CandidateScoreEnsemble()
        thres_lb.sim_constell.i_ovlp_sum = 1  # 进一步降低
        thres_lb.sim_constell.i_ovlp_max_one = 1
        thres_lb.sim_constell.i_in_ang_rng = 1  # 最低要求
        thres_lb.sim_pair.i_indiv_sim = 1
        thres_lb.sim_pair.i_orie_sim = 1  # 最低要求
        thres_lb.sim_post.correlation = 0.001  # 几乎不要求
        thres_lb.sim_post.area_perc = 0.0001
        thres_lb.sim_post.neg_est_dist = -100.0  # 很宽松的距离要求

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
        """加载Chilean点云文件"""
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

        # 组合旋转矩阵
        R = R_x @ R_y @ R_z

        # 应用旋转到点云的xyz坐标
        rotated_pointcloud = (R @ pointcloud.T).T

        return rotated_pointcloud

    def save_pointcloud_to_txt(self, pointcloud: np.ndarray, filename: str):
        """将点云保存为txt文件"""
        try:
            save_dir = "不旋转点云&随机旋转点云"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            filepath = os.path.join(save_dir, filename)
            np.savetxt(filepath, pointcloud, fmt='%.6f', delimiter=' ',
                       header='x y z', comments='')

            self.logger.info(f"点云已保存到: {filepath}")
            self.logger.info(f"点云信息: {len(pointcloud)} 个点")

        except Exception as e:
            self.logger.error(f"保存点云失败: {e}")

    def build_database(self, set_idx: int) -> bool:
        """为指定时间段构建数据库"""
        session_id = 100 + set_idx
        self.logger.info(f"构建数据库时间段 {set_idx} (session {session_id})...")

        database_set = self.database_sets[set_idx]
        processed_count = 0
        total_count = len(database_set)

        for key in sorted(database_set.keys()):
            item = database_set[key]
            filename = item['query']

            # 加载点云
            pointcloud = self.load_chilean_pointcloud(filename)
            if len(pointcloud) == 0:
                continue

            # 如果是第10个点云，保存原始和旋转后的点云
            if self.processed_cloud_count == 9:
                self.save_pointcloud_to_txt(pointcloud, "tenth_pointcloud_original.txt")
                self.logger.info(f"已保存第10个原始点云")

            # 应用随机旋转
            pointcloud = self.apply_random_rotation(pointcloud)

            if self.processed_cloud_count == 9:
                self.save_pointcloud_to_txt(pointcloud, "tenth_pointcloud_rotated.txt")
                self.logger.info(f"已保存第10个旋转后点云")

            self.processed_cloud_count += 1

            # 创建轮廓管理器
            start_time = time.time()
            cm = ContourManager(self.cm_config, key)

            # 生成字符串ID
            str_id = f"db_session_{session_id}_frame_{key}"

            # 处理点云
            cm.make_bev(pointcloud, str_id)
            cm.make_contours_recursive()

            desc_time = (time.time() - start_time) * 1000

            # 记录处理信息
            total_contours = sum(len(cm.get_lev_contours(i))
                                 for i in range(len(self.cm_config.lv_grads)))

            self.logger.info(f"[DB {set_idx}] Frame {key}: 总轮廓数: {total_contours}, "
                             f"处理时间: {desc_time:.1f}ms")

            # 添加到数据库
            start_time = time.time()
            self.contour_db.add_scan(cm, processed_count)  # 使用processed_count作为时间戳
            update_time = (time.time() - start_time) * 1000

            # 存储轮廓管理器
            self.database_cms.append(cm)

            processed_count += 1

            if processed_count % 50 == 0:
                self.logger.info(f"  已处理: {processed_count}/{total_count}")

        self.logger.info(f"数据库时间段 {set_idx} (session {session_id}) 构建完成: {processed_count}/{total_count}")

        # 强制将所有缓冲区数据移动到KD树
        if processed_count > 0:
            self.logger.info("开始强制平衡，将缓冲区数据移动到KD树...")

            # 使用一个很大的时间戳强制触发数据迁移
            final_timestamp = 999999.0

            # 打印平衡前的状态
            self.logger.info("平衡前的状态：")
            for ll in range(len(self.contour_db.layer_db)):
                layer_db = self.contour_db.layer_db[ll]
                tree_size = sum(bucket.get_tree_size() for bucket in layer_db.buckets)
                buffer_size = sum(len(bucket.buffer) for bucket in layer_db.buckets)
                self.logger.info(f"  LayerDB {ll}: 树大小={tree_size}, 缓冲区大小={buffer_size}")

            # 多次调用平衡操作确保所有数据都被处理
            self.logger.info("执行自动平衡操作...")
            for balance_round in range(10):
                for bucket_pair in range(5):  # 0到4，对应bucket pairs (0,1), (1,2), (2,3), (3,4), (4,5)
                    self.contour_db.push_and_balance(bucket_pair, final_timestamp)

            # 检查自动平衡后的状态
            self.logger.info("自动平衡后的状态：")
            any_data_in_tree = False
            for ll in range(len(self.contour_db.layer_db)):
                layer_db = self.contour_db.layer_db[ll]
                tree_size = sum(bucket.get_tree_size() for bucket in layer_db.buckets)
                buffer_size = sum(len(bucket.buffer) for bucket in layer_db.buckets)
                if tree_size > 0:
                    any_data_in_tree = True
                self.logger.info(f"  LayerDB {ll}: 树大小={tree_size}, 缓冲区大小={buffer_size}")

            # 如果自动平衡失败，手动清空缓冲区到树中
            if not any_data_in_tree:
                self.logger.info("自动平衡失败，开始手动清空缓冲区...")

                for ll in range(len(self.contour_db.layer_db)):
                    layer_db = self.contour_db.layer_db[ll]
                    self.logger.info(f"处理LayerDB {ll}...")

                    for bucket_idx, bucket in enumerate(layer_db.buckets):
                        if len(bucket.buffer) > 0:
                            self.logger.info(f"  手动清空Bucket {bucket_idx}: {len(bucket.buffer)} 个项目")

                            # 将缓冲区数据强制移动到树
                            for item in bucket.buffer:
                                tree_key, ts, iok = item
                                bucket.data_tree.append(tree_key.copy())
                                bucket.gkidx_tree.append(iok)

                            # 清空缓冲区
                            bucket.buffer.clear()

                            # 重建树
                            try:
                                bucket.rebuild_tree()
                                self.logger.info(f"    Bucket {bucket_idx} 树重建成功，新大小: {bucket.get_tree_size()}")
                            except Exception as e:
                                self.logger.error(f"    Bucket {bucket_idx} 树重建失败: {e}")

            # 最终检查和验证
            self.logger.info("最终状态检查：")
            total_tree_size = 0
            total_buffer_size = 0

            for ll in range(len(self.contour_db.layer_db)):
                layer_db = self.contour_db.layer_db[ll]
                tree_size = sum(bucket.get_tree_size() for bucket in layer_db.buckets)
                buffer_size = sum(len(bucket.buffer) for bucket in layer_db.buckets)
                total_tree_size += tree_size
                total_buffer_size += buffer_size

                self.logger.info(f"  LayerDB {ll} (q_level={self.contour_db.cfg.q_levels[ll]}): "
                                 f"树大小={tree_size}, 缓冲区大小={buffer_size}")

                # 打印每个非空桶的详细信息
                for bucket_idx, bucket in enumerate(layer_db.buckets):
                    bucket_tree_size = bucket.get_tree_size()
                    bucket_buffer_size = len(bucket.buffer)
                    if bucket_tree_size > 0 or bucket_buffer_size > 0:
                        self.logger.info(f"    Bucket {bucket_idx}: 树={bucket_tree_size}, 缓冲区={bucket_buffer_size}")

            self.logger.info(f"总计: 树大小={total_tree_size}, 缓冲区大小={total_buffer_size}")

            # 验证结果
            if total_tree_size == 0:
                self.logger.error("严重错误：所有数据仍在缓冲区中，KD树为空！")
                self.logger.error("这将导致查询时无法找到任何匹配。")

                # 尝试最后的手段：直接操作内部数据结构
                self.logger.info("尝试最后的修复手段...")
                try:
                    for ll in range(len(self.contour_db.layer_db)):
                        layer_db = self.contour_db.layer_db[ll]
                        for bucket in layer_db.buckets:
                            if len(bucket.buffer) > 0:
                                # 强制调用pop_buffer_max
                                bucket.pop_buffer_max(final_timestamp)
                                self.logger.info(f"强制调用pop_buffer_max后，树大小: {bucket.get_tree_size()}")

                    # 再次检查
                    final_tree_size = sum(
                        sum(bucket.get_tree_size() for bucket in layer_db.buckets)
                        for layer_db in self.contour_db.layer_db
                    )
                    self.logger.info(f"最终修复后树大小: {final_tree_size}")

                    if final_tree_size == 0:
                        return False

                except Exception as e:
                    self.logger.error(f"最后修复尝试失败: {e}")
                    return False
            else:
                self.logger.info(f"✅ 成功：数据已移动到KD树中")

        return processed_count > 0

    def query_in_database(self, query_set_idx: int, database_set_idx: int,
                          k: int = 25) -> Tuple[List, List, float]:
        """在指定数据库时间段中查询"""
        query_set = self.query_sets[query_set_idx]

        recall = [0] * k
        top1_similarity_score = []
        one_percent_retrieved = 0
        threshold = max(int(round(len(self.database_sets[database_set_idx]) / 100.0)), 1)
        num_evaluated = 0

        # 失败统计
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
                failure_statistics['no_ground_truth'] += 1
                continue

            true_neighbors = query_item[database_set_idx]
            if len(true_neighbors) == 0:
                failure_statistics['no_ground_truth'] += 1
                continue

            num_evaluated += 1

            # 加载查询点云
            filename = query_item['query']
            pointcloud = self.load_chilean_pointcloud(filename)
            if len(pointcloud) == 0:
                failure_statistics['pointcloud_load_failed'] += 1
                continue

            # 应用随机旋转
            # pointcloud = self.apply_random_rotation(pointcloud)

            # 创建查询轮廓管理器
            query_cm = ContourManager(self.cm_config, query_key + 10000)  # 避免ID冲突
            str_id = f"query_session_{100 + query_set_idx}_frame_{query_key}"
            query_cm.make_bev(pointcloud, str_id)
            query_cm.make_contours_recursive()

            # 在数据库中检索
            try:
                candidate_cms, correlations, transforms = self.contour_db.query_ranged_knn(
                    query_cm, self.thres_lb, self.thres_ub)

                if len(candidate_cms) == 0:
                    failure_statistics['no_retrieval_results'] += 1
                    continue

                # 将候选CM ID映射回数据库索引
                results = []
                for i, (candidate_cm, correlation) in enumerate(zip(candidate_cms, correlations)):
                    # 假设数据库CM的ID就是其在database_cms中的索引
                    db_idx = candidate_cm.get_int_id()
                    if db_idx < len(self.database_cms):
                        distance = 1.0 - correlation  # 将相关性转换为距离
                        results.append((db_idx, distance))

                if len(results) == 0:
                    failure_statistics['no_retrieval_results'] += 1
                    continue

                # 按距离排序，返回前k个
                results.sort(key=lambda x: x[1])
                results = results[:k]

                indices = [result[0] for result in results]

            except Exception as e:
                self.logger.error(f"查询失败: {e}")
                failure_statistics['no_retrieval_results'] += 1
                continue

            # 检查是否有有效匹配
            has_valid_match = False
            for j, idx in enumerate(indices):
                if idx in true_neighbors:
                    has_valid_match = True
                    if j == 0:
                        similarity = 1.0 - results[0][1]
                        top1_similarity_score.append(similarity)
                    for k_idx in range(j, len(recall)):
                        recall[k_idx] += 1
                    break

            if not has_valid_match:
                failure_statistics['no_valid_matches'] += 1

            # 计算top 1% recall
            top_percent_indices = indices[:threshold]
            if len(set(top_percent_indices).intersection(set(true_neighbors))) > 0:
                one_percent_retrieved += 1

        # 记录失败统计
        if failure_statistics['total_queries'] > 0:
            self.logger.info(f"\n=== 失败查询分析 (DB{database_set_idx} <- Query{query_set_idx}) ===")
            self.logger.info(f"总查询数: {failure_statistics['total_queries']}")
            self.logger.info(f"成功评估数: {num_evaluated}")
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
        self.logger.info("开始Chilean数据集Contour Context评估...")

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
        for m in range(len(self.database_sets)):  # 数据库时间段 (100-100)
            for n in range(len(self.query_sets)):  # 查询时间段 (100-100)
                db_session_id = 100 + m
                query_session_id = 100 + n
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
        self.logger.info(f"\n=== Chilean数据集 Contour Context 评估结果 ===")
        self.logger.info(f"数据库时间段: 100-100 (历史地图)")
        self.logger.info(f"查询时间段: 100-100 (当前观测)")
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

        output_file = os.path.join(results_dir, "contour_results_chilean_不同时段_session.txt")

        with open(output_file, "w") as f:
            f.write("Chilean Dataset Contour Context Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write("Dataset Info:\n")
            f.write("Database: Sessions 100-100 (Historical Map)\n")
            f.write("Query: Sessions 100-100 (Current Observations)\n\n")
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
    DATABASE_FILE = 'chilean_NoRot_NoScale_5cm_evaluation_database_100_100.pickle'  # 数据库pickle文件
    QUERY_FILE = 'chilean_NoRot_NoScale_5cm_evaluation_query_100_100.pickle'  # 查询pickle文件
    LOG_FILE = 'contour_chilean_不同时段_log.txt'  # 日志文件
    # =====================================================

    # 如果仍然希望支持命令行参数，可以取消注释以下代码
    import argparse
    parser = argparse.ArgumentParser(description='Chilean数据集上的Contour Context评估')
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
    evaluator = ChileanContourEvaluator(
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
