#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ContourContext Chilean场景识别评估
基于contour_context改编，用于Chilean数据集的跨时间段场景识别
评估任务：历史地图(100-101) vs 当前观测(100-100)
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
                 query_file: str = 'chilean_NoRot_NoScale_5cm_evaluation_query_100_100.pickle',
                 log_file: str = 'contourcontext_chilean_不旋转_相同时段_log.txt'):
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
        self.logger.info(f"评估任务：历史地图(100-100) vs 当前观测(100-100)")
        self.processed_cloud_count = 0

    def create_contour_manager_config_for_chilean(self) -> ContourManagerConfig:
        config = ContourManagerConfig()

        # 大幅增加ROI半径以增加环形特征变化
        config.roi_radius = 30.0  # 从12.0增加到30.0

        # 更精细的分辨率增加轮廓细节
        config.reso_row = 0.4
        config.reso_col = 0.4

        # 更多的高度层级增加多样性
        config.lv_grads = [0.1, 0.5, 1.2, 2.5, 4.5, 7.0]

        # 增加轮廓数量
        config.piv_firsts = 12
        config.dist_firsts = 20

        return config

    def create_contour_db_config_for_chilean(self) -> ContourDBConfig:
        """为Chilean数据集创建数据库配置"""
        config = ContourDBConfig()

        # 调整搜索参数适应地下环境
        config.nnk = 80  # 增加邻近搜索数量
        config.max_fine_opt = 15
        config.q_levels = [0, 1, 2, 3]  # 使用更多层级

        # 树桶配置
        config.tb_cfg.max_elapse = 0.1
        config.tb_cfg.min_elapse = 0.05

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
            save_dir = "不旋转点云"
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

            # 不应用随机旋转（按照您的要求）
            # pointcloud = self.apply_random_rotation(pointcloud)

            self.processed_cloud_count += 1

            # 创建轮廓管理器并处理
            start_time = time.time()
            cm = ContourManager(self.cm_config, key + 100000)  # 数据库ID加偏移

            # 生成BEV和轮廓
            cm.make_bev(pointcloud, f"db_{set_idx}_{key}")
            cm.make_contours_recursive()

            desc_time = (time.time() - start_time) * 1000

            # 统计轮廓信息
            total_contours = sum(len(cm.get_lev_contours(i)) for i in range(len(self.cm_config.lv_grads)))
            valid_keys = sum(1 for level in range(len(self.cm_config.lv_grads))
                             for key_arr in cm.get_lev_retrieval_key(level)
                             if np.sum(key_arr) != 0)

            self.logger.info(f"[DB {set_idx}] Frame {key}: contours: {total_contours}, valid_keys: {valid_keys}")

            # 添加到数据库 - 使用较大的时间间隔
            start_time = time.time()
            timestamp = float(processed_count * 20.0)  # 每次间隔20秒
            contour_db.add_scan(cm, timestamp)
            update_time = (time.time() - start_time) * 1000

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
            for ll in range(len(contour_db.layer_db)):
                layer_db = contour_db.layer_db[ll]
                tree_size = sum(bucket.get_tree_size() for bucket in layer_db.buckets)
                buffer_size = sum(len(bucket.buffer) for bucket in layer_db.buckets)
                self.logger.info(f"  LayerDB {ll}: 树大小={tree_size}, 缓冲区大小={buffer_size}")

            # 多次调用平衡操作确保所有数据都被处理
            self.logger.info("执行自动平衡操作...")
            for balance_round in range(10):
                for bucket_pair in range(5):  # 0到4，对应bucket pairs (0,1), (1,2), (2,3), (3,4), (4,5)
                    contour_db.push_and_balance(bucket_pair, final_timestamp)

            # 检查自动平衡后的状态
            self.logger.info("自动平衡后的状态：")
            any_data_in_tree = False
            for ll in range(len(contour_db.layer_db)):
                layer_db = contour_db.layer_db[ll]
                tree_size = sum(bucket.get_tree_size() for bucket in layer_db.buckets)
                buffer_size = sum(len(bucket.buffer) for bucket in layer_db.buckets)
                if tree_size > 0:
                    any_data_in_tree = True
                self.logger.info(f"  LayerDB {ll}: 树大小={tree_size}, 缓冲区大小={buffer_size}")

            # 如果自动平衡失败，手动清空缓冲区到树中
            if not any_data_in_tree:
                self.logger.info("自动平衡失败，开始手动清空缓冲区...")

                for ll in range(len(contour_db.layer_db)):
                    layer_db = contour_db.layer_db[ll]
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

            for ll in range(len(contour_db.layer_db)):
                layer_db = contour_db.layer_db[ll]
                tree_size = sum(bucket.get_tree_size() for bucket in layer_db.buckets)
                buffer_size = sum(len(bucket.buffer) for bucket in layer_db.buckets)
                total_tree_size += tree_size
                total_buffer_size += buffer_size

                self.logger.info(f"  LayerDB {ll} (q_level={contour_db.cfg.q_levels[ll]}): "
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
                    for ll in range(len(contour_db.layer_db)):
                        layer_db = contour_db.layer_db[ll]
                        for bucket in layer_db.buckets:
                            if len(bucket.buffer) > 0:
                                # 强制调用pop_buffer_max
                                bucket.pop_buffer_max(final_timestamp)
                                self.logger.info(f"强制调用pop_buffer_max后，树大小: {bucket.get_tree_size()}")

                    # 再次检查
                    final_tree_size = sum(
                        sum(bucket.get_tree_size() for bucket in layer_db.buckets)
                        for layer_db in contour_db.layer_db
                    )
                    self.logger.info(f"最终修复后树大小: {final_tree_size}")

                    if final_tree_size == 0:
                        return False

                except Exception as e:
                    self.logger.error(f"最后修复尝试失败: {e}")
                    return False
            else:
                self.logger.info(f"✅ 成功！数据已移动到KD树中")

        return processed_count > 0

    def query_in_database(self, query_set_idx: int, database_set_idx: int,
                          k: int = 25) -> Tuple[List, List, float]:
        """在指定数据库时间段中查询 - 带详细调试信息"""
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

        # ======= 添加数据库状态检查 =======
        self.logger.info(f"\n=== 数据库调试信息 (DB {database_set_idx}) ===")
        self.logger.info(f"数据库大小: {len(contour_db.all_bevs)} 个扫描")

        # 检查layer_db状态
        self.logger.info(f"LayerDB数量: {len(contour_db.layer_db)}")
        for ll in range(len(contour_db.layer_db)):
            layer_db = contour_db.layer_db[ll]
            total_tree_size = sum(bucket.get_tree_size() for bucket in layer_db.buckets)
            total_buffer_size = sum(len(bucket.buffer) for bucket in layer_db.buckets)
            self.logger.info(
                f"LayerDB {ll} (q_level={contour_db.cfg.q_levels[ll]}): 树大小={total_tree_size}, 缓冲区大小={total_buffer_size}")

            # 打印每个桶的状态
            for bucket_idx, bucket in enumerate(layer_db.buckets):
                if bucket.get_tree_size() > 0 or len(bucket.buffer) > 0:
                    self.logger.info(f"  Bucket {bucket_idx}: 树={bucket.get_tree_size()}, 缓冲区={len(bucket.buffer)}")

        # 检查数据库中的检索键
        if len(contour_db.all_bevs) > 0:
            self.logger.info(f"=== 数据库检索键分析 ===")
            first_cm = contour_db.all_bevs[0]
            for level in range(min(4, len(self.cm_config.lv_grads))):  # 只检查前4个层级
                keys = first_cm.get_lev_retrieval_key(level)
                self.logger.info(f"数据库Level {level}: {len(keys)} keys")
                if len(keys) > 0:
                    valid_keys = [key for key in keys if np.sum(key) != 0]
                    self.logger.info(f"  有效keys数量: {len(valid_keys)}")
                    if len(valid_keys) > 0:
                        first_key = valid_keys[0]
                        self.logger.info(f"  第一个有效key形状: {first_key.shape}")
                        self.logger.info(f"  第一个有效key: {first_key}")
                        self.logger.info(
                            f"  key统计: min={np.min(first_key):.6f}, max={np.max(first_key):.6f}, mean={np.mean(first_key):.6f}")
                        self.logger.info(f"  非零元素数量: {np.count_nonzero(first_key)}")

        # 检查第一个查询的详细信息 (只检查前3个查询避免日志过多)
        query_keys_to_debug = list(sorted(query_set.keys()))[:3]

        for query_idx, query_key in enumerate(sorted(query_set.keys())):
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

            # 应用随机旋转（与数据库保持一致） - 如果不要旋转可以注释掉这行
            # pointcloud = self.apply_random_rotation(pointcloud)

            # 创建查询轮廓管理器
            try:
                query_cm = ContourManager(self.cm_config, query_key + 200000)
                query_cm.make_bev(pointcloud, f"query_{query_set_idx}_{query_key}")
                query_cm.make_contours_recursive()

                # ======= 添加查询轮廓调试信息 (只对前几个查询) =======
                if query_key in query_keys_to_debug:
                    self.logger.info(f"\n=== 查询调试信息 Query {query_key} ===")
                    self.logger.info(f"点云大小: {len(pointcloud)} 点")
                    self.logger.info(f"Ground truth neighbors: {true_neighbors}")

                    # 轮廓统计
                    total_contours = sum(len(query_cm.get_lev_contours(i)) for i in range(len(self.cm_config.lv_grads)))
                    self.logger.info(f"生成轮廓总数: {total_contours}")

                    # 检索键分析
                    for level in range(min(4, len(self.cm_config.lv_grads))):
                        q_keys = query_cm.get_lev_retrieval_key(level)
                        self.logger.info(f"查询Level {level}: {len(q_keys)} keys")
                        if len(q_keys) > 0:
                            valid_q_keys = [key for key in q_keys if np.sum(key) != 0]
                            self.logger.info(f"  有效查询keys数量: {len(valid_q_keys)}")
                            if len(valid_q_keys) > 0:
                                first_q_key = valid_q_keys[0]
                                self.logger.info(f"  第一个有效查询key形状: {first_q_key.shape}")
                                self.logger.info(f"  第一个有效查询key: {first_q_key}")
                                self.logger.info(
                                    f"  查询key统计: min={np.min(first_q_key):.6f}, max={np.max(first_q_key):.6f}, mean={np.mean(first_q_key):.6f}")
                                self.logger.info(f"  非零元素数量: {np.count_nonzero(first_q_key)}")

                                # 与数据库key对比
                                if len(contour_db.all_bevs) > 0:
                                    db_keys = contour_db.all_bevs[0].get_lev_retrieval_key(level)
                                    if len(db_keys) > 0:
                                        valid_db_keys = [key for key in db_keys if np.sum(key) != 0]
                                        if len(valid_db_keys) > 0:
                                            first_db_key = valid_db_keys[0]
                                            diff = np.linalg.norm(first_q_key - first_db_key)
                                            self.logger.info(f"  与数据库第一个key的欧氏距离: {diff:.6f}")

                    # BCI信息
                    for level in [0, 1, 2]:  # 检查前3个层级的BCI
                        if level < len(query_cm.get_config().lv_grads):
                            bcis = query_cm.get_lev_bci(level)
                            self.logger.info(f"Level {level} BCI数量: {len(bcis)}")
                            if len(bcis) > 0:
                                first_bci = bcis[0]
                                self.logger.info(f"  第一个BCI - seq: {first_bci.piv_seq}, level: {first_bci.level}")
                                self.logger.info(f"  BCI邻居点数量: {len(first_bci.nei_pts)}")
                                self.logger.info(f"  BCI距离bins中True的数量: {np.sum(first_bci.dist_bin)}")

                # 在数据库中检索
                start_time = time.time()
                self.logger.info(f"开始查询 {query_key} 在数据库中...")

                cand_ptrs, cand_corr, cand_tf = contour_db.query_ranged_knn(
                    query_cm, self.thres_lb, self.thres_ub)
                query_time = time.time() - start_time

                self.logger.info(
                    f"Query {query_key}: 找到 {len(cand_ptrs)} 个候选")

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
                indices = [cand.get_int_id() - 100000 for cand in cand_ptrs]  # 减去我们之前加的偏移

                self.logger.info(f"Query {query_key}: 检索到的索引 {indices[:5]}... (显示前5个)")
                self.logger.info(
                    f"Query {query_key}: 对应的相关性分数 {cand_corr[:5] if cand_corr else []}... (显示前5个)")

                # 检查是否有有效匹配
                has_valid_match = False
                for j, idx in enumerate(indices):
                    if idx in true_neighbors:
                        has_valid_match = True
                        self.logger.info(f"Query {query_key}: 在第{j + 1}位找到有效匹配 {idx}")
                        if j == 0:
                            similarity = cand_corr[0] if cand_corr else 0.0
                            top1_similarity_score.append(similarity)
                            self.logger.info(f"Query {query_key}: Top1匹配相关性分数: {similarity:.6f}")
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
                    self.logger.info(f"Query {query_key}: 未找到有效匹配")

                # 计算top 1% recall
                top_percent_indices = indices[:threshold]
                if len(set(top_percent_indices).intersection(set(true_neighbors))) > 0:
                    one_percent_retrieved += 1

            except Exception as e:
                self.logger.error(f"查询处理错误 {query_key}: {e}")
                import traceback
                self.logger.error(f"错误详情: {traceback.format_exc()}")
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
        self.logger.info(f"\n=== Chilean数据集 ContourContext 评估结果 ===")
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

        output_file = os.path.join(results_dir, "contourcontext_results_chilean_不旋转_相同时段_session.txt")

        with open(output_file, "w") as f:
            f.write("Chilean Dataset ContourContext Evaluation Results\n")
            f.write("=" * 60 + "\n\n")
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

    # 在 ChileanContourContextEvaluator 类中添加方法
    def debug_key_consistency(self, query_idx: int):
        """调试检索键一致性"""
        query_set = self.query_sets[0]
        database_set = self.database_sets[0]

        # 找到查询和数据库中的相同文件
        query_item = query_set[query_idx]
        query_file = query_item['query']

        # 在数据库中找到相同的文件
        db_match_key = None
        for db_key, db_item in database_set.items():
            if db_item['query'] == query_file:
                db_match_key = db_key
                break

        if db_match_key is not None:
            print(f"\n[CONSISTENCY_CHECK] Found matching file:")
            print(f"  Query key: {query_idx}, DB key: {db_match_key}")
            print(f"  File: {query_file}")
            print(f"  Expected: Same point cloud should generate identical keys")

    def debug_pointcloud_consistency(self):
        """验证点云一致性"""
        # 找到第一个查询和对应的数据库条目
        query_item = self.query_sets[0][0]
        database_item = self.database_sets[0][0]

        print(f"\n[POINTCLOUD_CHECK]")
        print(f"Query file: {query_item['query']}")
        print(f"Database file: {database_item['query']}")
        print(f"Files identical: {query_item['query'] == database_item['query']}")

        # 加载并比较点云
        query_pc = self.load_chilean_pointcloud(query_item['query'])
        db_pc = self.load_chilean_pointcloud(database_item['query'])

        print(f"Query pointcloud shape: {query_pc.shape}")
        print(f"Database pointcloud shape: {db_pc.shape}")

        if query_pc.shape == db_pc.shape:
            print(f"Pointclouds identical: {np.array_equal(query_pc, db_pc)}")
            if not np.array_equal(query_pc, db_pc):
                print(f"Max difference: {np.max(np.abs(query_pc - db_pc))}")


def main():
    """主函数"""
    # ========== 配置参数 - 在PyCharm中直接修改这里 ==========
    DATASET_FOLDER = '/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times'  # 修改为你的Chilean数据集路径
    DATABASE_FILE = 'chilean_NoRot_NoScale_5cm_evaluation_database_100_100.pickle'  # 数据库pickle文件
    QUERY_FILE = 'chilean_NoRot_NoScale_5cm_evaluation_query_100_100.pickle'  # 查询pickle文件
    LOG_FILE = 'contourcontext_chilean_不旋转_相同时段_log.txt'  # 日志文件
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