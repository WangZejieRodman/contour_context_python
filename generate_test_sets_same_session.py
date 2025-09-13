import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KDTree
import pickle
import random


def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


def construct_same_session_test_sets(base_path, runs_folder, test_session,
                                     pointcloud_fols, filename, output_name,
                                     query_ratio=0.4):
    """
    在同一个session内构建数据库集和查询集进行测试

    Args:
        test_session: 测试用的session号（如'100'）
        query_ratio: 查询集占总数据的比例（默认0.4，即40%作为查询，100%作为数据库）
    """

    print(f"Testing within session {test_session}")
    print(f"Query ratio: {query_ratio} (查询集占比)")
    print(f"Database: 100% of session {test_session}")
    print(f"Query: {query_ratio * 100}% randomly selected from session {test_session}")

    # 读取session数据
    csv_path = os.path.join(base_path, runs_folder, test_session, filename)
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return None, None

    folder_path = os.path.join(base_path, runs_folder, test_session, pointcloud_fols.strip('/'))
    if not os.path.exists(folder_path):
        print(f"Error: Pointcloud folder not found: {folder_path}")
        return None, None

    df_locations = pd.read_csv(csv_path, sep=',')
    df_locations['timestamp'] = runs_folder + test_session + pointcloud_fols + df_locations['timestamp'].astype(
        str) + '.bin'
    df_locations = df_locations.rename(columns={'timestamp': 'file'})

    # 筛选存在的文件
    valid_rows = []
    for index, row in df_locations.iterrows():
        full_path = os.path.join(base_path, row['file'])
        if os.path.exists(full_path):
            valid_rows.append(row)

    if len(valid_rows) == 0:
        print(f"Error: No valid pointcloud files found in session {test_session}")
        return None, None

    print(f"Found {len(valid_rows)} valid pointcloud files in session {test_session}")

    # 构建完整的数据库集（包含所有文件）
    database = {}
    database_coordinates = []

    for i, row in enumerate(valid_rows):
        database[i] = {
            'query': row['file'],
            'northing': row['northing'],
            'easting': row['easting']
        }
        database_coordinates.append([row['northing'], row['easting']])

    database_coordinates = np.array(database_coordinates)
    database_tree = KDTree(database_coordinates)

    # 随机选择查询集
    query_indices = random.sample(range(len(valid_rows)), int(len(valid_rows) * query_ratio))
    query_indices.sort()  # 排序便于调试

    print(f"Selected {len(query_indices)} files as queries")
    print(f"Query indices: {query_indices[:10]}..." if len(query_indices) > 10 else f"Query indices: {query_indices}")

    # 构建查询集
    queries = {}
    for i, query_idx in enumerate(query_indices):
        row = valid_rows[query_idx]
        queries[i] = {
            'query': row['file'],
            'northing': row['northing'],
            'easting': row['easting']
        }

    # 计算正样本匹配（在15米范围内）
    print("Computing positive matches within 15m radius...")

    for query_key in queries.keys():
        query_coord = np.array([[queries[query_key]["northing"], queries[query_key]["easting"]]])

        # 在数据库中找到距离15米内的正样本
        positive_indices = database_tree.query_radius(query_coord, r=15)[0].tolist()

        # 将正样本索引存储到查询集中
        # 这里只有一个数据库session，所以索引为0
        queries[query_key][0] = positive_indices

    # 将数据库和查询集放入列表格式（保持与原始代码格式一致）
    database_sets = [database]  # 只有一个session
    query_sets = [queries]  # 只有一个session

    # 输出文件
    database_filename = f'{output_name}_evaluation_database_{test_session}_{test_session}.pickle'
    query_filename = f'{output_name}_evaluation_query_{test_session}_{test_session}.pickle'

    output_to_file(database_sets, database_filename)
    output_to_file(query_sets, query_filename)

    # 验证并统计正样本
    print(f"\n=== 验证同session测试集 ===")
    total_positive_pairs = 0
    self_matches = 0  # 自己匹配自己的情况

    for query_key in queries.keys():
        positive_indices = queries[query_key][0]
        total_positive_pairs += len(positive_indices)

        # 检查是否包含自己（应该包含，因为距离为0）
        query_file = queries[query_key]['query']
        for pos_idx in positive_indices:
            if database[pos_idx]['query'] == query_file:
                self_matches += 1
                break

    print(f"✅ 总正样本对数: {total_positive_pairs}")
    print(f"✅ 自匹配数量: {self_matches}/{len(queries)} ({self_matches / len(queries) * 100:.1f}%)")
    print(f"✅ 平均每个查询的正样本数: {total_positive_pairs / len(queries):.1f}")

    # 距离统计
    distances = []
    for query_key in queries.keys():
        query_coord = np.array([queries[query_key]["northing"], queries[query_key]["easting"]])
        positive_indices = queries[query_key][0]

        for pos_idx in positive_indices:
            db_coord = np.array([database[pos_idx]["northing"], database[pos_idx]["easting"]])
            dist = np.linalg.norm(query_coord - db_coord)
            distances.append(dist)

    if distances:
        print(f"✅ 正样本距离统计: min={min(distances):.2f}m, max={max(distances):.2f}m, avg={np.mean(distances):.2f}m")

    print(f"\n=== Generated Files ===")
    print(f"Database: {database_filename}")
    print(f"Query: {query_filename}")
    print(f"\n=== Test Setup ===")
    print(f"这是同session内部测试，预期应该有很高的召回率")
    print(f"如果召回率仍然很低，说明BTC参数设置有问题")
    print(f"如果召回率正常，说明跨session的环境变化过大")

    # ✅ 添加验证逻辑
    print(f"\n=== 验证数据一致性 ===")

    # 检查第一个查询的匹配
    first_query = queries[0]
    query_file = first_query['query']
    query_coord = [first_query['northing'], first_query['easting']]
    ground_truth_indices = first_query[0]

    print(f"查询文件: {query_file}")
    print(f"查询坐标: {query_coord}")
    print(f"Ground truth数据库索引: {ground_truth_indices}")

    # 检查这些索引对应的数据库文件
    print(f"对应的数据库文件:")
    for idx in ground_truth_indices[:3]:  # 只看前3个
        if idx < len(database):
            db_file = database[idx]['query']
            db_coord = [database[idx]['northing'], database[idx]['easting']]
            dist = np.linalg.norm(np.array(query_coord) - np.array(db_coord))
            print(f"  DB[{idx}]: {db_file}, coord={db_coord}, dist={dist:.2f}m")

            # 检查是否是同一个文件
            if query_file == db_file:
                print(f"    ✅ 文件匹配：相同文件")
            else:
                print(f"    ❌ 文件不匹配：不同文件")

    return database_sets, query_sets


# 主执行部分
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    base_path = "/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/"

    runs_folder = "chilean_NoRot_NoScale_5cm/"
    path = os.path.join(base_path, runs_folder)
    print(f"Base path: {path}")

    if not os.path.exists(path):
        print(f"Error: Base path {path} does not exist!")
        exit(1)

    # 测试参数
    TEST_SESSION = "100"  # 使用session 100进行测试
    QUERY_RATIO = 0.4  # 40%的文件作为查询
    pointcloud_fols = "/pointcloud_20m_10overlap/"

    print(f"\n=== Same Session Test Setup ===")
    print(f"Test session: {TEST_SESSION}")
    print(f"Query ratio: {QUERY_RATIO}")
    print(f"Strategy: 方法3 - 空间重叠子集测试")

    # 设置随机种子以确保结果可重复
    random.seed(42)

    # 执行同session测试集构建
    database_sets, query_sets = construct_same_session_test_sets(
        base_path,
        runs_folder,
        TEST_SESSION,
        pointcloud_fols,
        "pointcloud_locations_20m_10overlap.csv",
        "chilean_NoRot_NoScale_5cm"
    )

    if database_sets is not None:
        print(f"\n=== Final Summary ===")
        print(f"Database entries: {len(database_sets[0])}")
        print(f"Query entries: {len(query_sets[0])}")
        print(f"\n现在可以运行 btc_chilean_场景识别.py 进行测试")
        print(f"预期结果：如果BTC算法和参数正确，召回率应该很高（>50%）")
    else:
        print("❌ 测试集构建失败")