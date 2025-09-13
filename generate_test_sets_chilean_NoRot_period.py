import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KDTree
import pickle

# 基于时间/session的划分策略
DATABASE_SESSION_START = 100  # 数据库使用的session范围（历史数据）
DATABASE_SESSION_END = 101
QUERY_SESSION_START = 180  # 查询使用的session范围（当前数据）
QUERY_SESSION_END = 181


def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


def construct_query_and_database_sets_separate(base_path, runs_folder,
                                               database_folders, query_folders,
                                               pointcloud_fols, filename, output_name):
    """
    分别构建数据库集和查询集

    Args:
        database_folders: 数据库session列表 (100-101)
        query_folders: 查询session列表 (180-181)
    """

    print(f"Using  folder: {pointcloud_fols}")
    print(f"Database sessions: {database_folders}")
    print(f"Query sessions: {query_folders}")

    # 第一步：构建数据库集并同时记录有效文件的坐标
    database_sets = []
    database_coordinates_list = []  # 存储每个数据库session的有效坐标

    for folder in database_folders:
        print(f"Processing database session {folder}...")
        database = {}
        valid_coordinates = []  # 当前session的有效坐标列表

        csv_path = os.path.join(base_path, runs_folder, folder, filename)
        if not os.path.exists(csv_path):
            print(f"Skipping database {folder}: CSV file not found")
            database_sets.append(database)
            database_coordinates_list.append(np.array([]).reshape(0, 2))
            continue

        folder_path = os.path.join(base_path, runs_folder, folder, pointcloud_fols.strip('/'))
        if not os.path.exists(folder_path):
            print(f"Skipping database {folder}:  folder not found")
            database_sets.append(database)
            database_coordinates_list.append(np.array([]).reshape(0, 2))
            continue

        df_locations = pd.read_csv(csv_path, sep=',')
        df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + df_locations['timestamp'].astype(
            str) + '.bin'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})

        for index, row in df_locations.iterrows():
            full_path = os.path.join(base_path, row['file'])
            if not os.path.exists(full_path):
                continue

            database[len(database.keys())] = {
                'query': row['file'],
                'northing': row['northing'],
                'easting': row['easting']
            }

            # 同时记录有效坐标，确保索引一致
            valid_coordinates.append([row['northing'], row['easting']])

        database_sets.append(database)
        # 转换为numpy数组，如果没有有效坐标则创建空的(0,2)数组
        if valid_coordinates:
            database_coordinates_list.append(np.array(valid_coordinates))
        else:
            database_coordinates_list.append(np.array([]).reshape(0, 2))

    # 第二步：基于有效坐标构建KDTree
    database_trees = []
    for coords in database_coordinates_list:
        if len(coords) > 0:
            database_tree = KDTree(coords)
        else:
            database_tree = None  # 空的session
        database_trees.append(database_tree)

    # 第三步：构建查询集
    query_sets = []
    for folder in query_folders:
        print(f"Processing query session {folder}...")
        queries = {}

        csv_path = os.path.join(base_path, runs_folder, folder, filename)
        if not os.path.exists(csv_path):
            continue

        folder_path = os.path.join(base_path, runs_folder, folder, pointcloud_fols.strip('/'))
        if not os.path.exists(folder_path):
            continue

        df_locations = pd.read_csv(csv_path, sep=',')
        df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + df_locations['timestamp'].astype(
            str) + '.bin'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})

        for index, row in df_locations.iterrows():
            full_path = os.path.join(base_path, row['file'])
            if not os.path.exists(full_path):
                continue

            queries[len(queries.keys())] = {
                'query': row['file'],
                'northing': row['northing'],
                'easting': row['easting']
            }

        query_sets.append(queries)

    # 第四步：计算正样本匹配（查询集中的每个查询在数据库集中的正样本）
    print("Computing positive matches...")
    for i, (database_tree, database_set) in enumerate(zip(database_trees, database_sets)):
        if database_tree is None or len(database_set) == 0:
            print(f"Skipping empty database session {i}")
            # 为所有查询session的这个数据库session添加空的正样本列表
            for j, query_set in enumerate(query_sets):
                for key in query_set.keys():
                    query_sets[j][key][i] = []
            continue

        for j, query_set in enumerate(query_sets):
            for key in query_set.keys():
                query_coord = np.array([[query_set[key]["northing"], query_set[key]["easting"]]])

                # 在数据库session i中找到距离15米内的正样本
                positive_indices = database_tree.query_radius(query_coord, r=15)[0].tolist()

                # 将正样本索引存储到查询集中
                # 现在这些索引与database_sets[i]的键完全对应
                query_sets[j][key][i] = positive_indices

    # 输出文件
    database_filename = f'{output_name}_evaluation_database_{DATABASE_SESSION_START}_{DATABASE_SESSION_END}.pickle'
    query_filename = f'{output_name}_evaluation_query_{QUERY_SESSION_START}_{QUERY_SESSION_END}.pickle'

    output_to_file(database_sets, database_filename)
    output_to_file(query_sets, query_filename)

    # 验证修复结果
    print(f"\n=== 验证索引一致性 ===")
    total_positive_pairs = 0

    for i, database_set in enumerate(database_sets):
        if len(database_set) == 0:
            continue

        for j, query_set in enumerate(query_sets):
            for key in query_set.keys():
                if i in query_set[key]:
                    positive_indices = query_set[key][i]
                    total_positive_pairs += len(positive_indices)

                    # 验证索引有效性
                    for pos_idx in positive_indices:
                        if pos_idx not in database_set:
                            print(f"❌ 错误：索引 {pos_idx} 不存在于数据库session {i}")
                        else:
                            # 索引有效
                            pass

    print(f"✅ 总正样本对数: {total_positive_pairs}")

    print(f"\n=== Generated Files ===")
    print(f"Database: {database_filename}")
    print(f"Query: {query_filename}")

    return database_sets, query_sets


# 主执行部分
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = "/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/"

runs_folder = "chilean_NoRot_NoScale_5cm/"
path = os.path.join(base_path, runs_folder)
print(f"Base path: {path}")

if not os.path.exists(path):
    print(f"Error: Base path {path} does not exist!")
    exit(1)

all_folders = sorted(os.listdir(path))
print(f"Found {len(all_folders)} total folders")

# 筛选出有效的session folders（假设是数字命名）
valid_folders = []
for folder in all_folders:
    if not folder.startswith('.') and folder.isdigit():
        valid_folders.append(folder)

valid_folders.sort(key=int)  # 按数字大小排序
print(f"Valid session folders: {len(valid_folders)}")

# 划分数据库和查询sessions
database_folders = []  # 存储数据库session folders
query_folders = []  # 存储查询session folders

for folder in valid_folders:
    session_num = int(folder)
    if DATABASE_SESSION_START <= session_num <= DATABASE_SESSION_END:
        database_folders.append(folder)  # 100-101作为数据库
    elif QUERY_SESSION_START <= session_num <= QUERY_SESSION_END:
        query_folders.append(folder)  # 180-181作为查询

print(f"\n=== Session划分 ===")
print(f"数据库sessions ({DATABASE_SESSION_START}-{DATABASE_SESSION_END}): {len(database_folders)} sessions")
print(f"查询sessions ({QUERY_SESSION_START}-{QUERY_SESSION_END}): {len(query_folders)} sessions")

# 使用文件夹
pointcloud_fols = "/pointcloud_20m_10overlap/"

# 执行构建
database_sets, query_sets = construct_query_and_database_sets_separate(
    base_path,
    runs_folder,
    database_folders,  # 数据库用100-101
    query_folders,  # 查询用180-181
    pointcloud_fols,
    "pointcloud_locations_20m_10overlap.csv",
    "chilean_NoRot_NoScale_5cm"
)

print(f"\n=== Final Summary ===")
print(f"Total database sets: {len(database_sets)}")
print(f"Total query sets: {len(query_sets)}")

# 统计总的数据库和查询数量
total_db_entries = sum(len(db_set) for db_set in database_sets)
total_query_entries = sum(len(query_set) for query_set in query_sets)

print(f"Total database entries: {total_db_entries}")
print(f"Total query entries: {total_query_entries}")

print(f"\n=== Application Scenario ===")
print(f"Database (100-101): Historical point clouds serving as reference map")
print(f"Query (180-181): Current observations for localization")
print(f"Task: Find matching locations in historical map for current observations")