import glob
import os
import pickle

import numpy as np
import pandas
from sklearn.neighbors import KDTree

from utils.helper_data_processing import DataProcessing as DP
from utils.helper_ply_io import write_ply

dataset_path = '/home/yc/chen/data/segmentation/s3dis/Stanford3dDataset_v1.2_Aligned_Version'
anno_paths = [os.path.join(dataset_path, line.rstrip()) for line in open('../utils/meta/s3dis_anno_paths.txt')]
cloud_names = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']

label_to_names = {0: 'ceiling',
                  1: 'floor',
                  2: 'wall',
                  3: 'beam',
                  4: 'column',
                  5: 'window',
                  6: 'door',
                  7: 'table',
                  8: 'chair',
                  9: 'sofa',
                  10: 'bookcase',
                  11: 'board',
                  12: 'clutter'}

# Initiate a bunch of variables converning class labels
label_values = np.sort([key for key, value in label_to_names.items()])  # 0,1,2,3,...
label_names = [label_to_names[key] for key in label_values]  # ceiling, floor, wall, beam...
# 如果label_to_names中的key值不连续，这一步可以让label的值变得连续
label_to_idx = {l: i for i, l in enumerate(label_values)}  # '0':0, '1':1, '2':2 ...
name_to_idx = {name: i for i, name in enumerate(label_names)}

ignored_labels = np.array([])
ignored_label_inds = [label_to_idx[ign_label] for ign_label in ignored_labels]

# Create dump dirs
sub_grid_size = 0.04
original_pc_folder = os.path.join(os.path.dirname(dataset_path), 'original_ply')
sub_pc_folder = os.path.join(os.path.dirname(dataset_path), 'input_{:.3f}').format(sub_grid_size)
os.makedirs(original_pc_folder) if not os.path.exists(original_pc_folder) else None
os.makedirs(sub_pc_folder) if not os.path.exists(sub_pc_folder) else None

for cloud_name in cloud_names:
    cloud_file = os.path.join(original_pc_folder, cloud_name + '.ply')
    if os.path.exists(cloud_file):
        continue

    cloud_folder = os.path.join(dataset_path, cloud_name)
    room_folders = [os.path.join(cloud_folder, room) for room in os.listdir(cloud_folder) if
                    os.path.isdir(os.path.join(cloud_folder, room))]

    data_list = []
    for i, room_folder in enumerate(room_folders):
        print('Cloud %s - Room %d/%d' % (cloud_name, i + 1, len(room_folders)))
        for f in glob.glob(os.path.join(room_folder, 'Annotations', '*.txt')):
            class_name = os.path.basename(f).split('_')[0]
            if class_name not in label_names:
                class_name = 'clutter'
            pc = pandas.read_csv(f, header=None, delim_whitespace=True).values
            labels = np.ones((pc.shape[0], 1)) * name_to_idx[class_name]
            data_list.append(np.concatenate([pc, labels], 1))  # N*7
    pc_label = np.concatenate(data_list, 0)
    xyz_min = np.amin(pc_label, axis=0)[0:3]
    pc_label[:, 0:3] -= xyz_min
    xyz = pc_label[:, :3].astype(np.float32)
    colors = pc_label[:, 3:6].astype(np.uint8)
    labels = pc_label[:, 6].astype(np.uint8)
    write_ply(cloud_file, (xyz, colors, labels), ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, sub_grid_size)
    sub_colors = sub_colors / 255.0
    sub_ply_file = os.path.join(sub_pc_folder, cloud_name + '.ply')
    write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])

    search_tree = KDTree(sub_xyz,leaf_size=50)
    kd_tree_file = os.path.join(sub_pc_folder, cloud_name + '_KDTree.pkl')
    with open(kd_tree_file, 'wb') as f:
        pickle.dump(search_tree, f)

    proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
    proj_idx = proj_idx.astype(np.int32)
    proj_save = os.path.join(sub_pc_folder, cloud_name + '_proj.pkl')
    with open(proj_save, 'wb') as f:
        pickle.dump([proj_idx, labels], f)
