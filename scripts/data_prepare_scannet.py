import json
import os
import pickle

import numpy as np
from sklearn.neighbors import KDTree

from utils.helper_ply_io import write_ply, read_ply
from utils.helper_data_processing import DataProcessing as DP

label_to_names = {0: 'unclassified',
                  1: 'wall',
                  2: 'floor',
                  3: 'cabinet',
                  4: 'bed',
                  5: 'chair',
                  6: 'sofa',
                  7: 'table',
                  8: 'door',
                  9: 'window',
                  10: 'bookshelf',
                  11: 'picture',
                  12: 'counter',
                  14: 'desk',
                  16: 'curtain',
                  24: 'refridgerator',
                  28: 'shower curtain',
                  33: 'toilet',
                  34: 'sink',
                  36: 'bathtub',
                  39: 'otherfurniture'}

# Initiate a bunch of variables converning class labels
label_values = np.sort([key for key, values in label_to_names.items()])  # 0,1,2,3,...
label_names = [label_to_names[key] for key in label_values]  # unclassified, wall, floor, cabinet...
# 如果label_to_names中的key值不连续，这一步可以让label的值变得连续
label_to_idx = {l: i for i, l in enumerate(label_values)}  # '0':0, '1':1, ...'14':13,'16':14
name_to_idx = {name: i for i, name in enumerate(label_names)}

ignored_labels = np.array([0])

ignored_label_inds = [label_to_idx[ign_label] for ign_label in ignored_labels]

sub_grid_size = 0.04

data_path = '/home/yc/chen/data/segmentation/scannet'

raw_train_path = os.path.join(data_path, 'scans')
raw_test_path = os.path.join(data_path, 'scans_test')

mesh_train_path = os.path.join(data_path, 'raw_meshes_train')
mesh_test_path = os.path.join(data_path, 'raw_meshes_test')

original_train_path = os.path.join(data_path, 'original_train')
original_test_path = os.path.join(data_path, 'original_test')

input_train_path = os.path.join(data_path, 'input_{:.3f}_train').format(sub_grid_size)
input_test_path = os.path.join(data_path, 'input_{:.3f}_test').format(sub_grid_size)

raw_paths = [raw_train_path, raw_test_path]
mesh_paths = [mesh_train_path, mesh_test_path]
original_paths = [original_train_path, original_test_path]
input_paths = [input_train_path, input_test_path]

# Mapping from annotation names to NYU labels ID

# Annotation files
anno_files = os.path.join(data_path, 'scannetv2-labels.combined.tsv')

with open(anno_files, 'r') as f:
    lines = f.readlines()
    names1 = [line.split('\t')[1] for line in lines[1:]]
    IDs = [int(line.split('\t')[4]) for line in lines[1:]]
    annot_to_nyuID = {n: id for n, id in zip(names1, IDs)}

for raw_path, original_path, input_path, mesh_path in zip(raw_paths, original_paths, input_paths, mesh_paths):
    if not os.path.exists(mesh_path):
        os.makedirs(mesh_path)
    if not os.path.exists(original_path):
        os.makedirs(original_path)
    if not os.path.exists(input_path):
        os.makedirs(input_path)

    # scene0000_00, scene0000_01
    scenes = np.sort([f for f in os.listdir(raw_path)])
    N = len(scenes)

    for i, scene in enumerate(scenes):
        print(i, scene)
        if os.path.exists(os.path.join(input_path, scene + '.ply')):
            continue

        raw_data, faces = read_ply(os.path.join(raw_path, scene, scene + '_vh_clean_2.ply'), triangular_mesh=True)
        raw_xyz = np.vstack((raw_data['x'], raw_data['y'], raw_data['z'])).T
        raw_colors = np.vstack((raw_data['red'], raw_data['green'], raw_data['blue'])).T
        raw_labels = np.zeros((raw_xyz.shape[0],), dtype=np.int32)

        if original_path == original_train_path:
            # load alignment matrix to realign points
            align_mat = None
            with open(os.path.join(raw_path, scene, scene + '.txt'), 'r') as txtfile:
                lines = txtfile.readlines()
            for line in lines:
                line = line.split()
                if line[0] == 'axisAlignment':
                    align_mat = np.array([float(x) for x in line[2:]]).reshape([4, 4]).astype(np.float32)
            R = align_mat[:3, :3]
            T = align_mat[:3, 3]
            raw_xyz = raw_xyz.dot(R.T) + T

            # get objects segmentations
            with open(os.path.join(raw_path, scene, scene + '_vh_clean_2.0.010000.segs.json'), 'r') as f:
                segmentations = json.load(f)
            segIndices = np.array(segmentations['segIndices'])

            # Get objects classes
            with open(os.path.join(raw_path, scene, scene + '_vh_clean.aggregation.json'), 'r') as f:
                aggregation = json.load(f)

            # Loop on object to classify points
            for segGroup in aggregation['segGroups']:
                c_name = segGroup['label']  # 取出分类的名字
                if c_name in names1:
                    nyuID = annot_to_nyuID[c_name]  # 根据分类名字，将其转换为nyu的类别编号
                    if nyuID in label_values:
                        for segment in segGroup['segments']:
                            raw_labels[segIndices == segment] = nyuID

            write_ply(os.path.join(mesh_path, scene + '_mesh.ply'),
                      [raw_xyz, raw_colors, raw_labels],
                      ['x', 'y', 'z', 'red', 'green', 'blue', 'class'],
                      triangular_faces=faces)
        else:
            write_ply(os.path.join(mesh_path, scene + '_mesh.ply'),
                      [raw_xyz, raw_colors],
                      ['x', 'y', 'z', 'red', 'green', 'blue'],
                      triangular_faces=faces)

        # Create finer point clouds, save as original data
        xyz_min = np.amin(raw_xyz, axis=0)
        finer_xyz = raw_xyz - xyz_min
        finer_xyz, associated_vert_inds = DP.rasterize_mesh(finer_xyz, faces, 0.003)

        # Subsampling points
        if original_path == original_train_path:
            ori_xyz, sub_vert_inds = DP.grid_sub_sampling(points=finer_xyz, labels=associated_vert_inds,
                                                          grid_size=0.01)
            ori_colors = raw_colors[sub_vert_inds.ravel(), :]
            ori_labels = raw_labels[sub_vert_inds.ravel()]

            write_ply(os.path.join(original_path, scene + '.ply'),
                      [ori_xyz, ori_colors, ori_labels, sub_vert_inds],
                      ['x', 'y', 'z', 'red', 'green', 'blue', 'class', 'vert_ind'])

            sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(points=ori_xyz, features=ori_colors,
                                                                   labels=ori_labels, grid_size=sub_grid_size)
            sub_colors = sub_colors / 255.0
            write_ply(os.path.join(input_path, scene + '.ply'),
                      [sub_xyz, sub_colors, sub_labels],
                      ['x', 'y', 'z', 'red', 'green', 'blue', 'class'])
        else:
            ori_xyz, sub_vert_inds = DP.grid_sub_sampling(points=finer_xyz,
                                                          labels=associated_vert_inds,
                                                          grid_size=0.01)
            ori_colors = raw_colors[sub_vert_inds.ravel(), :]
            write_ply(os.path.join(original_path, scene + '.ply'),
                      [ori_xyz, ori_colors, sub_vert_inds],
                      ['x', 'y', 'z', 'red', 'green', 'blue', 'vert_ind'])

            sub_xyz, sub_colors, sub_vert_inds = DP.grid_sub_sampling(points=ori_xyz, features=ori_colors,
                                                                      labels=sub_vert_inds, grid_size=sub_grid_size)
            sub_colors = sub_colors / 255.0
            write_ply(os.path.join(input_path, scene + '.ply'),
                      [sub_xyz, sub_colors, sub_vert_inds],
                      ['x', 'y', 'z', 'red', 'green', 'blue', 'vert_ind'])

        # KDTree File
        search_tree = KDTree(sub_xyz, leaf_size=50)
        kd_tree_file = os.path.join(input_path, scene + '_KDTree.pkl')
        with open(kd_tree_file, 'wb')as f:
            pickle.dump(search_tree, f)

        proj_idx = np.squeeze(search_tree.query(raw_xyz, return_distance=False))
        proj_idx = proj_idx.astype(np.int32)
        proj_save = os.path.join(input_path, scene + '_proj.pkl')
        if input_path == input_test_path:
            raw_labels = np.zeros(proj_idx.shape[0], dtype=np.int32)
        with open(proj_save, 'wb') as f:
            pickle.dump([proj_idx, raw_labels], f)
