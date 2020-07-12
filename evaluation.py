import os
from utils.helper_ply import write_ply
from sklearn.metrics import confusion_matrix
from utils.helper_tool import DataProcessing as DP

import torch
import numpy as np
import time
from utils.visualization import Plot

import csv
import warnings
from path import Path
import h5py
import numpy as np
import scipy.misc
import torch
from path import Path

import models
from config.args_test_s3dis import *
from utils.utils import save_path_formatter

warnings.filterwarnings('ignore')
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda") if args.cuda else torch.device("cpu")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("1. Path to save the output.")
save_path = Path(save_path_formatter(args))
args.save_path = args.output_path / save_path
args.save_path.makedirs_p()
print("=> will save everything to {}".format(args.save_path))



print("2.Data Loading...")

room_file_path = Path(args.data_path) / 'room_filelist.txt'

shape_names_path = Path(args.data_path) / 'shape_names.txt'
with open(shape_names_path, "r") as f:
    shape_names_list = [line.strip() for line in f.readlines()]

test_list_path = Path(args.data_path) / 'test_files.txt'
with open(test_list_path, "r") as f:
    test_list = [line.strip() for line in f.readlines()]

test_points = []
test_label = []

for i in np.arange(len(test_list)):
    h5_filename = test_list[i]
    f = h5py.File(h5_filename)
    test_points.extend(f['data'][:])
    test_label.extend(f['label'][:])

test_points = np.array(test_points)
test_label = np.array(test_label)

n_pts = args.n_pts

print("3.Creating Model")

pointnet_cls = models.PointNet_cls(K=13).to(device)

if args.pretrained:
    print('=> using pre-trained weights for PoseNet')
    weights = torch.load(args.pretrained)
    pointnet_cls.load_state_dict(weights['state_dict'], strict=False)

print("4. Create csvfile to save log information")

with open(args.save_path / args.log_full, 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter='\t')
    csv_writer.writerow(['Pointnet evaluation'])

print("5. Start Testing!")


@torch.no_grad()
def main():
    pointnet_cls.eval()
    total_correct = 0
    total_num = 0
    total_num_class = [0 for _ in range(args.num_classes)]
    total_correct_class = [0 for _ in range(args.num_classes)]
    error_cnt = 0
    for index in range(len(test_points)):
        origin_points = test_points[index]
        points = np.transpose(origin_points, (1, 0))
        points = np.expand_dims(points, 0)
        label = test_label[index]
        points = torch.from_numpy(points)
        label = torch.from_numpy(label)
        points, label = points.to(device).float(), label.to(device).long()

        targets, _ = pointnet_cls(points)

        pred_val = torch.argmax(targets, 1)
        correct = torch.sum(pred_val == label)

        total_correct += correct.item()
        total_num += 1

        l = label[0]
        total_num_class[l] += 1
        total_correct_class[l] += torch.sum(pred_val == label)

        if pred_val != l:
            img_filename = args.save_path / '{}_label_{}_pred_{}.jpg'.format(error_cnt, shape_names_list[l],
                                                                             shape_names_list[pred_val])
            output_img = pc_util.point_cloud_three_views(origin_points)
            scipy.misc.imsave(img_filename, output_img)
            error_cnt += 1

    avg_precision = total_correct / total_num
    print('Average_precision:{}'.format(avg_precision))
    with open(args.save_path / args.log_full, 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='\t')
        csv_writer.writerow(['Average_precision:{}'.format(avg_precision)])

    class_accuracies = np.array(total_correct_class) / np.array(total_num_class, dtype=np.float)
    for i, name in enumerate(shape_names_list):
        print('{}: {}'.format(name, class_accuracies[i]))
        with open(args.save_path / args.log_full, 'w') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t')
            csv_writer.writerow(['{}:{}'.format(name, class_accuracies[i])])


if __name__ == '__main__':
    main()




from helper_tool import Plot

##################
# Visualize data #
##################

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(dataset.train_init_op)
    while True:
        flat_inputs = sess.run(dataset.flat_inputs)
        pc_xyz = flat_inputs[0]
        sub_pc_xyz = flat_inputs[1]
        labels = flat_inputs[21]
        Plot.draw_pc_sem_ins(pc_xyz[0, :, :], labels[0, :])
        Plot.draw_pc_sem_ins(sub_pc_xyz[0, :, :], labels[0, 0:np.shape(sub_pc_xyz)[1]])

