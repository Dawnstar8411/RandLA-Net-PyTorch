import argparse

parser = argparse.ArgumentParser(description="RandLA-Net Visualization",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 模型信息
parser.add_argument('--dataset_name', type=str, default="ScanNet")
parser.add_argument('--model_name', type=str, default="RandLA-Net")
parser.add_argument('--seed', default=2048, type=int, help="seed for random function and network initialization.")

parser.add_argument('log_path', default='', metavar='DIR', help='')

parser.add_argument('--relu', default=0, type=int, help='Which feature is visualized')
parser.add_argument('--snapshot', default=-1, type=int, help='')
parser.add_argument('--deformation', default=0, type=int, help='')
parser.add_argument('--compute_activations', default=True, type=bool, help="")

parser.add_argument('--no_cuda', default=False, type=bool)
parser.add_argument('--workers', '-j', default=8, type=int, metavar='N', help="number of data loading workers")
parser.add_argument('--batch_size', default=9, type=int, help="Batch Size during training")

parser.add_argument('--in_radius',default=4, type = int, help="")
parser.add_argument('--first_subsampling_dl',default= 0.02,type = int, help="")


# 读取与保存


parser.add_argument('--data_path', default='/home/yc/chen/data/segmentation/scannet/', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--pretrained', default=None, metavar='PATH', help="path to pre-trained model.")
parser.add_argument('--log_summary', default='progress_log_summary.csv', metavar='PATH')
parser.add_argument('--log_full', default='progress_log_full.csv', metavar='PATH')

# 网络训练


parser.add_argument('--epochs', default=300, type=int, metavar='N', help="number of total epochs to run")

parser.add_argument('--epoch_size', default=500, type=int, metavar='N', help="manual epoch size")
parser.add_argument('--lr', default=0.01, type=float, metavar='LR', help="initial learning rate")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help="momentum for sgd, alpha for adam")
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight_decay', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--decay_step', default=50, type=int, help="Decay step for lr decay")
parser.add_argument('--decay_rate', default=0.7, type=float, help="Decay rate for lr decay")
parser.add_argument('--base_num', default=0.95, type=float, help="parameter for lambdaLR")
# 模型超参（网络结构与损失函数）


# 具体算法相关 （待定）
parser.add_argument('--n_pts', default=40960, type=int, help='Point Number')
parser.add_argument('--num_cls', default=20, type=int, help='class number')
parser.add_argument('--num_knn', default=16, type=int, help='number of k nearest points')
parser.add_argument('--num_layers', default=5, type=int, help='Number of layers')
parser.add_argument('--sub_grid_size', default=0.04, type=float, help='preprocess parameter.')
parser.add_argument('--noise_init', default=3.5, type=float, help='noise initial parameter')
parser.add_argument('--sub_sampling_ratio', default=[4, 4, 4, 4, 2],
                    help='Sampling Ratio of random sampling at each layer')
parser.add_argument('--sub_sampling_index', default=[4, 16, 64, 128, 256])
parser.add_argument('--d_out', default=[16, 64, 128, 256, 512], help='feature dimension')

# 是否为debug模式
parser.add_argument('--is_debug', type=bool, default=True)

args = parser.parse_args()
