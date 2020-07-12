import argparse

parser = argparse.ArgumentParser(description="RandLA-Net for 3D Point Cloud Segmentation",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 数据，模型权重，训练日志的读取与保存
parser.add_argument('data_path', metavar='DIR', help='Path to dataset')
parser.add_argument('--dataset_name', default="s3dis", type=str,
                    choices=['s3dis', 'scannet', 'semantic3d', 'shapenet', 'modelnet40', 'semantic_kitti'])
parser.add_argument('--model_name', type=str, default="RandLA-Net")
parser.add_argument('--task', default="segmentation", typ=str, choices=['segmentation', 'classification'])
parser.add_argument('--part_task', default='multi', type=str)  # part segmentation的类型，全部都参与分类，还是只对某一个类进行分割
parser.add_argument('--seed', default=2048, type=int, help="seed for random function and network initialization.")
parser.add_argument('--pretrained', default=None, metavar='PATH', help="path to pre-trained model.")
parser.add_argument('--log_file', default='progress_log.csv', metavar='PATH')

# 网络训练
parser.add_argument('--no_cuda', default=False, type=bool)
parser.add_argument('--workers', '-j', default=1, type=int, metavar='N', help="number of data loading workers")
parser.add_argument('--epochs', default=50, type=int, metavar='N', help="number of total epochs to run")
parser.add_argument('--batch_size', default=9, type=int, help="Batch Size during training")
parser.add_argument('--train_steps', default=1000, type=int, metavar='N', help="Number of batches for training")
parser.add_argument('--valid_steps', default=100, type=int, metavar='N', help="Mumber of batches for validation")
parser.add_argument('--snapshot_gap', default=50, type=int, help="number of epoch between each snapshot")

# 网络优化器
parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'], type=str)
parser.add_argument('--lr', default=0.01, type=float, metavar='LR', help="initial learning rate")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help="momentum for sgd, alpha for adam")
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight_decay', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--decay_style', default='LambdaLR', choices=['LambdaLR', 'StepLR'])
parser.add_argument('--decay_basenum', default=0.95, type=float, help="parameter for lambdaLR")
parser.add_argument('--decay_step', default=50, type=int, help="Decay step for StepLR")
parser.add_argument('--decay_rate', default=0.7, type=float, help="Decay rate for StepLR")
parser.add_argument('--base_num', default=0.95, type=float, help="parameter for lambdaLR")

# 网络结构
parser.add_argument('--sub_sampling_ratio', default=[4, 4, 4, 4, 2],
                    help='Sampling Ratio of random sampling at each layer')
parser.add_argument('--d_out', default=[16, 64, 128, 256, 512], help='feature dimension')
parser.add_argument('--num_layers', default=5, type=int, help='Number of layers')

# 损失函数



# 数据准备
parser.add_argument('--n_pts', default=40960, type=int, help='Point Number')
parser.add_argument('--num_cls', default=13, type=int, help='class number')
parser.add_argument('--num_knn', default=16, type=int, help='number of k nearest points')
parser.add_argument('--sub_grid_size', default=0.04, type=float, help='preprocess parameter.')
parser.add_argument('--noise_init', default=3.5, type=float, help='noise initial parameter')

# 具体数据库

# S3DIS
parser.add_argument('--val_split', default=5, type=int, help='Which area to use for test, [default:5]')

# 是否为debug模式
parser.add_argument('--is_debug', type=bool, default=False)

args = parser.parse_args()
