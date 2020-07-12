import csv
import datetime
import shutil
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from path import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix

import models
from config.args_train_scannet import *
from datasets.scannet_loader import ScanNetDataset
from utils.utils import AverageMeter, save_path_formatter

warnings.filterwarnings('ignore')
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda") if args.cuda else torch.device("cpu")

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

print("1. Path to save the output.")
save_path = Path(save_path_formatter(args))
args.save_path = 'checkpoints' / save_path
args.save_path.makedirs_p()
print("=> will save everything to {}".format(args.save_path))

print("2.Data Loading...")

train_set = ScanNetDataset(args.data_path, sub_grid_size=args.sub_grid_size, train=True)
val_set = ScanNetDataset(args.data_path, sub_grid_size=args.sub_grid_size, train=False)

print('{} clouds found train scenes'.format(len(train_set.trees['train'])))
print('{} clouds found valid scenes'.format(len(val_set.trees['valid'])))

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                          pin_memory=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

print("3.Creating Model")
shutil.copyfile('models/randla_net_scannet.py', args.save_path / 'randla_net_scannet.py')
shutil.copyfile('config/args_train_scannet.py', args.save_path / 'args_train_scannet.py')
randla_net = models.RandLA_Net_ScanNet(args).to(device)
if args.pretrained:
    print('=> using pre-trained weights for RandLa-Net')
    weights = torch.load(args.pretrained)
    randla_net.load_state_dict(weights['state_dict'], strict=False)
else:
    randla_net.init_weights()

randla_net = torch.nn.DataParallel(randla_net)

print("4. Setting Optimization Solver")
optimizer = torch.optim.Adam(randla_net.parameters(), lr=args.lr, betas=(args.momentum, args.beta),
                             weight_decay=args.weight_decay)


def poly_scheduler(epoch, base_num=args.base_num):
    return base_num ** epoch


exp_lr_scheduler_R = lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_scheduler)

print("5. Start Tensorboard ")
# tensorboard --logdir=/path_to_log_dir/ --port 6006
training_writer = SummaryWriter(args.save_path)

print("6. Create csvfile to save log information")

with open(args.save_path / args.log_file, 'w') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter='\t')
    csv_writer.writerow(['Train_loss', 'Train_Accuracy', 'Train_MIoU', 'Valid_accuracy', 'Valid_MIoU'])

print("7. Start Training!")


def main():
    best_error = -1
    for epoch in range(args.epochs):
        start_time = time.time()
        losses, loss_names = train(epoch, randla_net, optimizer)
        errors, error_names = validate(randla_net)

        decisive_error = errors[1]
        if best_error < 0:
            best_error = decisive_error

        is_best = decisive_error > best_error
        best_error = max(best_error, decisive_error)

        torch.save({
            'epoch': epoch + 1,
            'state_dict': randla_net.state_dict()
        }, args.save_path / 'randla_net_seg_{}.pth.tar'.format(epoch))

        if is_best:
            shutil.copyfile(args.save_path / 'randla_net_seg_{}.pth.tar'.format(epoch),
                            args.save_path / 'randla_net_seg_best.pth.tar')

        for loss, name in zip(losses, loss_names):
            training_writer.add_scalar(name, loss, epoch)
            training_writer.flush()
        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)
            training_writer.flush()

        with open(args.save_path / args.log_file, 'a') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='\t')
            csv_writer.writerow([losses[0], losses[1], losses[2], errors[0], errors[1]])

        print("\n---- [Epoch {}/{}] ----".format(epoch, args.epochs))
        print("Train---Loss:{}, Accuracy:{}, MIoU:{}".format(losses[0], losses[1], losses[2]))
        print("Valid---Accuracy:{}, MIoU:{}".format(errors[0], errors[1]))

        epoch_left = args.epochs - (epoch + 1)
        time_left = datetime.timedelta(seconds=epoch_left * (time.time() - start_time))
        print("----ETA {}".format(time_left))


def train(epoch, randla_net, optimizer):
    loss_names = ['total_loss', 'train_accuracy', 'train_miou']
    randla_net.train()
    exp_lr_scheduler_R.step(epoch=epoch)
    gt_classes = [0 for _ in range(args.num_cls)]
    positive_classes = [0 for _ in range(args.num_cls)]
    true_positive_classes = [0 for _ in range(args.num_cls)]
    val_total_correct = 0
    val_total_seen = 0
    total_loss = 0
    total_samples = 0

    # points: B,num_layers,N,3   neighbors: B,num_layers,N,num_knn   pools: B, num_layers, N', num_knn
    # up_samples: B, num_layers, N, num_knn, inputs: B, 6, N   labels: B,N
    for i, (points, neighbors, pools, up_samples, inputs, labels) in enumerate(train_loader):
        points = points.to(device).float()
        neighbors = neighbors.to(device).long()
        pools = pools.to(device).long()
        up_samples = up_samples.to(device).long()
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()

        outputs = randla_net(points, neighbors, pools, up_samples, inputs)  # B,num_cls,n_pts

        outputs = outputs.permute(0, 2, 1).contiguous()
        outputs = outputs.view(-1, args.num_cls)
        labels = labels.view(-1)
        # Ignore the invalid point
        ignored_bool = torch.zeros_like(labels, dtype=torch.bool).to(device)
        ignored_label_inds = np.array([0])
        for ing_label in ignored_label_inds:
            ignored_bool = ignored_bool + torch.eq(labels, ing_label)
        valid_idx = torch.squeeze(torch.nonzero(torch.logical_not(ignored_bool)))

        valid_logits = outputs[valid_idx]
        valid_labels_init = labels[valid_idx]

        reducing_list = torch.arange(0, args.num_cls, dtype=torch.long).to(device)
        inserted_value = torch.zeros((1,), dtype=torch.long).to(device)
        for ing_label in ignored_label_inds:
            reducing_list = torch.cat([reducing_list[:ing_label], inserted_value, reducing_list[ing_label:]], 0)
        valid_labels = reducing_list[valid_labels_init]

        loss = F.cross_entropy(valid_logits, valid_labels)

        pred_val = torch.argmax(valid_logits, 1)  # （n_pts,）
        correct = torch.sum(pred_val == valid_labels)
        valid_labels_cpu = valid_labels.cpu()
        pred_val_cpu = pred_val.cpu()

        val_total_correct += correct.item()  # 有多少个点分类正确
        val_total_seen += len(valid_labels_cpu)  # 总共有多少个点
        conf_matrix = confusion_matrix(valid_labels_cpu, pred_val_cpu, np.arange(0, args.num_cls, 1))
        positive_classes += np.sum(conf_matrix, axis=0)
        gt_classes += np.sum(conf_matrix, axis=1)
        true_positive_classes += np.diagonal(conf_matrix)

        total_loss += loss.item() * args.batch_size
        total_samples += args.batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    accuracy = val_total_correct / val_total_seen
    iou_list = []
    for n in range(0, args.num_cls, 1):
        iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
        iou = np.nan_to_num(iou)
        iou_list.append(iou)
    mean_iou = sum(iou_list) / float(args.num_cls)
    losses = [total_loss / total_samples, accuracy, mean_iou]
    return losses, loss_names


@torch.no_grad()
def validate(randla_net):
    error_names = ['train_accuracy', 'val_miou']
    randla_net.eval()

    gt_classes = [0 for _ in range(args.num_cls)]
    positive_classes = [0 for _ in range(args.num_cls)]
    true_positive_classes = [0 for _ in range(args.num_cls)]
    val_total_correct = 0
    val_total_seen = 0

    # points: (B,9,n_pts）; label:（B,n_pts); targets:(B,13,n_pts)
    for i, (points, neighbors, pools, up_samples, inputs, labels) in enumerate(val_loader):
        points = points.to(device).float()
        neighbors = neighbors.to(device).long()
        pools = pools.to(device).long()
        up_samples = up_samples.to(device).long()
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()

        outputs = randla_net(points, neighbors, pools, up_samples, inputs)  # B,num_cls,n_pts
        # Ignore the invalid point
        outputs = outputs.permute(0, 2, 1).contiguous()
        outputs = outputs.view(-1, args.num_cls)
        labels = labels.view(-1)
        ignored_label_inds = np.array([0])
        ignored_bool = torch.zeros_like(labels, dtype=torch.bool)
        for ing_label in ignored_label_inds:
            ignored_bool = ignored_bool + torch.eq(labels, ing_label)
        valid_idx = torch.squeeze(torch.nonzero(torch.logical_not(ignored_bool)))

        valid_logits = outputs[valid_idx]
        valid_labels_init = labels[valid_idx]

        reducing_list = torch.arange(0, args.num_cls, dtype=torch.long).to(device)
        inserted_value = torch.zeros((1,), dtype=torch.long).to(device)
        for ing_label in ignored_label_inds:
            reducing_list = torch.cat([reducing_list[:ing_label], inserted_value, reducing_list[ing_label:]], 0)
        valid_labels = reducing_list[valid_labels_init]

        pred_val = torch.argmax(valid_logits, 1)  # （n_pts,）
        correct = torch.sum(pred_val == valid_labels)

        valid_labels_cpu = valid_labels.cpu()
        pred_val_cpu = pred_val.cpu()

        val_total_correct += correct.item()  # 有多少个点分类正确
        val_total_seen += len(valid_labels_cpu)  # 总共有多少个点

        conf_matrix = confusion_matrix(valid_labels_cpu, pred_val_cpu, np.arange(0, args.num_cls, 1))

        positive_classes += np.sum(conf_matrix, axis=0)
        gt_classes += np.sum(conf_matrix, axis=1)
        true_positive_classes += np.diagonal(conf_matrix)

    accuracy = val_total_correct / val_total_seen
    iou_list = []
    for n in range(0, args.num_cls, 1):
        iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
        iou = np.nan_to_num(iou)
        iou_list.append(iou)
    mean_iou = sum(iou_list) / float(args.num_cls)
    errors = [accuracy, mean_iou]
    return errors, error_names


if __name__ == '__main__':
    main()
