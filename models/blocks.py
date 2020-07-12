import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super(Attention_pooling, self).__init__()
        self.fc1 = nn.Linear(d_in, d_in)
        self.bn1 = nn.BatchNorm1d(d_in)

        self.conv1 = nn.Conv2d(d_in, d_out, kernel_size=(1, 1))
        self.bn2 = nn.BatchNorm2d(d_out)

    def forward(self, feature_set):
        # feature_set: [B, d, N,k]
        batch_size = feature_set.size()[0]  # B
        d = feature_set.size()[1]
        num_points = feature_set.size()[2]  # N
        num_neigh = feature_set.size()[3]  # k

        f_reshaped = feature_set.permute(0, 2, 3, 1).contiguous()  # B,N,k,d
        f_reshaped = f_reshaped.view(-1, d)  # B*N*k, d

        att_activation = self.bn1(self.fc1(f_reshaped))  # B*N*k, d
        att_activation = att_activation.reshape(-1, num_neigh, d)  # B*N,k,d
        f_reshaped = f_reshaped.reshape(-1, num_neigh, d)  # B*N,k,d

        att_scores = F.softmax(att_activation, dim=1)  # B*N, k,d

        f_agg = f_reshaped * att_scores  # B*N,k,d

        f_agg = torch.sum(f_agg, dim=1, keepdim=True)  # B*N,1,d
        f_agg = f_agg.reshape(batch_size, num_points, 1, d).permute(0, 3, 1, 2)  # B,d,N,1
        f_agg = F.leaky_relu(self.bn2(self.conv1(f_agg)))  # B, d_out, N, 1
        return f_agg


# d_out [16, 64, 128, 256, 512]
class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super(Dilated_res_block, self).__init__()
        self.conv1 = nn.Conv2d(d_in, d_out // 2, kernel_size=(1, 1), stride=1)  # B,d_in,1,n_pts
        self.bn1 = nn.BatchNorm2d(d_out // 2)

        self.conv2 = nn.Conv2d(10, d_out // 2, kernel_size=(1, 1))
        self.bn2 = nn.BatchNorm2d(d_out // 2)

        self.att_pooling1 = Attention_pooling(d_out, d_out // 2)

        self.conv3 = nn.Conv2d(d_out // 2, d_out // 2, kernel_size=(1, 1))
        self.bn3 = nn.BatchNorm2d(d_out // 2)

        self.att_pooling2 = Attention_pooling(d_out, d_out)

        self.conv4 = nn.Conv2d(d_out, d_out * 2, kernel_size=(1, 1))
        self.bn4 = nn.BatchNorm2d(d_out * 2)

        self.conv5 = nn.Conv2d(d_in, d_out * 2, kernel_size=(1, 1))
        self.bn5 = nn.BatchNorm2d(d_out * 2)

    def forward(self, feature, xyz, neigh_idx):
        # feature: (B,d_in,N,1) xyx:(B,3,N) neigh_idx:(B,N,k)
        batch_size = neigh_idx.size()[0]
        n_pts = neigh_idx.size()[1]
        num_k = neigh_idx.size()[2]

        f_xyz = relative_pos_encoding(xyz, neigh_idx)  # B,10,N,k

        f_pc = F.leaky_relu(self.bn1(self.conv1(feature)))  # B,d_out//2,N,1
        f_xyz = F.leaky_relu(self.bn2(self.conv2(f_xyz)))  # B,d_out//2,N,K

        index_input = neigh_idx.view(batch_size, -1)  # B, N*k
        f_neighbours = batch_gather(torch.squeeze(f_pc, dim=2), index_input).view(batch_size, -1, n_pts,
                                                                                  num_k)  # B,d_out/2,N,k
        f_concat = torch.cat((f_neighbours, f_xyz), dim=1)  # B,d_out,N,K

        f_pc_agg = self.att_pooling1(f_concat)  # B, d_out//2, N,1

        f_xyz = F.leaky_relu(self.bn3(self.conv3(f_xyz)))  # B, d_out//2, N, k
        f_neighbours = batch_gather(torch.squeeze(f_pc_agg, dim=2), index_input).view(batch_size, -1, n_pts,
                                                                                      num_k)  # B,d_out//2,N,k

        f_concat = torch.cat((f_neighbours, f_xyz), dim=1)  # B,d_out,N,k

        f_pc = self.att_pooling2(f_concat)  # B, d_out, N,1

        f_pc = self.bn4(self.conv4(f_pc))  # B, 2* d_out, N,1

        shortcut = self.bn5(self.conv5(feature))  # B, d_out *2, N, 1
        ans = F.leaky_relu(f_pc + shortcut)  # B, d_out * 2, N,1
        return ans


def gather_neighbour(pc, neighbor_idx):
    """
    :param pc: [B,d,N]
    :param neighbor_idx: [B,N,k]
    :return:
    """
    # gather the coordinates or features of neighboring points
    batch_size = neighbor_idx.size()[0]  # B
    index_input = neighbor_idx.view(batch_size, -1)  # B,N*k
    ans = batch_gather(pc, index_input)  # B,d, N*k
    return ans


def relative_pos_encoding(xyz, neigh_idx):
    """
    :param xyz: [B,3,N]
    :param neigh_idx: [B,N,k]
    :return:
    """
    batch_size = neigh_idx.size()[0]  # B
    num_pc = neigh_idx.size()[1]  # N
    num_knn = neigh_idx.size()[2]  # k
    index_input = neigh_idx.view(batch_size, -1)  # B, N*k
    neighbor_xyz = batch_gather(xyz, index_input)  # B,3, N*k
    neighbor_xyz = neighbor_xyz.view(batch_size, 3, num_pc, -1)  # B,3,N,k
    xyz_tile = torch.unsqueeze(xyz, dim=3).repeat(1, 1, 1, num_knn)  # B,3,N,k
    relative_xyz = xyz_tile - neighbor_xyz  # B,3,N,k
    relative_dis = torch.sqrt(torch.sum(relative_xyz ** 2, dim=1, keepdim=True))  # B,1,N,K
    relative_feature = torch.cat((relative_dis, relative_xyz, xyz_tile, neighbor_xyz), dim=1)  # B,10,N,k
    return relative_feature


def random_sample(feature, pool_idx):
    """
    :param feature: [B, d, N, 1] input features matrix
    :param pool_idx: [B, N', k] N'<N, N' is the selected position after pooling
    :return: pool_features = [B, d, N',1] pooled features matrix
    """
    # a squeeze operation in the given dimension
    feature = torch.squeeze(feature, dim=3)  # B,d,N
    batch_size = feature.size()[0]
    d = feature.size()[1]
    num_neigh = pool_idx.size()[2]
    pool_idx = pool_idx.view(batch_size, -1)  # B,n'*k
    pool_features = batch_gather(feature, pool_idx)  # B,d, n' * k
    pool_features = pool_features.view(batch_size, d, -1, num_neigh)  # B,d,n',k
    pool_features = torch.max(pool_features, dim=3, keepdim=True)[0].data  # B, d,n',1

    return pool_features


def nearest_interpolation(feature, interp_idx):
    """
    :param feature: [B,d,N,1] input features matrix
    :param interp_idx: [B,up_num_points,1] nearest neighbour index
    :return:[B,d,up_num_points,1]
    """
    batch_size = interp_idx.size()[0]
    feature = torch.squeeze(feature, dim=3)  # [B,d,N]
    interp_idx = interp_idx.view(batch_size, -1)  # [B, num_points]
    interpolated_features = batch_gather(feature, interp_idx)  # [B, d, num_points]
    interpolated_features = torch.unsqueeze(interpolated_features, dim=3)
    return interpolated_features


def batch_gather(pc, index_input):
    """
    :param pc: [B,d,N]
    :param index_input: [B,num_points]
    :return:# (B,d,num_points)
    """
    batch_size = pc.size()[0]  # B
    for i in range(batch_size):
        points = pc[i]  # (d,N)
        index = index_input[i]  # (num_points,)
        result = torch.index_select(points, 1, index)  # （d，num_points）
        result = torch.unsqueeze(result, dim=0)  # (1,d,num_points)
        if i == 0:
            ans = result
        else:
            ans = torch.cat((ans, result), 0)
    return ans
