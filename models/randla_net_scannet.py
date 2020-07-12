from .blocks import *
import numpy as np


class RandLA_Net_ScanNet(nn.Module):
    def __init__(self, args):
        super(RandLA_Net_ScanNet, self).__init__()
        self.num_cls = args.num_cls
        self.dataset_name = args.dataset_name
        self.d_out = args.d_out  # [16, 64, 128, 256, 512]
        self.num_layers = args.num_layers
        self.n_pts = args.n_pts
        # [4, 4, 4, 4, 2]
        self.ratio = args.sub_sampling_ratio
        # 40960,51200,53760,54400,54560
        self.index = []
        index = 0
        div_num = 1
        for i in range(self.num_layers):
            if index == 0:
                index = args.n_pts
            else:
                div_num *= self.ratio[i - 1]
                index += args.n_pts // div_num
            self.index.append(index)
        self.index = np.array(self.index)
        # 10240, 12800,13440,13600,13680
        self.sample = []
        index = 0
        div_num = 1
        for i in range(self.num_layers):
            div_num *= self.ratio[i]
            index += args.n_pts // div_num
            self.sample.append(index)
        self.sample = np.array(self.sample)

        self.fc0 = nn.Linear(6, 8)
        self.bn0 = nn.BatchNorm1d(8)

        #######  Encoder  #########

        self.res_block0 = Dilated_res_block(8, self.d_out[0])
        self.res_block1 = Dilated_res_block(self.d_out[0] * 2, self.d_out[1])  # output: self.d_out[1] * 2
        self.res_block2 = Dilated_res_block(self.d_out[1] * 2, self.d_out[2])
        self.res_block3 = Dilated_res_block(self.d_out[2] * 2, self.d_out[3])
        self.res_block4 = Dilated_res_block(self.d_out[3] * 2, self.d_out[4])

        ############## Encoder #########

        self.conv1 = nn.Conv2d(self.d_out[4] * 2, self.d_out[4] * 2, kernel_size=(1, 1), stride=1)
        self.bn1 = nn.BatchNorm2d(self.d_out[4] * 2)

        ###############  Decoder  #########
        self.upconv2 = nn.ConvTranspose2d(self.d_out[4] * 2 + self.d_out[3] * 2, self.d_out[3] * 2, kernel_size=(1, 1),
                                          stride=1)
        self.bn2 = nn.BatchNorm2d(self.d_out[3] * 2)
        self.upconv3 = nn.ConvTranspose2d(self.d_out[3] * 2 + self.d_out[2] * 2, self.d_out[2] * 2, kernel_size=(1, 1),
                                          stride=1)
        self.bn3 = nn.BatchNorm2d(self.d_out[2] * 2)
        self.upconv4 = nn.ConvTranspose2d(self.d_out[2] * 2 + self.d_out[1] * 2, self.d_out[1] * 2, kernel_size=(1, 1),
                                          stride=1)
        self.bn4 = nn.BatchNorm2d(self.d_out[1] * 2)
        self.upconv5 = nn.ConvTranspose2d(self.d_out[1] * 2 + self.d_out[0] * 2, self.d_out[0] * 2, kernel_size=(1, 1),
                                          stride=1)
        self.bn5 = nn.BatchNorm2d(self.d_out[0] * 2)
        self.upconv6 = nn.ConvTranspose2d(self.d_out[0] * 4, self.d_out[0] * 2, kernel_size=(1, 1), stride=1)
        self.bn6 = nn.BatchNorm2d(self.d_out[0] * 2)

        ##############  Decoder   #########

        self.conv7 = nn.Conv2d(self.d_out[0] * 2, 64, kernel_size=(1, 1), stride=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 32, kernel_size=(1, 1), stride=1)
        self.bn8 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout2d(p=0.5)
        self.conv9 = nn.Conv2d(32, self.num_cls, kernel_size=(1, 1), stride=1)  # No activation

    def forward(self, xyz, neigh_idx, sub_idx, interp_idx, x):
        """
        :param xyz: [B,3,n + 1/4*n + 1/16*n + 1/64*n + 1/128*n]
        :param neigh_idx: [B, n + 1/4*n + 1/16*n + 1/64*n + 1/128*n,3]
        :param sub_idx: [ B,1/4*n + 1/16*n + 1/64*n + 1/128*n + 1/256*n,3]
        :param interp_idx: [B,n + 1/4*n + 1/16*n + 1/64*n + 1/128*n,1]
        :param x: [B,n_pts，6]
        :return:
        """
        batch_size = x.size()[0]
        n_pts = x.size()[1]
        x = x.view(-1, 6)
        x = F.leaky_relu(self.bn0(self.fc0(x)))  # B * n_pts，8
        x = x.view(batch_size, n_pts, 8)
        x = x.permute(0, 2, 1)
        x = torch.unsqueeze(x, dim=3)  # B * 8 * N * 1

        encoder_0 = self.res_block0(x, xyz[:, :, :self.index[0]],
                                    neigh_idx[:, :self.index[0], :])  # B, d_out[0] * 2, N,1
        sampled_0 = random_sample(encoder_0, sub_idx[:, :self.sample[0], :])  # B,d_out[0] *2, N', 1

        encoder_1 = self.res_block1(sampled_0, xyz[:, :, self.index[0]:self.index[1]],
                                    neigh_idx[:, self.index[0]:self.index[1], :])
        sampled_1 = random_sample(encoder_1, sub_idx[:, self.sample[0]:self.sample[1], :])

        encoder_2 = self.res_block2(sampled_1, xyz[:, :, self.index[1]:self.index[2]],
                                    neigh_idx[:, self.index[1]:self.index[2], :])
        sampled_2 = random_sample(encoder_2, sub_idx[:, self.sample[1]:self.sample[2], :])

        encoder_3 = self.res_block3(sampled_2, xyz[:, :, self.index[2]:self.index[3]],
                                    neigh_idx[:, self.index[2]:self.index[3], :])
        sampled_3 = random_sample(encoder_3, sub_idx[:, self.sample[2]:self.sample[3], :])

        encoder_4 = self.res_block4(sampled_3, xyz[:, :, self.index[3]:self.index[4]],
                                    neigh_idx[:, self.index[3]:self.index[4], :])
        sampled_4 = random_sample(encoder_4, sub_idx[:, self.sample[3]:self.sample[4], :])

        feature = F.leaky_relu(self.bn1(self.conv1(sampled_4)))

        interp_0 = nearest_interpolation(feature, interp_idx[:, self.index[3]:self.index[4], :])
        feature = torch.cat([interp_0, sampled_3], dim=1)
        decoder_0 = F.leaky_relu(self.bn2(self.upconv2(feature)))

        interp_1 = nearest_interpolation(decoder_0, interp_idx[:, self.index[2]:self.index[3], :])
        feature = torch.cat([interp_1, sampled_2], dim=1)
        decoder_1 = F.leaky_relu(self.bn3(self.upconv3(feature)))

        interp_2 = nearest_interpolation(decoder_1, interp_idx[:, self.index[1]:self.index[2], :])
        feature = torch.cat([interp_2, sampled_1], dim=1)
        decoder_2 = F.leaky_relu(self.bn4(self.upconv4(feature)))

        inter_3 = nearest_interpolation(decoder_2, interp_idx[:, self.index[0]:self.index[1], :])
        feature = torch.cat([inter_3, sampled_0], dim=1)
        decoder_3 = F.leaky_relu(self.bn5(self.upconv5(feature)))

        inter_4 = nearest_interpolation(decoder_3, interp_idx[:, :self.index[0], :])
        feature = torch.cat([inter_4, encoder_0], dim=1)
        decoder_4 = F.leaky_relu(self.bn6(self.upconv6(feature)))

        x = F.leaky_relu(self.bn7(self.conv7(decoder_4)))
        x = F.leaky_relu(self.bn8(self.conv8(x)))
        x = self.dropout(x)
        x = self.conv9(x)
        x = torch.squeeze(x, dim=3)  # B, num_class, N
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
