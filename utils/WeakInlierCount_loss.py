from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from ultralytics.nn.geotnf.transformation import GeometricTnf
import scipy.signal

def expand_dim(tensor,dim,desired_dim_len):
    sz = list(tensor.size())
    sz[dim]=desired_dim_len
    return tensor.expand(tuple(sz))

class WeakInlierCount(nn.Module):
    def __init__(self, geometric_model='affine', tps_grid_size=3, tps_reg_factor=0.2, h_matches=14, w_matches=14, use_conv_filter=False, dilation_filter=0, normalize_inlier_count=True, offset_factor=None, device='cpu'):
        super(WeakInlierCount, self).__init__()
        self.normalize=normalize_inlier_count
        self.geometric_model = geometric_model
        self.geometricTnf = GeometricTnf(geometric_model=geometric_model,
                                         tps_grid_size=tps_grid_size,
                                         tps_reg_factor=tps_reg_factor,
                                         out_h=h_matches, out_w=w_matches,
                                         offset_factor = offset_factor,
                                         device=device)
        if dilation_filter is None:
            dilation_filter = generate_binary_structure(2, 2)
        mask_id = np.zeros((w_matches,h_matches,w_matches*h_matches))
        idx_list = list(range(0, mask_id.size, mask_id.shape[2]+1))
        mask_id.reshape((-1))[idx_list]=1
        mask_id = mask_id.swapaxes(0,1)
        if not use_conv_filter:
            if not (isinstance(dilation_filter,int) and dilation_filter==0):
                for i in range(mask_id.shape[2]):
                    mask_id[:,:,i] = binary_dilation(mask_id[:,:,i],structure=dilation_filter).astype(mask_id.dtype)
        else:
            for i in range(mask_id.shape[2]):
                flt=np.array([[1/16,1/8,1/16],
                                 [1/8, 1/4, 1/8],
                                 [1/16,1/8,1/16]])
                mask_id[:,:,i] = scipy.signal.convolve2d(mask_id[:,:,i], flt, mode='same', boundary='fill', fillvalue=0)
        mask_id = Variable(torch.FloatTensor(mask_id).transpose(1,2).transpose(0,1).unsqueeze(0),requires_grad=False)
        self.mask_id = mask_id
        self.mask_id = self.mask_id.to(device)

    def forward(self, theta, matches, return_outliers=False):
        # print('matches score', matches.max())
        theta = theta.view(-1, 2, 3)
        if isinstance(theta,Variable): # handle normal batch transformations
            batch_size=theta.size()[0]
            theta=theta.clone()
            mask = self.geometricTnf(expand_dim(self.mask_id,0,batch_size),theta)

            if self.normalize:
                epsilon=1e-5
                mask = torch.div(mask,
                                 torch.sum(torch.sum(torch.sum(mask+epsilon,3),2),1).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(mask))
            score = torch.sum(torch.sum(torch.sum(torch.mul(mask,matches),3),2),1)

            # print('matches score', score)


        # return torch.mean(-score)
        # return torch.mean(1-score) #luoru
        return torch.mean(score) #luoru

import torch
import torch.nn as nn


def featureL2Norm(feature):
    epsilon = 1e-6
    #        print(feature.size())
    #        print(torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).size())
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    # print(torch.sum(torch.pow(feature,2),1).shape)
    # exit()
    return torch.div(feature, norm)


class correlation(nn.Module):
    def __init__(self, normalization=True, size_thr=28):
        super(correlation, self).__init__()
        self.normalization = normalization
        # self.down = focus(downsample_rato xuyao e, out_dim)
        self.ReLU = nn.ReLU()
        self.size_thr = size_thr
        self.upsample = nn.Upsample(scale_factor=1 / 2, mode='bilinear')

    def forward(self, f_A_list, f_B_list, bidirection=False):
        assert len(f_A_list) == len(f_B_list)
        correlation_tensor_B2A_list, correlation_tensor_A2B_list = [], []
        for f_A, f_B in zip(f_A_list, f_B_list):
            b, c, h, w = f_B.size()
            # f_A = f_A.view(-1, h, w, c).transpose(2, 3).transpose(1, 2)
            f_A = f_A.reshape(-1, h, w, c).transpose(2, 3).transpose(1, 2)
            assert b == f_A.size(0)
            if h > self.size_thr and w > self.size_thr:
                f_A, f_B = self.upsample(f_A), self.upsample(f_B)
                b, c, h, w = f_B.size()

            if self.normalization:
                f_A, f_B = featureL2Norm(f_A), featureL2Norm(f_B)

                feature_mul_B2A = torch.bmm(f_A.view(b, c, h * w).transpose(1, 2),
                                            f_B.transpose(2, 3).contiguous().view(b, c, h * w))
                correlation_tensor_B2A = feature_mul_B2A.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
                correlation_tensor_B2A = featureL2Norm(self.ReLU(correlation_tensor_B2A))
                correlation_tensor_B2A_list.append(correlation_tensor_B2A)

                feature_mul_A2B = torch.bmm(f_B.view(b, c, h * w).transpose(1, 2),
                                            f_A.transpose(2, 3).contiguous().view(b, c, h * w))
                correlation_tensor_A2B = feature_mul_A2B.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
                correlation_tensor_A2B = featureL2Norm(self.ReLU(correlation_tensor_A2B))
                correlation_tensor_A2B_list.append(correlation_tensor_A2B)
        if not bidirection:
            return correlation_tensor_B2A_list
        else:
            return correlation_tensor_B2A_list, correlation_tensor_A2B_list

if __name__ == '__main__':
    correlation_layer = correlation()
    WeakInlierCount_loss = WeakInlierCount( h_matches=32, w_matches=32)
    input= torch.randn(4,23,64,64)
    input1= torch.randn(4,23,64,64)
    matches = correlation_layer([input],[input1])
     

    affines = torch.tensor([[[-1.678448, - 0.152306, 0.923077], [0.152159, - 0.831081, 0.054054]],
                            [[-1.678448, - 0.152306, 0.923077], [0.152159, - 0.831081, 0.054054]],
                            [[-1.678448, - 0.152306, 0.923077], [0.152159, - 0.831081, 0.054054]],
                           [[1, 0, 0], [0, 1, 0]]
                            ], dtype=torch.float64)

    loss = WeakInlierCount_loss(theta=affines, matches=matches[0])
    # print(matches[0].shape)
    # print(loss)
  