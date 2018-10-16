import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
import itertools
import numpy as np
import Relation_Resoning_Simple
from torch.autograd import Variable
import myresnet

class Net(nn.Module):
    def __init__(self, basemodel, num_segments, num_frames, dataset, enable_bn=True, d_model=512, start=2):
        super(Net, self).__init__()
        self.num_segments = num_segments
        self.num_frames = num_frames
        self.dataset = dataset
        self.enable_bn = enable_bn
        self.d_model = d_model
        self.basemodel = basemodel
        self.start = start

        if self.dataset == 'hmdb':
            self.num_class = 51
        elif self.dataset == 'ucf':
            self.num_class = 101

        if self.basemodel == 'resnet34':
            self.img_feature_dim = 896

        self.net = myresnet.resnet34(True)
        self.pyramid_low = Relation_Resoning_Simple.Feature_Pyramid_low()
        self.pyramid_mid = Relation_Resoning_Simple.Feature_Pyramid_Mid()
        self.pyramid_high = Relation_Resoning_Simple.Feature_Pyramid_High()

        self.reason = Relation_Resoning_Simple.Resoning(num_segments=self.num_segments, num_frames=self.num_frames, num_class=self.num_class, img_dim=self.img_feature_dim, start=self.start)

    def forward(self, x):
        low, mid, high = self.net(x)
        low_0 = self.pyramid_low(low).squeeze()
        mid_0 = self.pyramid_mid(mid).squeeze()
        high_0 = self.pyramid_high(high).squeeze()
        cnn_output = torch.cat((high_0, mid_0, low_0), dim=-1)
        # cnn_output = high_0 + mid_0 + low_0
        frame_feature = cnn_output.view(-1, self.num_segments * self.num_frames, self.img_feature_dim)
        output = self.reason(frame_feature)
        return output
