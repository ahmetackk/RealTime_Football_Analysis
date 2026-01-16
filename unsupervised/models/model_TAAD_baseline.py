import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.ops import roi_align


class X3D_TAAD_Baseline(nn.Module):

    """ 
    Inputs : sequences of T frames, M sequences of ROIs, M sequences of masks
        [(B,3,T,352,640), (B,M,T,5), (B,M,T)]
    """

    def __init__(self):
        super().__init__()

        self.x3d = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_l', pretrained=True)
        self.x3d_L4 = nn.Sequential(*[self.x3d.blocks[i] for i in range(2)])
        self.up_L32 = nn.Upsample(scale_factor=(1,2,2))
        self.conv_L16_32 = nn.Conv3d(in_channels=288, out_channels=192, kernel_size=(1,3,3), padding='same', bias=False)
        self.bn_L16_32 = nn.BatchNorm3d(192)
        self.up_L16 = nn.Upsample(scale_factor=(1,2,2))
        self.conv_L8_16 = nn.Conv3d(in_channels=240, out_channels=192, kernel_size=(1,3,3), padding='same', bias=False)
        self.bn_L8_16 = nn.BatchNorm3d(192)
        self.avgpool2D = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv1d(in_channels=192, out_channels=512, kernel_size=3, padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.fc1 = nn.Linear(512,9)

        
    def forward(self, in_x, return_embedding=False):

        x, roi, mask = in_x
        b, c, l, h, w = x.shape

        w = self.x3d_L4(x)
        z = self.x3d.blocks[2](w)
        y = self.x3d.blocks[3](z)
        x = self.x3d.blocks[4](y)
        x = self.up_L32(x)
        x = torch.concat((x,y), dim=1)
        x = self.conv_L16_32(x)
        x = F.gelu(self.bn_L16_32(x))
        x = self.up_L16(x)
        x = torch.concat((x,z), dim=1)
        x = self.conv_L8_16(x)
        x = F.gelu(self.bn_L8_16(x))

        _, _, _, fh, fw = x.shape
        x = x.permute(0,2,1,3,4).reshape(-1,192,fh,fw)

        _, M, _, _ = roi.shape
        roi = roi.permute(0,2,1,3).reshape(-1,5)
        f_num = roi[:,0]
        batch_indices = torch.arange(b).repeat_interleave(l * M).to(x.device)
        adjusted_frame_numbers = f_num + batch_indices * l
        roi[:,0] = adjusted_frame_numbers 

        x = roi_align(x, roi, (4,2), 0.125)
        x = self.avgpool2D(x).squeeze(-1).squeeze(-1).reshape(b,l,M,192).permute(0,2,3,1).reshape(b*M,192,l) 
        x = x*(mask.reshape(b*M,l).unsqueeze(1)) 
        
        x = F.gelu(self.bn1(self.conv1(x)))
        
        embedding_to_return = None
        if return_embedding:
            embedding_to_return = x.permute(0,2,1).reshape(b, M, l, 512)

        x = self.fc1(x.permute(0,2,1)) 
        
        logits = x.reshape(b,M,l,9).permute(0,3,1,2)

        if return_embedding:
            return logits, embedding_to_return
        else:
            return logits