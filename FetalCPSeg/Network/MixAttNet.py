import torch
from torch import nn
from torch.nn import functional as F
import random
import nibabel as nib
import glob
import string


def convolution_block(in_chan, out_chan, ksize=3, pad=1, stride=1, bias=False):
    """
    Convolution Block
    Convolution + Normalization + NonLinear
    """
    return nn.Sequential(
        #nn.Conv3d(in_chan, out_chan, kernel_size=ksize, padding=pad, stride=stride, bias=bias),
        nn.Conv2d(in_chan, out_chan, kernel_size=ksize, padding=pad, stride=stride, bias=bias),
        #nn.BatchNorm3d(out_chan),
        nn.BatchNorm2d(out_chan),
        nn.PReLU()
    )


def up_sample3d(x, t, mode="bilinear"):
    """
    3D Up Sampling
    """
    return F.interpolate(x, t.size()[2:], mode=mode, align_corners=False)


class ResStage(nn.Module):
    """
    3D Res stage
    """
    def __init__(self, in_chan, out_chan, stride=1):
        super(ResStage, self).__init__()
        self.conv1 = convolution_block(in_chan, out_chan, stride=stride)
        self.conv2 = nn.Sequential(
            #nn.Conv3d(out_chan, out_chan, kernel_size=3, padding=1),
            #nn.BatchNorm3d(out_chan)
            nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_chan)
        )
        self.non_linear = nn.PReLU()
        self.down_sample = nn.Sequential(
            #nn.Conv3d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
            #nn.BatchNorm3d(out_chan),
            nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_chan))

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        shortcut = self.down_sample(x)
        out = self.non_linear(out + shortcut)

        return out


def down_stage(in_chan, out_chan):
    return nn.Sequential(
        #nn.Conv3d(in_chan, out_chan, kernel_size=1, bias=False),
        #nn.BatchNorm3d(out_chan),
        nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_chan),
        nn.PReLU()
    )


class MixBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(MixBlock, self).__init__()
        #self.conv1 = nn.Conv3d(in_chan, out_chan // 4, 3, padding=1, bias=False)
        #self.conv3 = nn.Conv3d(in_chan, out_chan // 4, 5, padding=2, bias=False)
        #self.conv5 = nn.Conv3d(in_chan, out_chan // 4, 7, padding=3, bias=False)
        #self.conv7 = nn.Conv3d(in_chan, out_chan // 4, 9, padding=4, bias=False)
        #self.bn1 = nn.BatchNorm3d(out_chan // 4)
        #self.bn3 = nn.BatchNorm3d(out_chan // 4)
        #self.bn5 = nn.BatchNorm3d(out_chan // 4)
        #self.bn7 = nn.BatchNorm3d(out_chan // 4)
        self.conv1 = nn.Conv2d(in_chan, out_chan // 4, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_chan, out_chan // 4, 5, padding=2, bias=False)
        self.conv5 = nn.Conv2d(in_chan, out_chan // 4, 7, padding=3, bias=False)
        self.conv7 = nn.Conv2d(in_chan, out_chan // 4, 9, padding=4, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chan // 4)
        self.bn3 = nn.BatchNorm2d(out_chan // 4)
        self.bn5 = nn.BatchNorm2d(out_chan // 4)
        self.bn7 = nn.BatchNorm2d(out_chan // 4)
        self.nonlinear = nn.PReLU()

    def forward(self, x):
        k1 = self.bn1(self.conv1(x))
        k3 = self.bn3(self.conv3(x))
        k5 = self.bn5(self.conv5(x))
        k7 = self.bn7(self.conv7(x))
        return self.nonlinear(torch.cat((k1, k3, k5, k7), dim=1))


class Attention(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Attention, self).__init__()
        self.mix1 = MixBlock(in_chan, out_chan)
        #self.conv1 = nn.Conv3d(out_chan, out_chan, kernel_size=1)
        self.conv1 = nn.Conv2d(out_chan, out_chan, kernel_size=1)
        self.mix2 = MixBlock(out_chan, out_chan)
        #self.conv2 = nn.Conv3d(out_chan, out_chan, kernel_size=1)
        #self.norm1 = nn.BatchNorm3d(out_chan)
        #self.norm2 = nn.BatchNorm3d(out_chan)
        self.conv2 = nn.Conv2d(out_chan, out_chan, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(out_chan)
        self.norm2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.PReLU()

    def forward(self, x):
        shortcut = x
        mix1 = self.conv1(self.mix1(x))
        mix2 = self.mix2(mix1)
        att_map = F.sigmoid(self.conv2(mix2))
        out = self.norm1(x*att_map) + self.norm2(shortcut)
        return self.relu(out), att_map


def out_stage(in_chan, out_chan):
    return nn.Sequential(
        #nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1),
        #nn.BatchNorm3d(out_chan),
        #nn.PReLU(),
        #nn.Conv3d(out_chan, 1, kernel_size=1)
        nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_chan),
        nn.PReLU(),
        nn.Conv2d(out_chan, 1, kernel_size=1)
    )


class MixAttNet(nn.Module):
    def __init__(self):
        super(MixAttNet, self).__init__() 
        self.init_block = convolution_block(1, 16)
        self.enc1 = ResStage(16, 16, 1)
        self.enc2 = ResStage(16, 32, 2)
        self.enc3 = ResStage(32, 64, 2)
        self.enc4 = ResStage(64, 128, 2)
        self.enc5 = ResStage(128, 128, 2)

        self.dec4 = ResStage(128+128, 64)
        self.dec3 = ResStage(64+64, 32)
        self.dec2 = ResStage(32+32, 16)
        self.dec1 = ResStage(16+16, 16)

        self.down4 = down_stage(64, 16)
        self.down3 = down_stage(32, 16)
        self.down2 = down_stage(16, 16)
        self.down1 = down_stage(16, 16)

        # Attentive Module
        self.mix1 = Attention(16, 16)
        self.mix2 = Attention(16, 16)
        self.mix3 = Attention(16, 16)
        self.mix4 = Attention(16, 16)

        # Generate Attentive Maps
        #self.att_out1 = nn.Conv2d(16, 1, kernel_size=1)
        #self.att_out2 = nn.Conv2d(16, 1, kernel_size=1)
        #self.att_out3 = nn.Conv2d(16, 1, kernel_size=1)
        #self.att_out4 = nn.Conv2d(16, 1, kernel_size=1)

        #self.mix_outX = nn.Conv3d(16, 1, kernel_size=1)
        self.mix_out1 = nn.Conv2d(16, 1, kernel_size=1)
        self.mix_out2 = nn.Conv2d(16, 1, kernel_size=1)
        self.mix_out3 = nn.Conv2d(16, 1, kernel_size=1)
        self.mix_out4 = nn.Conv2d(16, 1, kernel_size=1)
        #self.down_outX = nn.Conv3d(16, 1, kernel_size=1)
        self.down_out1 = nn.Conv2d(16, 1, kernel_size=1)
        self.down_out2 = nn.Conv2d(16, 1, kernel_size=1)
        self.down_out3 = nn.Conv2d(16, 1, kernel_size=1)
        self.down_out4 = nn.Conv2d(16, 1, kernel_size=1)

        self.out = out_stage(16*4, 64)

    def forward(self, x):
        x = self.init_block(x)
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        dec4 = self.dec4(
            torch.cat((enc4, up_sample3d(enc5, enc4)), dim=1))
        dec3 = self.dec3(
            torch.cat((enc3, up_sample3d(dec4, enc3)), dim=1))
        dec2 = self.dec2(
            torch.cat((enc2, up_sample3d(dec3, enc2)), dim=1))
        dec1 = self.dec1(
            torch.cat((enc1, up_sample3d(dec2, enc1)), dim=1))

        down1 = up_sample3d(self.down1(dec1), x)
        down4 = up_sample3d(self.down4(dec4), x)
        down3 = up_sample3d(self.down3(dec3), x)
        down2 = up_sample3d(self.down2(dec2), x)
        
        down_out1 = self.down_out1(down1)
        down_out2 = self.down_out2(down2)
        down_out3 = self.down_out3(down3)
        down_out4 = self.down_out4(down4)
        
        #down_out1_ = nib.Nifti1Image(down_out1.squeeze().detach().cpu().numpy(), affine = None)
        #down_out2_ = nib.Nifti1Image(down_out2.squeeze().detach().cpu().numpy(), affine = None)
        #down_out3_ = nib.Nifti1Image(down_out3.squeeze().detach().cpu().numpy(), affine = None)
        #down_out4_ = nib.Nifti1Image(down_out4.squeeze().detach().cpu().numpy(), affine = None)

        # Attentive Maps
        mix1, att1 = self.mix1(down1)
        mix2, att2 = self.mix2(down2)
        mix3, att3 = self.mix3(down3)
        mix4, att4 = self.mix4(down4)

        #att_out1 = self.att_out1(att1)
        #att_out2 = self.att_out2(att2)
        #att_out3 = self.att_out3(att3)
        #att_out4 = self.att_out4(att4)

        #att_out1_ = nib.Nifti1Image(att_out1.squeeze().detach().cpu().numpy(), affine = None)
        #att_out2_ = nib.Nifti1Image(att_out2.squeeze().detach().cpu().numpy(), affine = None)
        #att_out3_ = nib.Nifti1Image(att_out3.squeeze().detach().cpu().numpy(), affine = None)
        #att_out4_ = nib.Nifti1Image(att_out4.squeeze().detach().cpu().numpy(), affine = None)

        mix_out1 = self.mix_out1(mix1)
        mix_out2 = self.mix_out2(mix2)
        mix_out3 = self.mix_out3(mix3)
        mix_out4 = self.mix_out4(mix4)

        #mix_out1_ = nib.Nifti1Image(mix_out1.squeeze().detach().cpu().numpy(), affine = None)
        #mix_out2_ = nib.Nifti1Image(mix_out2.squeeze().detach().cpu().numpy(), affine = None)
        #mix_out3_ = nib.Nifti1Image(mix_out3.squeeze().detach().cpu().numpy(), affine = None)
        #mix_out4_ = nib.Nifti1Image(mix_out4.squeeze().detach().cpu().numpy(), affine = None) 
        #out = down_out1#self.out(mix1)
        
        out = self.out(torch.cat((mix1, mix2, mix3, mix4), dim=1))
        #out_ = nib.Nifti1Image(out.squeeze().detach().cpu().numpy(), affine = None)

        value = 0
        if value == 10:
            #if len(glob.glob('att_maps/*.nii.gz')) < 10:                          
            idx = ''.join(random.choice(string.ascii_letters) for m in range(6))   
            nib.save(att1_, 'att_maps/att/'+idx+'_1.nii.gz')                         
            #nib.save(mix1_, 'att_maps/mix/'+idx+'_1.nii.gz')                           
            nib.save(mix_out1_, 'att_maps/mix_out/'+idx+'_1.nii.gz')
            nib.save(down_out1_, 'att_maps/down_out/'+idx+'_1.nii.gz')
            #nib.save(att_out1_, 'att_maps/att_out/'+idx+'_1.nii.gz')
            nib.save(att2_, 'att_maps/att/'+idx+'_2.nii.gz')
            #nib.save(mix2_, 'att_maps/mix/'+idx+'_2.nii.gz')
            nib.save(mix_out2_, 'att_maps/mix_out/'+idx+'_2.nii.gz')
            nib.save(down_out2_, 'att_maps/down_out/'+idx+'_2.nii.gz')
            #nib.save(att_out2_, 'att_maps/att_out/'+idx+'_2.nii.gz')
            nib.save(att3_, 'att_maps/att/'+idx+'_3.nii.gz')
            #nib.save(mix3_, 'att_maps/mix/'+idx+'_3.nii.gz')
            nib.save(mix_out3_, 'att_maps/mix_out/'+idx+'_3.nii.gz')
            nib.save(down_out3_, 'att_maps/down_out/'+idx+'_3.nii.gz')
            #nib.save(att_out3_, 'att_maps/att_out/'+idx+'_3.nii.gz')
            nib.save(att4_, 'att_maps/att/'+idx+'_4.nii.gz')
            #nib.save(mix4_, 'att_maps/mix/'+idx+'_4.nii.gz')
            nib.save(mix_out4_, 'att_maps/mix_out/'+idx+'_4.nii.gz')
            nib.save(down_out4_, 'att_maps/down_out/'+idx+'_4.nii.gz')
            #nib.save(att_out4_, 'att_maps/att_out/'+idx+'_4.nii.gz')
            nib.save(out_, 'att_maps/'+idx+'.nii.gz') 

        if self.training:
            #return out
            return out, mix_out1, mix_out2, mix_out3, mix_out4, down_out1, down_out2, down_out3, down_out4
        else:
            return torch.sigmoid(out)


if __name__ == '__main__':
    net = MixAttNet().cuda()
    torch.save(net.state_dict(), "MixAttNet.pth.gz")
