from pytorch_msssim import ssim
import torch
import torch.nn as nn
import torch.optim as optim

class CompositeLoss(nn.Module):
    def __init__(self):
        super(CompositeLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = 0.84

    def forward(self, pred, target):
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)

        mse_loss = self.mse(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0, size_average=True)

        total_loss = (1 - self.alpha) * mse_loss + self.alpha * ssim_loss

        return total_loss
