import torch
import torch.nn.functional as Func
from torch import nn


class DistillationLoss(nn.Module):
    def __init__(self):
        super(DistillationLoss, self).__init__()

    def forward(self, output_feature_t, output_feature_s):
        total_loss = 0
        for t_feat, s_feat in zip(output_feature_t, output_feature_s):
            cos_sim = Func.cosine_similarity(s_feat, t_feat, dim=1, eps=1e-6)
            total_loss += torch.mean(1 - cos_sim)
        total_loss = total_loss / len(output_feature_t)
        return total_loss


def deep_uncertain_loss(output_deep, gaze, t1, t2):
    loss_deep_uncertain = 0
    for output in output_deep:
        gaze = Func.interpolate(gaze, size=(output.shape[2], output.shape[3]), mode='nearest')
        gaze_binary = torch.where(gaze < t1, 0, gaze)
        gaze_binary = torch.where(gaze_binary > t2, 1, gaze_binary)
        mask = torch.where((gaze_binary >= t1) & (gaze_binary <= t2), 0, 1)

        output_fg, output_bg, output_uc = output[:, 0:1, :, :], output[:, 1:2, :, :], output[:, 2:3, :, :]

        gaze_binary_m, output_m = gaze_binary * mask, output_fg * mask
        certain_num = torch.sum(mask, dim=(1, 2, 3))
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        loss_fg = criterion(output_m, gaze_binary_m.float())
        loss_fg = (torch.sum(loss_fg, dim=(1, 2, 3)) + 1e-5) / (certain_num + 1e-5)
        loss_fg = loss_fg.mean()

        gaze_binary = torch.where(gaze < t1, -1, gaze)
        gaze_binary = torch.where(gaze_binary > t2, 0, gaze_binary)
        gaze_binary = torch.where(gaze_binary == -1, 1, gaze_binary)
        gaze_binary_m, output_m = gaze_binary * mask, output_bg * mask
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        loss_bg = criterion(output_m, gaze_binary_m.float())
        loss_bg = (torch.sum(loss_bg, dim=(1, 2, 3)) + 1e-5) / (certain_num + 1e-5)
        loss_bg = loss_bg.mean()

        output_fg, output_bg, output_uc = nn.Sigmoid()(output_fg), nn.Sigmoid()(output_bg), nn.Sigmoid()(output_uc)
        loss_uc = ((((1.0 + gaze) * output_fg * output_bg).sum() +
                    ((1.0 + gaze) * output_fg * output_uc).sum() +
                    ((1.0 + gaze) * output_bg * output_uc).sum()) /
                   (output_fg.size(0) * output_fg.size(2) * output_fg.size(3)))

        loss_deep_uncertain += loss_fg.mean() + loss_bg.mean() + loss_uc / 3.0

    return loss_deep_uncertain / len(output_deep)


def cross_entropy_loss_uncertain(output1, gaze, t1, t2):
    gaze_binary = torch.where(gaze < t1, 0, gaze)
    gaze_binary = torch.where(gaze_binary > t2, 1, gaze_binary)
    mask = torch.where((gaze_binary >= t1) & (gaze_binary <= t2), 0, 1)
    gaze_binary_m, output1_m = gaze_binary * mask, output1 * mask
    certain_num = torch.sum(mask, dim=(1, 2, 3))

    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    ce_loss_1 = criterion(output1_m, gaze_binary_m.float())
    ce_loss_1 = (torch.sum(ce_loss_1, dim=(1, 2, 3)) + 1e-5) / (certain_num + 1e-5)
    ce_loss = ce_loss_1.mean()
    return ce_loss
