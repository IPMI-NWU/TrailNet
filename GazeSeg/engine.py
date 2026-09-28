import datetime
import time
import numpy as np
import torch
import torchvision
from torch import nn
from github_version.MambaGazeSeg.utils.loss import DistillationLoss
from utils.loss import cross_entropy_loss_uncertain, deep_uncertain_loss
from utils.strong_aug import StrongAugmentations
from utils.misc import MetricLogger, SmoothedValue
from medpy.metric.binary import dc


def train_one_epoch(teacher_model, student_model, train_loader, optimizer_t, optimizer_s, device, epoch, args, writer):
    start_time = time.time()
    teacher_model.train()
    student_model.train()
    print('-' * 40)
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    total_steps = len(train_loader)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    Aug = StrongAugmentations()
    distill_loss_fun = DistillationLoss()
    # -------------------------------------------
    # Record the training process
    # -------------------------------------------
    img_list, label_list, gaze_list, output_list  = [], [], [], []
    dice_score_list = []
    # -------------------------------------------
    # Training
    # -------------------------------------------
    for step, batch in enumerate(train_loader):
        start = time.time()
        # -------------------------------------------
        # Load and move data
        # -------------------------------------------
        img, gaze, fixation = batch['image'].to(device), batch['pseudo_label'].to(device), batch['fixation'].to(device)
        label = batch['label'].to(device)
        datatime = time.time() - start
        img_aug = Aug(img)
        # -------------------------------------------
        # forward
        # -------------------------------------------
        output_t = teacher_model(img, fixation)
        output_s = student_model(img)
        output_logit_t = output_t['logits']
        output_logit_s = output_s['logits']
        output_aug_t = teacher_model(img_aug, fixation)
        output_aug_s = student_model(img_aug)
        output_aug_logit_t = output_aug_t['logits']
        output_aug_logit_s = output_aug_s['logits']
        # -------------------------------------------
        # gaze supervision loss
        # -------------------------------------------
        ce_loss_t = cross_entropy_loss_uncertain(output_logit_t, gaze, args.t1, args.t2)
        ce_loss_s = cross_entropy_loss_uncertain(output_logit_s, gaze, args.t1, args.t2)
        ce_loss_aug_t = cross_entropy_loss_uncertain(output_aug_logit_t, gaze, args.t1, args.t2)
        ce_loss_aug_s = cross_entropy_loss_uncertain(output_aug_logit_s, gaze, args.t1, args.t2)
        # -------------------------------------------
        # distillation loss
        # -------------------------------------------
        output_feature_t = output_t['features']
        output_feature_s = output_s['features']
        output_aug_feature_t = output_aug_t['features']
        output_aug_feature_s = output_aug_s['features']
        distill_loss_ts = distill_loss_fun(output_feature_t, output_feature_s)
        distill_loss_tt = distill_loss_fun(output_feature_t, output_aug_feature_t)
        distill_loss_aug_ts = distill_loss_fun(output_aug_feature_t, output_aug_feature_s)
        distill_loss_ss = distill_loss_fun(output_feature_s, output_aug_feature_s)
        distill_loss = (distill_loss_ts + distill_loss_tt + distill_loss_aug_ts + distill_loss_ss) / 4.
        # -------------------------------------------
        # Deep supervision loss
        # -------------------------------------------
        output_deep_t = output_t['deep_sup']
        output_deep_s = output_s['deep_sup']
        deep_loss_t = deep_uncertain_loss(output_deep_t, gaze, args.t1, args.t2)
        deep_loss_s = deep_uncertain_loss(output_deep_s, gaze, args.t1, args.t2)
        deep_loss = (deep_loss_t + deep_loss_s) / 2.
        # -------------------------------------------
        # Total loss
        # -------------------------------------------
        loss = ((ce_loss_t + ce_loss_s) + (ce_loss_aug_t + ce_loss_aug_s) +
                distill_loss + deep_loss)

        optimizer_t.zero_grad()
        optimizer_s.zero_grad()
        loss.backward()
        optimizer_t.step()
        optimizer_s.step()

        if step % 30 == 0:
            img_list.append(img[0].detach())
            label_list.append(label[0].detach())
            gaze_list.append(gaze[0].detach())
            output_list.append(torch.where(nn.Sigmoid()(output_logit_s[0]) > 0.5, 1, 0).detach())

        metric_logger.update(lr=optimizer_t.param_groups[0]["lr"])
        metric_logger.update(loss=loss)
        metric_logger.update(ce_loss_t=ce_loss_t)
        metric_logger.update(ce_loss_s=ce_loss_s)
        metric_logger.update(distill_loss=distill_loss)
        metric_logger.update(deep_loss=deep_loss)

        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)
        dice_score_list.append(dc(label, torch.where(nn.Sigmoid()(output_logit_s) > 0.5, 1, 0)))

    # gather the stats from all processes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    dice_score = np.array(dice_score_list).mean()

    writer.add_scalar('loss', loss.item(), epoch)
    writer.add_scalar('ce_loss_t', ce_loss_t.item(), epoch)
    writer.add_scalar('ce_loss_s', ce_loss_s.item(), epoch)
    writer.add_scalar('distill_loss', distill_loss.item(), epoch)
    writer.add_scalar('deep_loss', deep_loss.item(), epoch)
    writer.add_scalar('Train Dice Score', dice_score, epoch)

    def save_image(image, tag, epoch, writer):
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        grid = torchvision.utils.make_grid(torch.tensor(image), nrow=1, pad_value=1)
        writer.add_image(tag, grid, epoch)

    save_image(torch.stack(img_list).float(), 'img', epoch, writer)
    save_image(torch.stack(label_list).float(), 'label', epoch, writer)
    save_image(torch.stack(gaze_list).float(), 'gaze', epoch, writer)
    save_image(torch.stack(output_list).float(), 'output', epoch, writer)
