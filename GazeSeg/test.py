import datetime
import os
import time
import random
import argparse
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
import torch.nn.functional as F
import utils.misc as utils
from medpy.metric.binary import dc
from Data import get_dataloader
from github_version.MambaGazeSeg.models.StudentUNet import StudentUNet


class Visualization(nn.Module):
    def __init__(self):
        super().__init__()

    def save_image(self, image, tag, epoch, writer):
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        grid = torchvision.utils.make_grid(torch.tensor(image), nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    def forward(self, inputs, outputs, labels, epoch, writer):
        self.save_image(inputs, 'val_inputs', epoch, writer)
        self.save_image(outputs.float(), 'val_outputs', epoch, writer)
        self.save_image(labels.float(), 'val_labels', epoch, writer)


@torch.no_grad()
def evaluate(model, test_loader, device, epoch, writer):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    print_freq = 50
    total_steps = len(test_loader)
    start_time = time.time()
    img_list, output_list, label_list = [], [], []
    step = 0
    dice_score_list = []
    for data in test_loader:
        start = time.time()
        img, label = data['image'], data['label']
        img, label = img.to(device), label.to(device)
        datatime = time.time() - start

        output = model(img)['logits']
        output = F.interpolate(output, size=label.shape[2:], mode="bilinear")
        output = torch.where(nn.Sigmoid()(output) > 0.5, 1, 0)

        dice_score_list.append(dc(label, output))

        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)

        if step % 50 == 0:
            img_list.append(img[0].detach())
            output_list.append(output[0].detach())
            label_list.append(label[0].detach())
        step = step + 1

    # gather the stats from all processes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    dice_score = np.array(dice_score_list).mean()
    writer.add_scalar('Dice Score', dice_score, epoch)
    visualizer = Visualization()
    visualizer(torch.stack(img_list), torch.stack(output_list), torch.stack(label_list), epoch, writer)

    return dice_score


def get_args_parser():
    parser = argparse.ArgumentParser('Full', add_help=False)
    parser.add_argument('--dataset', default='Kvasir', type=str) # Kvasir, NCI
    parser.add_argument('--output_dir', default='output/Kvasir/')
    parser.add_argument('--max_len', default=10, type=int)
    parser.add_argument('--strategy', default='interpolate', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Full supervised training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format('0')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = StudentUNet()

    paths = ['']
    for path in paths:
        print()
        print(path)
        model_path = args.output_dir + path

        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.to(args.device)
        model.eval()
        test_loader = get_dataloader(args, split="test", resize_label=False)
        output_save_dir = args.output_dir + "/output/"
        if not os.path.exists(output_save_dir):
            os.makedirs(output_save_dir)

        dice_score_list = []
        my_dice_score_list = []
        for data in test_loader:
            start = time.time()
            img, label = data['image'], data['label']
            img, label = img.to(args.device), label.to(args.device)

            output = model(img)['logits']
            output = F.interpolate(output, size=label.shape[2:], mode="bilinear")
            output = nn.Sigmoid()(output)
            output = torch.where(output > 0.5, 1, 0)

            dice_score_list.append(dc(label, output))

            reduce_axis = list(range(2, len(img.shape)))
            intersection = torch.sum(output * label, dim=reduce_axis)
            input_o = torch.sum(output, dim=reduce_axis)
            target_o = torch.sum(label, dim=reduce_axis)
            my_dice = torch.mean(2 * intersection / (input_o + target_o + 1e-10), dim=1)
            my_dice_score_list.append(my_dice.item())

            output_img = np.array(output[0, 0].detach().cpu(), np.uint8) * 255
            output_img = Image.fromarray(output_img).convert('L')
            img_name = data['path'][0]
            output_img.save(output_save_dir + img_name.split('.')[0] + '.png')


        dice_score = np.array(dice_score_list).mean()
        dice_std = np.array(dice_score_list).std()
        print(dice_score, dice_std)

        dice_score = np.array(my_dice_score_list).mean()
        dice_std = np.array(my_dice_score_list).std()
        print(dice_score, dice_std)
