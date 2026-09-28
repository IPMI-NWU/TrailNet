import os
import argparse
import datetime
import random
from pathlib import Path
from tensorboardX import SummaryWriter
import numpy as np
import torch
from Data import get_dataloader
from engine import train_one_epoch
import time
from test import evaluate
from models.StudentUNet import StudentUNet
from models.TeacherMamba import TeacherMamba


def get_args_parser_1():
    parser = argparse.ArgumentParser('Full', add_help=False)
    parser.add_argument('--full', default=False, type=bool)
    parser.add_argument('--w', default=1.0, type=float)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lr_scheduler', default='StepLR', type=str) # StepLR, cos
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=30, type=int)
    parser.add_argument('--in_channels', default=3, type=int) # 3, 1
    parser.add_argument('--dataset', default='Kvasir', type=str) # Kvasir, NCI
    parser.add_argument('--max_len', default=60, type=int)
    parser.add_argument('--strategy', default='interpolate', type=str)  # truncate, interpolate
    parser.add_argument('--output_dir', default='output/Kvasir/')
    parser.add_argument('--t1', default=0.3, type=float)
    parser.add_argument('--t2', default=0.6, type=float)
    parser.add_argument('--warmup', default=20, type=int)
    parser.add_argument('--threshold', default=0.79, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--GPU_ids', type=str, default='0', help='Ids of GPUs')
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    return parser


def main(args):
    writer = SummaryWriter(log_dir=args.output_dir + '/summary')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(args.device)

    teacher_model = TeacherMamba().to(device)
    student_model = StudentUNet().to(device)

    optimizer_t = torch.optim.Adam(teacher_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_s = torch.optim.Adam(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n_parameters = sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    n_parameters = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.lr_scheduler == 'StepLR':
        lr_scheduler_t = torch.optim.lr_scheduler.StepLR(optimizer_t, args.lr_drop, gamma=0.1)
        lr_scheduler_s = torch.optim.lr_scheduler.StepLR(optimizer_s, args.lr_drop, gamma=0.1)
    else:
        lr_scheduler_t = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_t, T_0=10)
        lr_scheduler_s = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_s, T_0=10)

    print('Building dataset...')
    train_loader = get_dataloader(args, split="train", resize_label=True)
    print('Number of training images: {}'.format(len(train_loader) * args.batch_size))
    test_loader = get_dataloader(args, split="test", resize_label=True)
    print('Number of validation images: {}'.format(len(test_loader)))

    output_dir = Path(args.output_dir)
    best_dice = None
    start_time = time.time()
    with torch.cuda.device(device):
        for epoch in range(args.epochs):
            train_one_epoch(teacher_model, student_model, train_loader, optimizer_t, optimizer_s, device, epoch, args, writer)

            lr_scheduler_t.step()
            lr_scheduler_s.step()

            dice_score = evaluate(student_model, test_loader, device, epoch, writer)

            print("dice score:", dice_score)
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                if best_dice is None or dice_score > best_dice:
                    best_dice = dice_score
                    print("Update best model!")
                    checkpoint_paths.append(output_dir / 'best_checkpoint.pth')

                if dice_score > args.threshold:
                    print("Update high dice score model!")
                    file_name = str(dice_score)[0:6] + '_' + str(epoch) + '_checkpoint.pth'
                    checkpoint_paths.append(output_dir / file_name)

                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    torch.save(student_model.state_dict(), checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Full supervised training and evaluation script', parents=[get_args_parser_1()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU_ids)
    main(args)

