import os, glob

import PIL.Image
import pandas as pd
import torch
from PIL import Image
import numpy as np
from .base_dataset import BaseImageDataset

from torchvision import transforms
from monai.transforms import (
    MapLabelValue, EnsureChannelFirstd, NormalizeIntensityd, Resized, RandFlipd,
    ToTensord, Compose, RandAffined, RandRotated, RandSpatialCrop, RandSpatialCropd
)


class KvasirSegDataset(BaseImageDataset):
    def __init__(self, *argv, **kargs):
        super().__init__(*argv, **kargs)

        split_file = pd.read_csv(os.path.join(self.root, f"{self.split}.txt"), sep=" ", header=None)
        self.sample_list = split_file.iloc[:, 0].tolist()

        self.images = np.array([os.path.join(self.root, "images", f"{file}.jpg") for file in self.sample_list])[
            : int(len(self.sample_list) * self.size_rate)
        ]
        self.labels = np.array([os.path.join(self.root, "masks", f"{file}.jpg") for file in self.sample_list])[
            : int(len(self.sample_list) * self.size_rate)
        ]

    def _fetch_data(self, idx):
        image = np.array(Image.open(self.images[idx]).convert("RGB"), dtype=np.float32)
        label = np.array(Image.open(self.labels[idx]).convert("L"), dtype=np.int16)

        return {"image": image, "label": label}

    def _transform_custom(self, data):
        label = data["label"]
        label = label.float() / 255.0

        label[label > 0.5] = 1
        label[label < 0.5] = 0

        data["label"] = label.long()

        return data


class KvasirGazeDataset(KvasirSegDataset):
    def __init__(self, pseudo_mask_root, fixation_path, max_len, strategy, *argv, **kargs):
        super().__init__(*argv, **kargs)
        self.pseudo_mask_root = pseudo_mask_root
        self.pseudo_labels = [np.array([os.path.join(pseudo_mask_root, f"heatmap", f"{file}.jpg",) for file in self.sample_list])]
        self.fixation_data = pd.read_csv(fixation_path)
        self.max_len = max_len
        self.strategy = strategy

    def _transform_custom(self, data):
        data = super()._transform_custom(data)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = data["image"]
        img = normalize(img / 255)
        data["image"] = img
        if self.split == "train":
            img_aug = data["image_aug"]
            img_aug = normalize(img_aug / 255)
            data["image_aug"] = img_aug
        return data

    def get_transform(self):
        if self.split == "train" and self.do_augmentation:
            resize_keys = ["image", "image_aug", "label", "label_aug", "pseudo_label", "pseudo_label_aug"]
            resize_mode = ["bilinear", "bilinear", "nearest", "nearest", "bilinear", "bilinear"]
            return Compose(
                [
                    EnsureChannelFirstd(
                        keys=["image", "image_aug"],
                        channel_dim=2,
                    ),
                    EnsureChannelFirstd(
                        keys=["label", "label_aug", "pseudo_label", "pseudo_label_aug"],
                        channel_dim="no_channel",
                    ),
                    Resized(
                        keys=resize_keys,
                        spatial_size=self.spatial_size,
                        mode=resize_mode,
                    ),
                    RandFlipd(
                        keys=["image_aug", "label_aug", "pseudo_label_aug"],
                        prob=0.5,
                        spatial_axis=0,
                    ),
                    RandFlipd(
                        keys=["image_aug", "label_aug", "pseudo_label_aug"],
                        prob=0.5,
                        spatial_axis=1,
                    ),
                    RandRotated(
                        keys=["image_aug", "label_aug", "pseudo_label_aug"],
                        mode=['bilinear', 'nearest', 'bilinear'],
                        range_x=np.pi / 18,
                        range_y=np.pi / 18,
                        prob=0.5,
                        padding_mode=['reflection', 'reflection', 'reflection'],
                    ),
                    RandAffined(
                        keys=["image_aug", "label_aug", "pseudo_label_aug"],
                        mode=('bilinear', 'nearest', 'bilinear'),
                        prob=0.3,
                        rotate_range=(np.pi / 2, np.pi / 2),
                        scale_range=(0.05, 0.05)
                    ),
                ]
            )
        else:
            resize_keys = ["image", "label", "pseudo_label"] if self.resize_label else ["image"]
            resize_mode = ["bilinear", "nearest", "bilinear"] if self.resize_label else ["bilinear"]
            return Compose(
                [
                    EnsureChannelFirstd(
                        keys=["image"],
                        channel_dim=2,
                    ),
                    EnsureChannelFirstd(
                        keys=["label", "pseudo_label"],
                        channel_dim="no_channel",
                    ),
                    Resized(
                        keys=resize_keys,
                        spatial_size=self.spatial_size,
                        mode=resize_mode,
                    ),
                ]
            )

    def truncate_sequence(self, seq: np.ndarray, M: int) -> np.ndarray:
        if len(seq) <= M:
            return seq
        return seq[:M].copy()

    def interpolate_sequence(self, seq: np.ndarray, M: int) -> np.ndarray:
        N = len(seq)
        if N == 0:
            return np.zeros((M, 3), dtype=np.float32)
        if N == 1:
            return np.repeat(seq[0:1, :], M, axis=0).astype(np.float32)
        # original indices
        orig_idx = np.linspace(0, 1, N)
        target_idx = np.linspace(0, 1, M)
        out = np.zeros((M, 3), dtype=np.float32)
        for dim in range(3):
            out[:, dim] = np.interp(target_idx, orig_idx, seq[:, dim])
        return out

    def _fetch_data(self, idx):
        data = super()._fetch_data(idx)
        pseudo_label = np.array(Image.open(self.pseudo_labels[0][idx])).astype(np.float32)
        data[f"pseudo_label"] = pseudo_label / 255

        if self.split == "train":
            data[f"image_aug"] = data[f"image"]
            data[f"label_aug"] = data[f"label"]
            data[f"pseudo_label_aug"] = data[f"pseudo_label"]

        # Get fixation data
        img_name = self.images[idx].split("\\")[-1]
        image_data = self.fixation_data[self.fixation_data['IMAGE'] == img_name]
        points = []
        for idx, row in enumerate(image_data.itertuples(), start=1):
            points.append((float(row.CURRENT_FIX_X), float(row.CURRENT_FIX_Y), float(row.CURRENT_FIX_DURATION)))
        points = np.array(points)

        if len(points) > self.max_len:
            if self.strategy == "truncate":
                points_data = self.truncate_sequence(points, self.max_len)
            elif self.strategy == "interpolate":
                points_data = self.interpolate_sequence(points, self.max_len)
            else:
                raise ValueError("Unknown overflow_strategy")
        else:
            # len <= M: pad with zeros at end
            points_data = np.zeros((self.max_len, 3), dtype=np.float32)
            points_data[:len(points), :] = points
        stop_targets = torch.zeros(self.max_len, dtype=torch.float32)
        # if L > M and we used truncate, last real is M-1
        last_index = min(len(points), self.max_len) - 1
        stop_targets[last_index] = 1.0
        data[f"fixation"] = torch.from_numpy(points_data)
        data[f"stop_targets"] = stop_targets

        return data


if __name__ == "__main__":
    def save_img(img, name):
        img = img.permute(1, 2, 0).numpy()
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = (img * 255).astype(np.uint8)
        img = PIL.Image.fromarray(img)
        img.save(name)


    def save_label(img, name):
        img = img[0]
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = (img * 255).astype(np.uint8)
        img = PIL.Image.fromarray(img)
        img.save(name)


    dataset = KvasirGazeDataset(
                root="Kvasir-SEG/",
                pseudo_mask_root=os.path.join("Kvasir-SEG/", "gaze"),
                fixation_path=os.path.join("Kvasir-SEG/", "kvasir_fixation.csv"),
                split='train',
                max_len=20,
                strategy='interpolate',
                spatial_size=224,
                do_augmentation=True,
                resize_label=True,
                size_rate=1,
            )
    for i in range(len(dataset)):
        data = dataset[i]
    # data = dataset[0]
    # print(data["image"].shape, data["image_aug"].shape)
    # print(data["label"].shape, data["label_aug"].shape)
    # print(data["pseudo_label"].shape, data["pseudo_label_aug"].shape)

    # save_img(data["image"], "image.png")
    # save_label(data["label"], "label.png")
    # save_label(data["pseudo_label"], "pseudo_label.png")
    # save_img(data["image_aug"], "image_aug.png")
    # save_label(data["label_aug"], "label_aug.png")
    # save_label(data["pseudo_label_aug"], "pseudo_label_aug.png")

    # dataset = KvasirGazeDataset(
    #     root="Kvasir-SEG/",
    #     pseudo_mask_root=os.path.join("Kvasir-SEG/", "gaze"),
    #     fixation_path=os.path.join("Kvasir-SEG/", "kvasir_fixation.csv"),
    #     split='test',
    #     max_len=20,
    #     strategy='interpolate',
    #     spatial_size=224,
    #     do_augmentation=True,
    #     resize_label=False,
    #     size_rate=1,
    # )
    # data = dataset[0]
    # save_img(data["image"], "image.png")
    # save_label(data["label"], "label.png")