import os
import PIL
import pandas as pd
import torch
from PIL import Image
import numpy as np
from .base_dataset import BaseImageDataset
import pydicom as dicom
from torchvision import transforms
from monai.transforms import (
    MapLabelValue,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    Resized,
    RandFlipd,
    ToTensord,
    Compose, RandRotated,
)

class NCIISBIProstateDataset(BaseImageDataset):
    def __init__(self, *argv, **kargs):
        super().__init__(*argv, **kargs)

        split_file = pd.read_csv(os.path.join(self.root, f"{self.split}.txt"), sep=" ", header=None)
        self.sample_list = split_file.iloc[:, 0].tolist()

        self.images = np.array([os.path.join(self.root, "images", f"{file}.dcm") for file in self.sample_list])[
            : int(len(self.sample_list) * self.size_rate)
        ]
        self.labels = np.array([os.path.join(self.root, "masks", f"{file}.png") for file in self.sample_list])[
            : int(len(self.sample_list) * self.size_rate)
        ]

    def _fetch_data(self, idx):
        image = dicom.dcmread(self.images[idx]).pixel_array.astype(np.float32)
        label = np.array(Image.open(self.labels[idx]).convert("L"), dtype=np.int16)

        return {"image": image, "label": label}

    def _transform_custom(self, data):
        data["label"] = (data["label"].float() / 255.0).long()
        if self.split == "train":
            data["label_aug"] = (data["label_aug"].float() / 255.0).long()
        return data

    def get_transform(self):
        resize_keys = ["image", "label"] if self.resize_label else ["image"]
        resize_mode = ["bilinear", "nearest"] if self.resize_label else ["bilinear"]

        if self.split == "train" and self.do_augmentation:
            return Compose(
                [
                    EnsureChannelFirstd(
                        keys=["image", "label"],
                        channel_dim="no_channel",
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                    ),
                    Resized(
                        keys=resize_keys,
                        spatial_size=self.spatial_size,
                        mode=resize_mode,
                    ),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    ToTensord(keys=["image", "label"]),
                ]
            )
        else:
            return Compose(
                [
                    EnsureChannelFirstd(
                        keys=["image", "label"],
                        channel_dim="no_channel",
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                    ),
                    Resized(
                        keys=resize_keys,
                        spatial_size=self.spatial_size,
                        mode=resize_mode,
                    ),
                    ToTensord(keys=["image", "label"]),
                ]
            )


class NCIISBIProstateGazeDataset(NCIISBIProstateDataset):
    def __init__(self, pseudo_mask_root, fixation_path, max_len, strategy, *argv, **kargs):
        super().__init__(*argv, **kargs)
        self.pseudo_mask_root = pseudo_mask_root
        self.pseudo_labels = [np.array([os.path.join(pseudo_mask_root, f"heatmap", f"{file}.png",) for file in self.sample_list])]
        self.fixation_data = pd.read_csv(fixation_path)
        self.max_len = max_len
        self.strategy = strategy

    def _transform_custom(self, data):
        data = super()._transform_custom(data)
        return data

    def get_transform(self):
        if self.split == "train" and self.do_augmentation:
            resize_keys = ["image", "image_aug", "label", "label_aug", "pseudo_label", "pseudo_label_aug"]
            resize_mode = ["bilinear", "bilinear", "nearest", "nearest", "bilinear", "bilinear"]
            return Compose(
                [
                    EnsureChannelFirstd(
                        keys=["image", "image_aug", "label", "label_aug", "pseudo_label", "pseudo_label_aug"],
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
                    ToTensord(keys=["image", "image_aug", "label", "label_aug", "pseudo_label", "pseudo_label_aug"]),
                ]
            )
        else:
            resize_keys = ["image", "label"] if self.resize_label else ["image"]
            resize_mode = ["bilinear", "nearest"] if self.resize_label else ["bilinear"]
            return Compose(
                [
                    EnsureChannelFirstd(
                        keys=["image", "label"],
                        channel_dim="no_channel",
                    ),
                    Resized(
                        keys=resize_keys,
                        spatial_size=self.spatial_size,
                        mode=resize_mode,
                    ),
                    ToTensord(keys=["image", "label"]),
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

        if self.split == "train":
            pseudo_label = np.array(Image.open(self.pseudo_labels[0][idx])).astype(np.float32)
            data[f"pseudo_label"] = pseudo_label / 255
            data[f"image_aug"] = data[f"image"]
            data[f"label_aug"] = data[f"label"]
            data[f"pseudo_label_aug"] = data[f"pseudo_label"]

        # Get fixation data
        img_name = self.images[idx].split("\\")[-1]
        image_data = self.fixation_data[self.fixation_data['IMAGE'] == img_name.split('.')[0]+'.jpg']
        points = []
        for idx, row in enumerate(image_data.itertuples(), start=1):
            points.append((float(row.CURRENT_FIX_X), float(row.CURRENT_FIX_Y), float(row.CURRENT_FIX_DURATION)))
        points = np.array(points)
        if len(points) == 0:
            points_data = np.zeros((self.max_len, 3), dtype=np.float32)
        elif len(points) > self.max_len:
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
        img = img[0]
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = (img * 255).astype(np.uint8)
        img = PIL.Image.fromarray(img)
        img.save(name)

    # dataset = NCIISBIProstateGazeDataset(
    #             root="NCI-ISBI-2013/",
    #             pseudo_mask_root=os.path.join("NCI-ISBI-2013/", "gaze"),
    #             fixation_path=os.path.join("NCI-ISBI-2013/", "nci-isbi_fixation.csv"),
    #             split='train',
    #             max_len=20,
    #             strategy='interpolate',
    #             spatial_size=224,
    #             do_augmentation=True,
    #             resize_label=True,
    #             size_rate=1,
    #         )
    # data = dataset[0]
    # print(data["image"].shape, data["image_aug"].shape)
    # print(data["label"].shape, data["label_aug"].shape)
    # print(data["pseudo_label"].shape, data["pseudo_label_aug"].shape)
    # for i in range(len(dataset)):
    #     data = dataset[i]
    # save_label(data["image"], "image.png")
    # save_label(data["label"], "label.png")
    # save_label(data["pseudo_label"], "pseudo_label.png")
    # save_label(data["image_aug"], "image_aug.png")
    # save_label(data["label_aug"], "label_aug.png")
    # save_label(data["pseudo_label_aug"], "pseudo_label_aug.png")


    dataset = NCIISBIProstateGazeDataset(
        root="NCI-ISBI-2013/",
        pseudo_mask_root=os.path.join("NCI-ISBI-2013/", "gaze"),
        fixation_path=os.path.join("NCI-ISBI-2013/", "nci-isbi_fixation.csv"),
        split='test',
        max_len=10,
        strategy='interpolate',
        spatial_size=224,
        do_augmentation=True,
        resize_label=False,
        size_rate=1,
    )

    for i in range(len(dataset)):
        data = dataset[i]
        print(data["image"].shape, data["label"].shape)
    # save_img(data["image"], "image.png")
    # save_img(data["label"], "label.png")




























