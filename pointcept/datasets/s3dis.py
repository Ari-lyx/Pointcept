"""
S3DIS Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import json
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .transform import Compose, TRANSFORMS


@DATASETS.register_module()
class S3DISDataset(Dataset):
    def __init__(
        self,
        split=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
        data_root="data/s3dis",
        transform=None,
        test_mode=False,
        test_cfg=None,
        cache=False,
        loop=1,
    ):
        super(S3DISDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def get_data_list(self):
        if isinstance(self.split, str):
            if self.split.endswith(".json"):
                json_path = os.path.join(self.data_root, self.split)
                if os.path.isfile(json_path):
                    with open(json_path, "r") as f:
                        room_list = json.load(f)
                    data_list = []
                    for room in room_list:
                        room_dir = os.path.join(self.data_root, room)
                        if os.path.isdir(room_dir):
                            data_list.append(room_dir)
                    return data_list
            # Look for room directories (new .npy format) instead of .pth files
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*"))
            data_list = [p for p in data_list if os.path.isdir(p)]
        elif isinstance(self.split, Sequence):
            data_list = []
            for sp in self.split:
                rooms = glob.glob(os.path.join(self.data_root, sp, "*"))
                data_list += [p for p in rooms if os.path.isdir(p)]
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        if not self.cache:
            # Load from .npy files in directory (newer format)
            data = {}
            for asset in os.listdir(data_path):
                if not asset.endswith(".npy"):
                    continue
                data[asset[:-4]] = np.load(os.path.join(data_path, asset))
        else:
            data_name = data_path.replace(os.path.dirname(self.data_root), "").split(
                "."
            )[0]
            cache_name = "pointcept" + data_name.replace(os.path.sep, "-")
            data = shared_dict(cache_name)
        name = self.get_data_name(idx)
        coord = data["coord"].astype(np.float32)
        color = data["color"].astype(np.float32)
        scene_id = data_path
        if "segment" in data.keys():
            segment = data["segment"].reshape([-1]).astype(np.int32)
        else:
            segment = np.ones(coord.shape[0], dtype=np.int32) * -1
        if "instance" in data.keys():
            instance = data["instance"].reshape([-1]).astype(np.int32)
        else:
            instance = np.ones(coord.shape[0], dtype=np.int32) * -1
        data_dict = dict(
            name=name,
            coord=coord,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
        )
        if "normal" in data.keys():
            data_dict["normal"] = data["normal"]
        return data_dict

    def get_data_name(self, idx):
        room_name = os.path.basename(self.data_list[idx % len(self.data_list)])
        remain, area_name = os.path.split(os.path.dirname(self.data_list[idx % len(self.data_list)]))
        return f"{area_name}-{room_name}"

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        segment = data_dict.pop("segment")
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        input_dict_list = []
        for data in data_dict_list:
            data_part_list = self.test_voxelize(data)
            for data_part in data_part_list:
                if self.test_crop:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                input_dict_list += data_part

        for i in range(len(input_dict_list)):
            input_dict_list[i] = self.post_transform(input_dict_list[i])
        data_dict = dict(
            fragment_list=input_dict_list, segment=segment, name=self.get_data_name(idx)
        )
        return data_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop
