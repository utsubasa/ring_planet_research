import numpy as np
from typing import List, Dict, Tuple, Sequence
import os.path as osp
import glob
import datetime
from tqdm import trange,tqdm
import random
import torchvision
import string
import os
import struct
from PIL import Image
from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw
import shutil
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
import base64
import itertools

def preprocess_example(img: np.array) -> np.array:
    # TODO: Clip outer blank
    h_nonzero_idx, w_nonzero_idx, _ = np.where((255-img).astype(np.uint8))
    h_nonzero_idx_min, h_nonzero_idx_max = min(h_nonzero_idx), max(h_nonzero_idx)
    w_nonzero_idx_min, w_nonzero_idx_max = min(w_nonzero_idx), max(w_nonzero_idx)
    img = img[h_nonzero_idx_min:h_nonzero_idx_max+1, w_nonzero_idx_min:w_nonzero_idx_max]
    return img

class COCOAnnotMaker():
    def __init__(self, background_size: Tuple[int, int], preprocess, img_dir: str):
        self.background_size = background_size
        self.preprocess = preprocess
        self.img_dir = img_dir
        img_path_pattern = osp.join(self.img_dir, "[a-zA-Z]", "*.png")
        self.img_paths = sorted(glob.glob(img_path_pattern))
        self.p_list = self._calc_prob_per_charactor()

    def _calc_prob_per_charactor(self) -> List[float]:
        charactor_list = os.listdir('./eng_charactor/')
        charactor_list.sort()
        p_list = []
        for charactor in charactor_list[10:]:#0-9を対象外にするために[10:]
            data_num = len(os.listdir(f'./eng_charactor/{charactor}')) #それぞれのアルファベットのデータ数を種痘
            char_p_list = [(1/52)/data_num]*data_num #全アルファベットで登場確率が均等になるように確率を設定
            p_list.append(char_p_list)
        p_list = list(itertools.chain.from_iterable(p_list)) 

        return p_list

    def _img_iterator(self, iter_num: int, p_list: List[float]) -> Tuple[np.array, Tuple[str, int]]:
        for _ in range(iter_num):
            sampled_path = random.choices(self.img_paths, weights=p_list)[0]
            
            label = sampled_path.split("/")[-2] # TODO: split char can be chnage.
            char_img_id = osp.splitext(osp.basename(sampled_path))[0]
            img = cv2.imread(sampled_path)

            if self.preprocess is not None:
                img = self.preprocess(img)
                
            yield img, (label, char_img_id)

    def make_coco_annot(self, img_num: int):
        coco_json = dict()
        coco_json["info"] = self._setup_info()
        coco_json["licenses"] = self._setup_licenses()
        coco_json["images"], coco_json["annotations"] = self._setup_img_annot_pair(img_num)
        coco_json["categories"] = self._setup_categories()

        return coco_json

    def _setup_info(self):
        default_info = {
            "description": "Sample Dataset",
            "version": "1.0",
            "url": "",
            "year": 2022,
            "contributer": "Umetani, Sasaki",
            "date_created": str(datetime.date.today()),
        }
        return default_info

    def _setup_licenses(self):
        default_license = {
            "id": 1,
            "url": "",
            "name": "MIT License"
        }
        return [default_license]

    def _setup_categories(self) -> List[Dict]:
        get_id_name_dict = lambda label_id, name: {"id": label_id, "name": name, "supercategory": "alphabet"}
        categories = [get_id_name_dict(label_id, label) for label_id, label in INT2LABEL.items()]
        return categories

    def _setup_imgs(self) -> List[Dict]:
        return []

    def _setup_annots(self) -> List[Dict]:
        return []

    def _setup_img_annot_pair(self, img_num: int) -> Tuple[List, List]:
        img_info = []
        annot = []
        for _ in trange(img_num):
            each_img_info, each_annot = self._get_single_word_img_annot_pair()
            img_info.append(each_img_info)
            annot.extend(each_annot)
        
        return img_info, annot

    def _get_single_word_img_annot_pair(self) -> Tuple[List, List]:
        prev_bbox_right_top = np.zeros(2, dtype=int)
        char_len = random.randint(3, 10) # TODO: random range should be confirmed!
        word_img_h, word_img_w = self.background_size
        word_img = np.full((word_img_h, word_img_w, 3), 255, dtype=np.uint8)
        each_char_img_id_digits = 7
        img_id = 0
        img_info = {
            "license": 1,
            "height": word_img_h,
            "width": word_img_w,
            "date_captured": None,
        }
        annot = []
        
        for char_idx, (char_img, (label, char_img_id)) in enumerate(self._img_iterator(iter_num=char_len, p_list=self.p_list) ):
            if char_idx == char_len:
                break
            
            radix = (10**each_char_img_id_digits)**char_idx
            img_id += int(char_img_id) * radix
            
            h, w, _ = char_img.shape
            bbox_left_top = prev_bbox_right_top + np.array([0, random.randint(5, 15)], dtype=int) # TODO: random range should be confirmed!
            relative_origin_h, relative_origin_w = bbox_left_top
            
            word_img[relative_origin_h:relative_origin_h+h, relative_origin_w:relative_origin_w+w] = char_img
            
            prev_bbox_right_top = bbox_left_top + np.array([0, w], dtype=int)
            
            each_char_annot = {
                "id": int(time.time() * 10_000), # UNIX time
                "category_id": LABEL2INT[label],
                # "bbox": [int(relative_origin_h), int(relative_origin_w), int(h), int(w)],
                "bbox": [int(relative_origin_w), int(relative_origin_h), int(w), int(h)],
                "segmentation": [[]],
                "area": h*w,
                "iscrowd": 0,
            }

            annot.append(each_char_annot)
        
        img_info["id"] = img_id
        img_info["file_name"] = f"{img_id}.png"
        if not osp.exists("./images"):
            os.makedirs("./images")
        cv2.imwrite(f"images/{img_id}.png", word_img) # TODO: img save dir should be confirmed
        
        for i in range(len(annot)):
            annot[i]["image_id"] = img_id
        return img_info, annot
data_type = "test"
sample_img_file = f"./EMNIST/raw/emnist-byclass-{data_type}-images-idx3-ubyte"
sample_label_file = f"./EMNIST/raw/emnist-byclass-{data_type}-labels-idx1-ubyte"
IMG_SAVE_DIR = "./EMNIST/imgs"
INT2LABEL = {idx: label for idx, label in enumerate(string.digits + string.ascii_letters)}
LABEL2INT = {value: key for key, value in INT2LABEL.items()}
for label in INT2LABEL.values():
    each_char_save_dir = osp.join(IMG_SAVE_DIR, label)
    if not osp.exists(each_char_save_dir):
        print(f"make {each_char_save_dir}")
        os.makedirs(each_char_save_dir)
sprix_img_dir = "eng_charactor"
sprix_imgs = glob.glob(osp.join(sprix_img_dir, "*.png"))
for label in LABEL2INT.keys():
    label_wise_dir = osp.join(sprix_img_dir, label)
    if not osp.exists(label_wise_dir):
        os.makedirs(label_wise_dir)
        print(label_wise_dir)
for each_img_path in tqdm(sprix_imgs):
    file_name = osp.basename(each_img_path)
    char, db_id_with_ext = file_name.split("_")
    shutil.move(each_img_path, osp.join(sprix_img_dir, char, db_id_with_ext))
BACKGROUND_SIZE = (200, 200*5)
BACKGROUND_H, BACKGROUND_W = BACKGROUND_SIZE
character_img_dir = "/home/ubuntu/sp/vocabulary/sprix/sp-hand-written/english/coco/eng_charactor"
coco_annot_maker = COCOAnnotMaker(BACKGROUND_SIZE, preprocess_example, img_dir=character_img_dir)
train_annot_json = coco_annot_maker.make_coco_annot(5000)
test_annot_json = coco_annot_maker.make_coco_annot(1000)
with open("/home/ubuntu/sp/vocabulary/sprix/sp-hand-written/english/coco/annotations/train_annot.json", "w") as f:
    json.dump(train_annot_json, f, indent=4)
with open("/home/ubuntu/sp/vocabulary/sprix/sp-hand-written/english/coco/annotations/test_annot.json", "w") as f:
    json.dump(test_annot_json, f, indent=4)