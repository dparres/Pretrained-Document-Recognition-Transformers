import torch
import numpy as np
from PIL import Image
from typing import Any
from ast import literal_eval
from torch.utils.data import Dataset

import paths
from utils_ctc import sample_text_to_seq

######################################################
# Dataset Swin + CTC 
######################################################

class myDatasetCTC(Dataset):

    def __init__(self, partition = "train"):
        
        self.processor = None
        self.partition = partition

        self.path_labels = paths.IMAGE_PATH
        self.path_images = paths.GT_PATH
        self.image_name_list = []
        self.label_list = []

        f = open(self.path_labels, 'r')
        Lines = f.readlines()
    
        for line in Lines:
            line = line.strip().split()
            self.image_name_list.append(self.path_images + line[0])
            self.label_list.append(' '.join(line[1:]))

        print("\tSamples Loaded: ", len(self.label_list), "\n-------------------------------------")

    def set_processor(self, processor):
        self.processor = processor

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
      
        with Image.open(self.image_name_list[idx]) as image:
            image = image.convert("RGB")
            image_tensor = np.array(image)
        label = self.label_list[idx]

        image_tensor = self.processor(
            image_tensor, 
            random_padding=self.partitions == "train", 
            return_tensors="pt"
            ).pixel_values
        image_tensor = image_tensor.squeeze()
        
        # ctc
        label_tensor = torch.tensor(sample_text_to_seq(label, self.text_to_seq))
        
        return {"idx": idx, "img": image_tensor, "label": label_tensor, "raw_label": label}


######################################################
# Dataset Vision Encoder-Decoder (VED)
######################################################

class myDatasetTransformerDecoder(Dataset):
    def __init__(self, partition="train"):
        
        self.max_length = paths.MAX_LENGTH
        self.partition = partition
        self.processor = None
        self.ignore_id = -100

        self.path_img = paths.IMAGE_PATH
        self.path_transcriptions = paths.GT_PATH
        self.image_name_list = []
        self.label_list = []

        template = '{"gt_parse": {"text_sequence" : '
        with open(self.path_transcriptions, 'r') as file:
            for line in file:
                line = line.strip().split()

                image_name = line[0]
                label_gt = ' '.join(line[1:])
                label_gt = template + '"' + label_gt + '"' + "}}"

                self.image_name_list.append(self.path_img + image_name)
                self.label_list.append(label_gt)
        
        print("\tSamples Loaded: ", len(self.label_list))

    def dict2token(self, obj: Any):
        return obj["text_sequence"]
    
    def set_processor(self, processor):
        self.processor = processor

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        
        image = Image.open(self.image_name_list[idx]).convert("RGB")
        image_tensor = np.array(image)

        pixel_values: torch.Tensor = self.processor(image_tensor, random_padding=self.partition == "train", return_tensors="pt").pixel_values[0]

        label = self.label_list[idx]
        label = literal_eval(label)
        assert "gt_parse" in label and isinstance(label["gt_parse"], dict)
        gt_dicts = [label["gt_parse"]]
        target_sequence=[self.dict2token(gt_dict) + self.processor.tokenizer.eos_token for gt_dict in gt_dicts]
        
        input_ids = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id  # model doesn't need to predict pad token

        return {"idx": idx, "img": pixel_values, "label": labels, "raw_label": target_sequence}