import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

POPE_PATH = {
    "coco_random": "benchmark/pope/coco/coco_pope_random.json",
    "coco_popular": "benchmark/pope/coco/coco_pope_popular.json",
    "coco_adversarial": "benchmark/pope/coco/coco_pope_adversarial.json",
    "gpa_random": "benchmark/pope/gpa/gqa_pope_seem_random.json",
    "gpa_popular": "benchmark/pope/gpa/gqa_pope_seem_popular.json",
    "gpa_adversarial": "benchmark/pope/gpa/gqa_pope_seem_adversarial.json",
    "aokvqa_random": "benchmark/pope/aokvqa/aokvqa_pope_seem_random.json",
    "aokvqa_popular": "benchmark/pope/aokvqa/aokvqa_pope_seem_popular.json",
    "aokvqa_adversarial": "benchmark/pope/aokvqa/aokvqa_pope_seem_adversarial.json",
}

class POPEDataSet(Dataset):
    def __init__(self, pope_path, data_path):
        self.pope_path = pope_path
        self.data_path = data_path

        image_list, query_list, label_list = [], [], []
        for q in open(pope_path, 'r'):
            line = json.loads(q)
            image_list.append(line['image'])
            query_list.append(line['text'])
            label_list.append(line['label'])

        for i in range(len(label_list)):
            if label_list[i] == 'no':
                label_list[i] = 0
            else:
                label_list[i] = 1

        assert len(image_list) == len(query_list)
        assert len(image_list) == len(label_list)

        self.image_list = image_list
        self.query_list = query_list
        self.label_list = label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])
        query = self.query_list[index]
        label = self.label_list[index]

        return {"image": image_path, "query": query, "label": label}

class GQADataset(Dataset):
    def __init__(self, pope_path, ds):
        self.pope_path = pope_path
        self.ds_dict = {row['id']: row for row in ds}

        self.image_list, self.query_list, self.label_list = [], [], []

        for q in open(pope_path, 'r'):
            line = json.loads(q)
            self.image_list.append(line['image'])
            self.query_list.append(line['text'])
            self.label_list.append(0 if line['label'] == 'no' else 1)

        assert len(self.image_list) == len(self.query_list) == len(self.label_list)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        if image_path.endswith('.jpg'):
            image_path = image_path[:-4]
        
        row = self.ds_dict.get(image_path)

        return {"image": row['image'], "query": self.query_list[index], "label": self.label_list[index]}

def pope_metric(pred_list, label_list, path):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    precision = float(TP) / float(TP + FP) if (TP + FP) > 0 else 0.0
    recall = float(TP) / float(TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    output = (
        f"TP\tFP\tTN\tFN\n"
        f"{TP}\t{FP}\t{TN}\t{FN}\n"
        f"Accuracy: {acc}\n"
        f"Precision: {precision}\n"
        f"Recall: {recall}\n"
        f"F1 score: {f1}\n"
        f"Yes ratio: {yes_ratio}\n"
    )

    print(output)

    with open(path, "w") as f:
        f.write(output)



def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out:

        line = line.replace('.', '')
        line = line.replace(',', '')
        words = line.split(' ')
        if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
            pred_list.append(0)
        else:
            pred_list.append(1)
    
    return pred_list
