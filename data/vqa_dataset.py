import os
import json
import random
import torchvision.transforms as T
from PIL import Image

import torch
from torch.utils.data import Dataset
from data.utils import pre_question

from torchvision.datasets.utils import download_url

class vqa_dataset(Dataset):
    def __init__(self, transform, ann_root, vqa_root, vg_root, train_files=[], split="train"):
        self.split = split        

        self.transform = transform
        self.vqa_root = vqa_root
        self.vg_root = vg_root
       
        if split=='train':
            urls = {'vqa_train':'https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_train.json',
                    'vqa_val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_val.json',
                    'vg_qa':'https://storage.googleapis.com/sfr-vision-language-research/datasets/vg_qa.json'}
        
            self.annotation = []
            for f in train_files:
                # download_url(urls[f],ann_root)
                self.annotation = json.load(open(os.path.join(ann_root,'%s.json'%f),'r'))
        else:
        #     download_url('https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_test.json',ann_root)
            self.annotation = json.load(open(os.path.join(ann_root,'evjvqa_warmup.json'),'r'))    
        #     download_url('https://storage.googleapis.com/sfr-vision-language-research/datasets/answer_list.json',ann_root)
            # self.answer_list = json.load(open(os.path.join(ann_root,'answer_list.json'),'r'))   
            self.answer_list = [self.annotation['annotations'][i]['answer'] for i in range(len(self.annotation['annotations']))]
        # self.answer_list = []
        
    def __len__(self):
        # return len(self.annotation)
        return len(self.annotation['annotations'])  
        # return 10  
    def __getitem__(self, index):    
        
        ann = self.annotation
        # if ann['dataset']!='evjvqa_warmup':
            
            # if ann['dataset']=='vqa':
            #     image_path = os.path.join(self.vqa_root,ann['image'])    
            # elif ann['dataset']=='vg':
            #     image_path = os.path.join(self.vg_root,ann['image'])  
                
            # image = Image.open(image_path).convert('RGB')   
            # image = self.transform(image)          
            
            # if self.split == 'test':
            #     question = pre_question(ann['question'])   
            #     question_id = ann['question_id']            
            #     return image, question, question_id


            # elif self.split=='train':                       
                
            #     question = pre_question(ann['question'])        
            #     # print(question)
            #     if ann['dataset']=='vqa':               
            #         answer_weight = {}
            #         for answer in ann['answer']:
            #             if answer in answer_weight.keys():
            #                 answer_weight[answer] += 1/len(ann['answer'])
            #             else:
            #                 answer_weight[answer] = 1/len(ann['answer'])

            #         answers = list(answer_weight.keys())
            #         weights = list(answer_weight.values())

            #     elif ann['dataset']=='vg':
            #         answers = [ann['answer']]
            #         weights = [0.2]  
                
            #     return image, question, answers, weights

# ---------- warmup data
        an = ann['annotations']
        # 
        resize = T.Resize(50)
        for img_inf in ann['images']:
            if(img_inf['id'] == an[index]['image_id']):
                image_path = os.path.join(self.vqa_root,img_inf['filename'])               
                image = Image.open(image_path).convert('RGB')  
                resized_img = resize(image)
                image = self.transform(resized_img)   
        question = pre_question(an[index]['question'])
        answers = [an[index]['answer']]
        # self.answer_list.append(an[index]['answer'])
        if self.split == 'test':
            question_id = an[index]['id']            
            return image, question, question_id
        elif self.split=='train':    
            weights = [0.2]      
            return image, question, answers, weights

        
def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n        

def isEnglish(question):
    if ("what" in question):
        return 1
    if ("who" in question):
        return 1
    if ("why" in question):
        return 1
    if ("where" in question):
        return 1
    if ("which" in question):
        return 1
    if ("how" in question):
        return 1
    if ("whom" in question):
        return 1
    if ("when" in question):
        return 1
    if ("whose" in question):
        return 1
    else: 
      return 0