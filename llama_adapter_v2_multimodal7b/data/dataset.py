import torch
import yaml
from torch.utils.data import Dataset
from PIL import Image
import json
import llama.utils
from llama import Tokenizer
import copy
import torchvision.transforms as transforms
import pandas as pd
import random
import cv2

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

# create data
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
                                 antialias=None),  # 3 is bicubic
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

# to evaluate
transform_test = transforms.Compose([
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
                                 antialias=None),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])

class FinetuneDataset(Dataset):
    def __init__(self, config_path, transform, max_words=30, tokenizer_path=None):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        ann = []
        for meta_path in self.config['META']:
            meta_l = json.load(open(meta_path))
            print(f"{meta_path}: len {len(meta_l)}")
            print('Meta_l:', meta_l) ##
            # ann += meta_l
            ann = meta_l
        self.ann = ann
        self.ann_idx_list = list(self.ann.keys())
        print('Ann:', self.ann) ##

        print(f"total length: {len(self)}")
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        print('Index:', index, '\n') ##
        data_item = self.ann[self.ann_idx_list[index]]
        print('Data_Item:', data_item) ##
        # print('Data_Item_Keys:', data_item.keys()) ##
        # if 'image' in data_item.keys():
        if data_item['image'] != None:
            # filename = data_item['image']
            # question = data_item['conversations'][0]['value']
            # answer = data_item['conversations'][1]['value']

            print('Image Item:', data_item['image'])
            filename = './ScienceQA_Data/' + data_item['split'] + '/' + str(self.ann_idx_list[index]) + '/' + data_item['image']
            question = data_item['question']
            hint = data_item['hint']
            choices = 'Your answer should be selected of this list: ' + str(data_item['choices'])
            answer = 'The answer is ' + data_item['choices'][data_item['answer']]
            print('Filename:', filename)
            # print('Raw Choices:', data_item['choices'])
            # print('Choices:', choices)
     
            image = cv2.imread(filename)
            image = Image.fromarray(image)
            image = self.transform(image)
            format_instruction = question + hint
            format_input = choices
        else:
            # format_instruction = data_item['instruction'],
            # format_input = data_item['input']
            # answer = data_item['output']
            image = torch.zeros(3, 224, 224)
            question = data_item['question']
            hint = data_item['hint']
            choices = 'Your answer should be selected of this list: ' + str(data_item['choices'])
            # print('Raw Choices:', data_item['choices'])
            # print('Choices:', choices)
            
            format_instruction = question + hint
            format_input = choices
            answer = 'The answer is ' + data_item['choices'][data_item['answer']]
        input1 = llama.utils.format_prompt(format_instruction, format_input)
        input2 = input1 + answer
        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()

        # print('Input2:', input2) ##
        # print('Labels:', labels) ##
        return input2, labels, input2_mask, image
    
class EvaluationDataset(Dataset):
    def __init__(self, config_path, transform, max_words=30, tokenizer_path=None):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        ann = []
        for meta_path in self.config['META']:
            meta_l = json.load(open(meta_path))
            print(f"{meta_path}: len {len(meta_l)}")
            # print('Meta_l:', meta_l) ##
            # ann += meta_l
            ann = meta_l
        self.ann = ann
        self.ann_idx_list = list(self.ann.keys())
        # print('Ann:', self.ann) ##

        print(f"total length: {len(self)}")
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # print('Index:', index, '\n') ##
        data_item = self.ann[self.ann_idx_list[index]]
        print('Index:', self.ann_idx_list[index]) 
        # print('Data_Item:', data_item) ##
        # print('Data_Item_Keys:', data_item.keys()) ##
        # if 'image' in data_item.keys():
        if data_item['image'] != None:
            # filename = data_item['image']
            # question = data_item['conversations'][0]['value']
            # answer = data_item['conversations'][1]['value']

            # print('Image Item:', data_item['image']) ##
            filename = './ScienceQA_Data/' + data_item['split'] + '/' + str(self.ann_idx_list[index]) + '/' + data_item['image']
            question = data_item['question']
            hint = data_item['hint']
            choices = 'Your answer should be selected of this list: ' + str(data_item['choices'])
            choices_list = data_item['choices']
            
            answer = 'The answer is ' + data_item['choices'][data_item['answer']]
            answer_idx = data_item['answer'] 
            # print('Filename:', filename) ##
            # print('Raw Choices:', data_item['choices']) ##
            # print('Choices:', choices) ##

            image = cv2.imread(filename)
            image = Image.fromarray(image)
            image = self.transform(image)
            format_instruction = question + hint
            format_input = choices
        else:
            # format_instruction = data_item['instruction'],
            # format_input = data_item['input']
            # answer = data_item['output']
            filename = ''
            image = torch.zeros(3, 224, 224)
            question = data_item['question']
            hint = data_item['hint']
            choices = 'Your answer should be selected of this list: ' + str(data_item['choices'])
            choices_list = data_item['choices']
            # print('Raw Choices:', data_item['choices'])
            # print('Choices:', choices)
            
            format_instruction = question + hint
            format_input = choices
            answer = 'The answer is ' + data_item['choices'][data_item['answer']]
            answer_idx = data_item['answer'] 
        input1_raw = llama.utils.format_prompt(format_instruction, format_input)
        input2_raw = input1_raw + answer
        input1 = torch.tensor(self.tokenizer.encode(input1_raw, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2_raw, bos=True, eos=True), dtype=torch.int64)

        # print('Input Prompts:', input1_raw)
        # print('Prompts Length:', input1.shape[0])

        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()

        # print('Input2:', input2) ##
        # print('Labels:', labels) ##
        return self.ann_idx_list[index], image, input1_raw, answer, answer_idx, choices_list


class PretrainDataset(Dataset):
    def __init__(self, config_path, transform, max_words=30, tokenizer_path=None):
        print(f"read dataset config from {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        print("DATASET CONFIG:")
        print(self.config)
        images, captions = [], []
        for meta_path in self.config['META']:
            images_this_meta, captions_this_meta = [], []
            for chunk in pd.read_csv(meta_path, sep='\t', lineterminator='\n', chunksize=10 ** 6):
                images_this_meta.extend(chunk['url'].tolist())
                captions_this_meta.extend(chunk['caption'].tolist())
            print(f"{meta_path}: len {len(images_this_meta)}")
            images.extend(images_this_meta)
            captions.extend(captions_this_meta)

        self.data_list = []
        for x, y in zip(images, captions):
            self.data_list.append({'url': x, 'caption': y})
        print(f"total length: {len(self)}")
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = Tokenizer(model_path=tokenizer_path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sample = self.data_list[index]
        image_path, caption = sample['url'], sample['caption']
        if isinstance(caption, list):
            caption = random.choice(caption)
        caption = str(caption)

        image = cv2.imread(image_path)
        image = Image.fromarray(image)
        image = self.transform(image)

        format_instruction = "Generate caption of this image"
        input1 = llama.utils.format_prompt(format_instruction, None)
        input2 = input1 + caption

        input1 = torch.tensor(self.tokenizer.encode(input1, bos=True, eos=False), dtype=torch.int64)
        input2 = torch.tensor(self.tokenizer.encode(input2, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - input2.shape[0]
        if padding > 0:
            input2 = torch.cat((input2, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            input2 = input2[:self.max_words]
        labels = copy.deepcopy(input2)
        labels[:len(input1)] = -1
        input2_mask = input2.ge(0)
        label_mask = labels.ge(0)
        input2[~input2_mask] = 0
        labels[~label_mask] = 0
        input2_mask = input2_mask.float()
        label_mask = label_mask.float()
        return input2, labels, input2_mask, image
