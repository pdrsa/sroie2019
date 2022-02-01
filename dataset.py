from torch.utils.data import Dataset
import os
from torchvision.io import read_image
from utils import image_to_text
from utils import read_line_from_json
from tqdm import tqdm
import os
import pickle
import json
from nltk.tokenize import word_tokenize
import torch

def createDataset():
    train_dataset = []
    for filename in tqdm(os.listdir('images')):
        try:
            txt = filename[:-4]+".txt"

            text              = image_to_text('images/' + filename)
            comp, addr, total, date = read_line_from_json("truth/"+filename[:-4]+".txt")

            fields = ["text", "company", "address", "total", "date"]
            values = [text, comp, addr, total, date]
            train_dataset.append(dict(zip(fields, values)))
        except:
            continue
            
    output_file = open("datasets/train_dataset.txt", 'w', encoding='utf-8')
    for dic in train_dataset:
        json.dump(dic, output_file) 
        output_file.write("\n")

class TrainDataset(Dataset):
    def __init__(self):
        self.VOCAB    = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n'
        self.receipts = list(torch.load('datasets/train.pth').values())
        
    def __len__(self):
        return len(self.receipts)

    def __getitem__(self, idx):
        receipt     = self.receipts[idx]
        text_tensor = torch.zeros(2048, dtype=torch.long)
        labels      = torch.zeros(2048, 5,   dtype=torch.float64)

        for i, char in enumerate(receipt[0]):
            text_tensor[i] = self.VOCAB.find(char)+1
            x = receipt[1][i]
            labels[i][x] = 1

        return text_tensor, labels, receipt[0]
    
class TestDataset(Dataset):
    def __init__(self):
        self.VOCAB        = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n'
        self.receipts     = list(torch.load('datasets/test.pth').items())
        
    def __len__(self):
        return len(self.receipts)

    def __getitem__(self, idx):
        receipt     = self.receipts[idx]
        text_tensor = torch.zeros(2048, dtype=torch.long)
        
        for i, char in enumerate(receipt[1]):
            text_tensor[i] = self.VOCAB.find(char)+1
        
        comp, addr, total, date = read_line_from_json("truth/"+ receipt[0] +".txt")
        labels = {"company": comp, "address": addr, "total": total, "date": date}
        
        return text_tensor, receipt[1], receipt[0]