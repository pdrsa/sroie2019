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
from string import ascii_lowercase, digits, punctuation
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
    def __init__(self, rebuild_data = False):
        if(rebuild_data): createDataset()
        
        self.VOCAB  = ascii_lowercase + digits + punctuation + " \t\n"

        self.train_dicts = []
        with open("datasets/train_dataset.txt", "r") as file:
            for line in file:
                self.train_dicts.append(json.loads(line))

    def __len__(self):
        return len(self.train_dicts)

    def __getitem__(self, idx):
        receipt     = self.train_dicts[idx]
        text_tensor = torch.zeros(2048, dtype=torch.long)
        labels      = torch.zeros(2048, 4,   dtype=torch.float64)

        for i, char in enumerate(receipt['text']):
            text_tensor[i] = self.VOCAB.find(char)+1

            # Label
            labels[i][3] = 1 # Indicates that this token means nothing
#             for j, key in enumerate(receipt.keys()):
#                 # The first field isn't a parameter
#                 if(key == "text"):
#                     continue

#                 # Found token in key
#                 if(receipt[key].lower().find(token) != -1):
#                     labels[i][3]   = 0
#                     labels[i][j-1] = 1
#                     break

        return text_tensor, labels, receipt