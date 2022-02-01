import pytesseract
import cv2
import json
import numpy as np
import regex as re


def read_line_from_json(json_file):
    with open(json_file, "r") as file:
        data  = json.load(file)
        comp  = data["company"].strip()
        total = data["total"].strip()
        addr  = data["address"].strip()
        date  = data["date"].strip()
    
    return comp, addr, total, date

def image_to_text(path, options = "--psm 4"):
    image = cv2.imread(path)
    text = pytesseract.image_to_string(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            config=options)
    
    return text

def pred_to_dict(text, classes, probabilities):
    dic = {"company": ("", 0), "date": ("", 0), "address": ("", 0), "total": ("", 0)}
    keys = list(dic.keys())

    seps = [0] + (np.nonzero(np.diff(classes))[0] + 1).tolist() + [len(classes)]
    for i in range(len(seps) - 1):
        pred_class = classes[seps[i]] - 1
        if pred_class == -1:
            continue

        new_key = keys[pred_class]
        new_prob = probabilities[seps[i] : seps[i + 1]].max()
        if new_prob > dic[new_key][1]:
            dic[new_key] = (text[seps[i] : seps[i + 1]], new_prob)

    return {k: re.sub(r"[\t\n]", " ", v[0].strip()) for k, v in dic.items()}

def extract_date(text):
    text = re.sub(r"[\t\n]", " ", text)
    
    patterns = ["[0-9]+-[0-9]+-[0-9]+", 
                '[0-9]+/[0-9]+/[0-9]+']
    
    months  = ["jan","fev","mar","apr","may","jun","jul", "aug", "sep", "oct", "nov", "dec"]
    patterns += ['[0-9]+.' + month + '.[0-9]+' for month in months]
    
    for pattern in patterns:
        res = re.findall(pattern, text, flags=re.IGNORECASE)
        if(len(res)):
            return res[0]
        
    return ''