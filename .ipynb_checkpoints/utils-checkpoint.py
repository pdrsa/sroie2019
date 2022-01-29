import pytesseract
import cv2
import json

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