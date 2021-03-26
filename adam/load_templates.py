import cv2
import os
from scipy import signal
import numpy as np

def set_template_scores(templates):
    """set metadata or scores for templates, appropriately"""
    for k, v in templates.items():
        v["fruit"] = "fruit" in k #fruit increases score
        v["food"] = "food" in k #food decreases score
        
    return templates

def load_templates():

    #downsample factor
    fact = 6

    templates = {}
    for f in os.listdir("images"):
        img = cv2.imread(os.path.join("images", f))
        templates[f] = {}
        templates[f]['img'] = cv2.resize(img, (int(img.shape[0] / fact), int(img.shape[1] / fact)), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        templates[f]['img'] = templates[f]['img'] / (np.mean(templates[f]['img']) * templates[f]['img'].shape[0] * templates[f]['img'].shape[1]) #normalize by filter size and values
        templates[f]['img'] -= np.mean(templates[f]['img']) #zero-mean the filter

        if "food" in f:
            print(f)
            print(templates[f]['img'])
            print(templates[f]['img'].shape)
            print(np.mean(templates[f]['img']))

    templates = set_template_scores(templates)
    return templates

if __name__ == "__main__":
    load_templates()