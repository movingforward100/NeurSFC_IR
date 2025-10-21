import numpy as np
import os

from utils.quantize import quantize255
import subprocess
from hashlib import md5


# Modified from https://github.com/h-khalifa/python-LZW/

def LZW_encode(uncompressed):
 
    # Build the dictionary.
    # only big letters 
    # dict_size = 26
    # dictionary = {chr(i+ord('A')): i for i in range(dict_size)}

    dict_size = 256
    dictionary = {chr(i): i for i in range(dict_size)}
 
    p = ""
    output = []
    for c in uncompressed:
        temp = p + c
        if temp in dictionary:
            p = temp
        else:
            output.append(dictionary[p])
            # Add temp to the dictionary.
            dictionary[temp] = dict_size
            dict_size += 1
            p = c
 
    # Output the code for w.
    if len(p):
        output.append(dictionary[p])
    return output


def LZW_decode(compressed):
    
    # Build the dictionary.
    # dict_size = 26
    # dictionary = {i: chr(i+ord('A')) for i in range(dict_size)}
    dict_size = 256
    dictionary = {i: chr(i) for i in range(dict_size)}
    
 
    result = ""
    p = ""
    bol = False     
    for k in compressed:
       
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = p + p[0]
        
        result += (entry)
        
        # Add p+entry[0] to the dictionary unless it's first element
        if bol:
            dictionary[dict_size] = p + entry[0]
            dict_size += 1
 
        p = entry
        bol = True
    return result 


def get_lzw_length(params):
    img255, sfc = params

    # pdb.set_trace()
    sfc = np.asarray(sfc)
    
    if img255.ndim == 3: # RGB image
        assert img255.shape[-1] == 3
        # pdb.set_trace()
        img: np.ndarray  = quantize255(img255, get_centroids())
    else: # greyscale image
        img: np.ndarray = img255.astype(np.uint8)

    seq = img[sfc[:, 0], sfc[:, 1]]
    seq = ''.join([chr(i) for i in seq])
    
    cipher = LZW_encode(seq)
    return len(cipher)


def get_centroids():
    from utils.cfg import cfg_global
    centroids = None

    path = os.path.join(cfg_global.data_dir, 'train_crpo_64x64_skip_128_centroids.npy')
    if os.path.exists(path):
        centroids = np.load(path) / 255  ## the saved centriods is calculated in 0-255, we need to 255
    else:
        print('loading centroids.npy not successful')
    return centroids

