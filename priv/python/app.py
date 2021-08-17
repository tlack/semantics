import json
import os
import re
import time

import torch
from sentence_transformers import SentenceTransformer
# Used to create and store the Faiss index.
import faiss
import numpy as np
import pickle
import goose3
import scipy
import urllib.request
import logging

CUR_MODEL_OBJ = None
CUR_MODEL_NAME = ""

def unpack(str_):
    return str_.decode('utf-8')

def get_model(model_name):
    global CUR_MODEL_NAME, CUR_MODEL_OBJ
    if model_name != CUR_MODEL_NAME:
       CUR_MODEL_OBJ = SentenceTransformer(model_name)
       CUR_MODEL_NAME = model_name
    return CUR_MODEL_OBJ

def predict(model_name, text):
    text = unpack(text)
    model_name = unpack(model_name)
    print('loading',model_name)
    model = get_model(model_name)
    test_emb = model.encode([text], show_progress_bar=False)
    # print('emb', test_emb)
    return list([float(x) for x in test_emb[0]])


