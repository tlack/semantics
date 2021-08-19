import json
import os
import re
import time

import torch
from sentence_transformers import SentenceTransformer
import numpy as np

import erlport

CUR_MODEL_OBJ = None
CUR_MODEL_NAME = ""

def unpack(str_):
    return str_.decode('utf-8') if type(str_) == bytes else str_

def load(model_name):
    global CUR_MODEL_NAME, CUR_MODEL_OBJ
    model_name = unpack(model_name)
    if model_name != CUR_MODEL_NAME:
       print('SEMANTICS: loading model',model_name)
       CUR_MODEL_OBJ = SentenceTransformer(model_name)
       CUR_MODEL_NAME = model_name
    return CUR_MODEL_OBJ

def predict_many(model, texts):
    text2 = [unpack(x) for x in texts]
    test_emb = model.encode(text2, show_progress_bar=False)
    return test_emb.tolist()

def predict_one(model, text):
    text = unpack(text)
    test_emb = model.encode([text], show_progress_bar=False)
    # print('emb', test_emb)
    return test_emb[0].tolist()

def predict(model_name, text):
    model = load(model_name)
    if type(text) == type([]):
        return predict_many(model, text)
    else:
        return predict_one(model, text)




