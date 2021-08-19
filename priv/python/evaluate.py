import torch
from sentence_transformers import SentenceTransformer, util
# Used to create and store the Faiss index.
import faiss
import numpy as np

import evaluate_data
import evaluate_model

def predict(model, text):
    test_emb = model.encode([text], show_progress_bar=False)
    # print('emb', test_emb)
    return test_emb[0]

def pair_similarity(model, a, b):
    emba = predict(model, a); embb = predict(model, b);
    return util.pytorch_cos_sim(emba, embb)

def test_example(model, test_name, test):
    close = test['close']
    far = test['far']
    good_sim = pair_similarity(model, close[0], close[1])
    bad_sim = pair_similarity(model, far[0], far[1])
    if good_sim < bad_sim:
        return [False, good_sim, close, bad_sim, far]
    else:
        return [True, good_sim, close, bad_sim, far]
    
def test_group(model_name, model, group_name, test_list):
    failures = []
    for test_name, test in test_list.items():
        result = test_example(model, test_name, test)
        [status, close_sim, close_pair, far_sim, far_pair] = result
        if not status:
            print(f'{model_name} - {group_name} - {test_name} - FAILED')
            print(f'{close_pair} -> {close_sim}')
            print(f'{far_pair} -> {far_sim}')
            failures.append([model_name, group_name, test_name, close_sim, close_pair, far_sim, far_pair])
    return failures

def load(model_name):
    return SentenceTransformer(model_name)

def start():
    model_failures = {}
    group_failures = {}
    for model_name in evaluate_model.MODEL:
        model = load(model_name)
        model_failures[model_name] = []
        for group_name, tests in evaluate_data.EVAL_GROUPS.items():
            group_failures[group_name] = []

            failures = test_group(model_name, model, group_name, tests)

            for fail in failures:
                model_failures[model_name].append(fail)
                group_failures[group_name].append(fail)
    
    print(f'model results:')
    for model_name in model_failures:
        f = len(model_failures[model_name])
        print(f'model: {model_name} failures: {f}')

start()
