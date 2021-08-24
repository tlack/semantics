from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

A = ["dogs", "cats", "horses"]
B = ["like", "love", "adore"]
SCORE = 0.8

def generate():

    pairs = []
    
    for a in A:
        for b in B:
            pairs.append(ItemExample(texts=[a, b], label=SCORE))
    
    print('Retraining data:', pairs)
    return pairs

