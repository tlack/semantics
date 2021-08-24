from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

import retrain_data_generator

model = SentenceTransformer("nli-roberta-base-v2")

train_examples = retrain_data_generator.generate()
print(f"got {len(train_examples)} examples")
print(train_examples[:10])

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
train_loss = losses.CosineSimilarityLoss(model)

# Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=2, warmup_steps=100, output_path="model_output")
