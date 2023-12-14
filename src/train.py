import pandas as pd
import json
import torch

from ocr_model import VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from utils import *
from data import *
from datasets import load_metric
from transformers import AdamW
from tqdm.notebook import tqdm

from transformers.integrations import is_deepspeed_zero3_enabled


df = pd.read_fwf('./data/gt_test.txt', header=None)
df.rename(columns={0: "file_name", 1: "text"}, inplace=True)
del df[2]
df['file_name'] = df['file_name'].apply(lambda x: x + 'g' if x.endswith('jp') else x)

train_loader, valid_loader = prepare_dataloaders(df, valid_size = 0.3, '../data/images/', batch_size = 4)

# Load the model from a path
state_dict = torch.load('../fine_tuned_models/ocr_v2.bin')

with open('config.json', 'r') as f:
    config = json.load(f)

config = VisionEncoderDecoderConfig(encoder = config['encoder'], decoder = config['decoder'])
model = VisionEncoderDecoderModel(config, encoder = encoder, decoder = decoder_model)

_load_state_dict_into_model(model, state_dict, '')

processor = get_processor()

cer_metric = load_metric("cer")

def compute_cer(pred_ids, label_ids):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(10):  
   # train
   model.train()
   train_loss = 0.0
   for batch in tqdm(train_loader):
      # get the inputs
      for k,v in batch.items():
        batch[k] = v.to(device)

      outputs = model(**batch)
      loss = outputs.loss
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      train_loss += loss.item()

   print(f"Loss after epoch {epoch}:", train_loss/len(train_loader))
    
   # evaluate
   model.eval()
   valid_cer = 0.0
   with torch.no_grad():
     for batch in tqdm(valid_loader):
       # run batch generation
       outputs = model.generate(batch["pixel_values"].to(device))
       # compute metrics
       cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
       valid_cer += cer 

   print("Validation CER:", valid_cer / len(valid_loader))

# change the path where you want to save
torch.save(model.state_dict(), '../fine_tuned_models/ocr_v3.pt')