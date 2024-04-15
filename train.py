import sys
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler

import utils_ctc
from models import Swin_CTC, VED
from mydatasets import myDatasetCTC, myDatasetTransformerDecoder

torch.set_float32_matmul_precision('medium')

#################################################################
# Experiment Settings
#################################################################

NUM_EPOCHS = int(sys.argv[0])
LR = float(sys.argv[1])
STRATEGY = str(sys.argv[2])
BATCH_SIZE = int(sys.argv[3])
MODEL_NAME = str(sys.argv[4])
NUM_ACCUMULATION_STEPS = int(sys.argv[5])

print(30*'*')
print("EXPERIMENT PARAMS: ")
print("\tNUM_EPOCHS: ", NUM_EPOCHS)
print("\tLR: ", LR)
print("\tSTRATEGY: ", STRATEGY)
print("\tBATCH_SIZE: ", BATCH_SIZE)
print("\tMODEL_NAME: ", MODEL_NAME)
print("\tNUM_ACCUMULATION_BATCHES: ", NUM_ACCUMULATION_STEPS)
print(30*'*')


#################################################################
# Load Torch Dataset and Create Vocab
#################################################################

l_of_transcrips = []
if MODEL_NAME == "Swin_CTC":
    train_dataset = myDatasetCTC(partition="train")
else:
    train_dataset = myDatasetTransformerDecoder(partition="train")

l_of_transcrips = train_dataset.label_list
text_to_seq, seq_to_text = utils_ctc.create_char_dicts(l_of_transcrips)

# update dics in datasets
train_dataset.text_to_seq = text_to_seq
train_dataset.seq_to_text = seq_to_text
print("Len dict text_to_seq: ", len(text_to_seq))
print("Len dict seq_to_text: ", len(seq_to_text))
print("Dict text_to_seq: ", (text_to_seq))
print("Dict seq_to_text: ", (seq_to_text))

#################################################################
# Load Model
#################################################################

# Create model
if MODEL_NAME == "Swin_CTC":
    model = Swin_CTC(len(text_to_seq))
else:
    model = VED()

#################################################################
# Training Settings
#################################################################

device = "cuda:0"

if MODEL_NAME == "Swin_CTC":
    mycollate_fn = utils_ctc.custom_collate
else:
    mycollate_fn = None

train_dataloader = DataLoader(
    train_dataset, 
    BATCH_SIZE, 
    shuffle=True, 
    num_workers=23, 
    collate_fn=mycollate_fn)

optimizer = AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0)

num_training_steps = NUM_EPOCHS # * len(train_dataloader)
lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

#################################################################
# Frozen Strategies
#################################################################

model.to(device)
model.train()

if MODEL_NAME == "Swin_CTC":
    if STRATEGY == "CTC-fclayer":
        for name_p,p in model.named_parameters():
            p.requires_grad = False
            if "projection_V" in name_p:
                p.requires_grad = True
                print("Train only: ", name_p)
    elif STRATEGY == "CTC-Swin":
        for name_p,p in model.named_parameters():
            p.requires_grad = True
            if "projection_V" in name_p:
                p.requires_grad = False
                print("No train: ", name_p)
    else:
        for name_p,p in model.named_parameters():
            p.requires_grad = True
        print("Train all layers")
else:
    if STRATEGY == "VED-encoder":
        for name_p,p in model.named_parameters():
            p.requires_grad = False
            if "model.encoder." in name_p:
                p.requires_grad = True
                print("Train only: ", name_p)
    elif STRATEGY == "VED-decoder":
        for name_p,p in model.named_parameters():
            p.requires_grad = False
            if "model.decoder." in name_p:
                p.requires_grad = True
                print("Train only: ", name_p)
    else:
        for name_p,p in model.named_parameters():
            p.requires_grad = True
        print("Train all layers")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Params: ", count_parameters(model))

#################################################################
# Training
#################################################################

for epoch in range(NUM_EPOCHS):

    epoch_loss = 0
    print("Epoch ", epoch)
    idx = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()

    with tqdm(iter(train_dataloader), desc="Training set", unit="batch") as tepoch:
        for batch in tepoch:
            
            inputs: torch.Tensor = batch["img"].to(device)
            labels: torch.Tensor = batch["label"].to(device)
            
            if MODEL_NAME == "Swin_CTC":
                target_lengths: torch.Tensor = batch["target_lengths"].to(device)
                outputs, loss = model(inputs, labels, target_lengths)
            else:
                outputs, loss = model(inputs, labels)

            loss.backward()

            if ((idx + 1) % NUM_ACCUMULATION_STEPS == 0):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            tepoch.set_postfix(loss=loss.data.item())
            epoch_loss += loss.data.item()
            idx += 1

# Save Final model 
torch.save(model.state_dict(), './FINAL_MODEL')