import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import trange
from transformers import T5ForConditionalGeneration, Adafactor, T5Tokenizer
import dataloading as dl
path = "datasets/preprocessed/sciEntsBank_train.npy"


# Find GPU
device = torch.device("cuda")

# constants:
BATCH_SIZE = 4
EPOCHS = 32
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Initialize Model and Optimizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
if torch.cuda.device_count() > 1:
    print("Running on ", torch.cuda.device_count(), " GPUs")
    model = nn.DataParallel(model)
model.cuda()
optimizer = Adafactor(model.parameters(), lr=None, warmup_init=True, relative_step=True)

# Load data
train_data, val_data = random_split(dl.SemEvalDataset("datasets/preprocessed/sciEntsBank_train.npy"),
                                    [4472, 497], generator=torch.Generator().manual_seed(42))


train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
val_loader = DataLoader(val_data, batch_size=1, num_workers=0, shuffle=False)
train_loss = []
tracker = 0
for i in trange(EPOCHS, desc="Epoch "):

    # Training
    model.train()
    training_loss = 0
    training_step = 0

    for step, batch in enumerate(train_loader):
        # Load batch on gpu
        batch = tuple(t.to(device) for t in batch)
        text, lab = batch
        optimizer.zero_grad()
        outputs = model(input_ids=text, labels=lab)
        loss = outputs[0]
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        training_step += 1

    print("Train Loss: {}".format(training_loss / training_step))

    # Validation
    model.eval()
    accuracy = 0
    validation_step = 0
    for batch in val_loader:
        batch = tuple(t.to(device) for t in batch)
        text, lab = batch
        with torch.no_grad():
            output = tokenizer.decode(model.generate(text)[0])
        lab_comp = tokenizer.decode(lab.to('cpu').numpy().squeeze())
        if output == lab_comp:
            accuracy += 1

        validation_step += 1
    print("Accuracy: ", accuracy / validation_step)
    if accuracy/validation_step > tracker:
        torch.save(model.state_dict(), "t5_testing_"+str(i)+"_"+str(accuracy/validation_step*10000)[:2]+'.pt')
        tracker = accuracy/validation_step

