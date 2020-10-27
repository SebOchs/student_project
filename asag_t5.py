import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from tqdm import trange
from transformers import T5ForConditionalGeneration, Adafactor, T5Tokenizer
import dataloading as dl
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
path = "datasets/preprocessed/sciEntsBank_train.npy"


# Find GPU
device = torch.device("cuda")

# constants:
BATCH_SIZE = 4
EPOCHS = 8
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Load data
train_data, val_data = random_split(dl.SemEvalDataset("datasets/preprocessed/sciEntsBank_train.npy"),
                                    [4472, 497], generator=torch.Generator().manual_seed(42))
# Initialize Modeand Optimizer
model = T5ForConditionalGeneration.from_pretrained('t5-large')

# Implement DDP
if torch.cuda.device_count() > 1:
    print("Running on ", torch.cuda.device_count(), " GPUs")
    device_ids = list(range(torch.cuda.device_count()))
    dist.init_process_group(backend='nccl', rank=2)
    model = DDP(model, device_ids=device_ids, find_unused_parameters=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, sampler=train_sampler)
    val_loader = DataLoader(val_data, batch_size=1, num_workers=0, shuffle=False, sampler=val_sampler)
else:
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, num_workers=0, shuffle=False)

model.cuda()
# Optimizer
optimizer = Adafactor(model.parameters(), lr=None, warmup_init=True, relative_step=True)

# Begin Training
tracker = 0
for i in trange(EPOCHS, desc="Epoch "):

    # Training
    model.train()

    for step, batch in enumerate(train_loader):
        # Load batch on gpu
        batch = tuple(t.to(device) for t in batch)
        text, lab = batch
        optimizer.zero_grad()
        outputs = model(input_ids=text, labels=lab)
        loss = outputs[0]

        if torch.cuda.device_count() > 1:
            loss.mean().backward()
        else:
            loss.backward()
        optimizer.step()


    # Validation
    model.eval()
    accuracy_point = 0
    validation_step = 0
    for batch in val_loader:
        batch = tuple(t.to(device) for t in batch)
        text, lab = batch
        with torch.no_grad():
            output = tokenizer.decode(model.generate(text)[0])
        lab_comp = tokenizer.decode(lab.to('cpu').numpy().squeeze())
        if output == lab_comp:
            accuracy_point += 1

        validation_step += 1
    accuracy = accuracy_point/validation_step
    print("Accuracy: ", accuracy / validation_step)
    if accuracy/validation_step > tracker:
        torch.save(model.state_dict(), "t5_testing_"+str(i)+"_"+str(accuracy/validation_step*10000)[:2]+'.pt')
        tracker = accuracy/validation_step

