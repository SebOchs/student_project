import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from tqdm import trange
from transformers import T5ForConditionalGeneration, Adafactor, T5Tokenizer
import dataloading as dl
import torch.multiprocessing as mp
import os
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
model = T5ForConditionalGeneration.from_pretrained('t5-small')
model.cuda()
# Implement DDP
if torch.cuda.device_count() > 1:
    print("Running on ", torch.cuda.device_count(), " GPUs")
    device_ids = list(range(torch.cuda.device_count()))
    store = torch.distributed.FileStore("tmp/test", 2)
    dist.init_process_group(backend='nccl', rank=2, world_size=2, store=store)
    model = DDP(model, find_unused_parameters=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, sampler=train_sampler)
    val_loader = DataLoader(val_data, batch_size=1, num_workers=0, shuffle=False, sampler=val_sampler)
else:
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, num_workers=0, shuffle=False)

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
    label_truth = []
    predictions = []
    for batch in val_loader:
        batch = tuple(t.to(device) for t in batch)
        text, lab = batch
        with torch.no_grad():
            output = tokenizer.decode(model.generate(text)[0])
        lab_comp = tokenizer.decode(lab.to('cpu').numpy().squeeze())
        label_truth.append(lab_comp)
        predictions.append(output)
        if output == lab_comp:
            accuracy_point += 1

        validation_step += 1
    accuracy = accuracy_point/validation_step


    def f1(pr, tr, class_num):
        """
        Calculates F1 score for a given class
        :param pr: list of predicted values
        :param tr: list of actual values
        :param class_num: indicates class
        :return: f1 score of class_num for predicted and true values in pr, tr
        """

        # Filter lists by class
        pred = [x == class_num for x in pr]
        truth = [x == class_num for x in tr]
        mix = list(zip(pred, truth))
        # Find true positives, false positives and false negatives
        tp = mix.count((True, True))
        fp = mix.count((False, True))
        fn = mix.count((True, False))
        # Return f1 score, if conditions are met
        if tp == 0 and fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        if tp == 0 and fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        if recall == 0 and precision == 0:
            return 0
        else:
            return 2 * recall * precision / (recall + precision)


    def macro_f1(predictions, truth):
        """
        Calculates macro f1 score, where all classes have the same weight
        :param predictions: logits of model predictions
        :param truth: list of actual values
        :return: macro f1 between model predictions and actual values
        """

        f1_0 = f1(predictions, truth, "incorrect")
        f1_1 = f1(predictions, truth, "contradict")
        f1_2 = f1(predictions, truth, "correct")
        if np.sum([x == "contradict" for x in truth]) == 0:
            return (f1_0 + f1_2) / 2
        else:
            return (f1_0 + f1_1 + f1_2) / 3


    def weighted_f1(predictions, truth):
        """
        Calculates weighted f1 score, where all classes have different weights based on appearance
        :param predictions: logits of model predictions
        :param truth: list of actual values
        :return: weighted f1 between model predictions and actual values
        """

        weight_0 = np.sum([x == "incorrect" for x in truth])
        weight_1 = np.sum([x == "contradict" for x in truth])
        weight_2 = np.sum([x == "correct" for x in truth])
        f1_0 = f1(predictions, truth, "incorrect")
        f1_1 = f1(predictions, truth, "contradict")
        f1_2 = f1(predictions, truth, "correct")
        return (weight_0 * f1_0 + weight_1 * f1_1 + weight_2 * f1_2) / len(truth)

    print("Accuracy: ", accuracy)
    print("Macro-F1: ", macro_f1(predictions, label_truth))
    print("Weighted-F1: ", weighted_f1(predictions, label_truth))
    if accuracy/validation_step > tracker:
        torch.save(model.state_dict(), "t5_testing_"+str(i)+"_"+str(accuracy*10000)[:3]+'.pt')
        tracker = accuracy

