import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, RandomSampler
from transformers import T5ForConditionalGeneration, Adafactor, T5Tokenizer
from utils import macro_f1, weighted_f1
import dataloading as dl
import warnings
from pytorch_lightning.callbacks import ModelCheckpoint

warnings.filterwarnings("ignore")


class LitT5(pl.LightningModule):

    def __init__(self):
        super(LitT5, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.train_data, self.val_data = random_split(dl.SemEvalDataset("datasets/preprocessed/sciEntsBank_train.npy"),
                                                      [4472, 497], generator=torch.Generator().manual_seed(42))

    def forward(self, tok_seq):
        return self.tokenizer.decode(self.model.generate(tok_seq)[0])

    def training_step(self, batch, batch_idx):
        text, lab = batch
        return self.model(input_ids=text, labels=lab)[0].mean()

    def validation_step(self, batch, batch_idx):
        text, lab = batch
        return {'prediction': self(text), 'truth': self.tokenizer.decode(lab.squeeze())}

    def validation_epoch_end(self, outputs):
        pred = [x['prediction'] for x in outputs]
        lab = [x['truth'] for x in outputs]
        acc_stack = np.stack((pred, lab), axis=0)
        acc = np.sum([1 for i in range(len(acc_stack.T)) if acc_stack[0, i] == acc_stack[1, i]]) / len(acc_stack.T)
        m_f1 = macro_f1(pred, lab)
        w_f1 = weighted_f1(pred, lab)
        print("Accuracy: " + str(acc)[:6] + ", Macro-F1: " + str(m_f1)[:6] + ", Weighted-F1 " + str(w_f1)[:6])
        self.log("val_macro", m_f1)

    """
    # prepared for later use
    def test_step(self, batch, batch_idx):
        text, lab = batch
        return {'prediction:': self(text), 'truth': lab}

    def test_epoch_end(self, outputs):
        pred = [x['prediction'] for x in outputs]
        lab = [x['truth'] for x in outputs]
        acc_stack = np.stack((pred, lab), axis=0)
        acc = np.sum([1 for i in range(len(acc_stack.T)) if acc_stack[0, i] == acc_stack[1, i]]) / len(acc_stack.T)
        m_f1 = macro_f1(pred, lab)
        w_f1 = weighted_f1(pred, lab)
        print("Accuracy: " + str(acc)[:6])
        print("Macro-F1: " + str(m_f1)[:6])
        print("Weighted-F1 " + str(w_f1)[:6])
        return [acc, m_f1, w_f1]
    """

    def configure_optimizers(self):
        return Adafactor(self.model.parameters(), lr=None, warmup_init=True, relative_step=True)

    def train_dataloader(self):
        train_sampler = RandomSampler(self.train_data)
        return DataLoader(self.train_data, batch_size=4, num_workers=0, sampler=train_sampler)

    def val_dataloader(self):
        """
        val_sampler = RandomSampler(self.val_data)
        """
        return DataLoader(self.val_data, batch_size=1, num_workers=0, shuffle=False, sampler=None)


# Testing
checkpoint_callback = ModelCheckpoint(
    monitor="val_macro",
    mode="max",
    filepath='models/asag/{epoch}-{val_macro:.2f}',
    save_top_k=1
)
t5_test = LitT5()
trainer = pl.Trainer(
    gpus=2,
    num_nodes=1,
    distributed_backend='ddp',
    max_epochs=10,
    accumulate_grad_batches=2
)
trainer.fit(t5_test)
