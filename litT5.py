import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, ConcatDataset
from transformers import T5ForConditionalGeneration, Adafactor, T5Tokenizer
from utils import *
import dataloading as dl
import warnings
import datasets
import torch
from collections import Counter


sacrebleu = datasets.load_metric('sacrebleu')
rouge = datasets.load_metric('rouge')
meteor = datasets.load_metric('meteor')
# bert_score = datasets.load_metric('bertscore')
warnings.filterwarnings("ignore")


class LitFineT5(pl.LightningModule):

    def __init__(self, batch_size):
        super(LitFineT5, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.batch_size = batch_size
        data = dl.T5Dataset('datasets/preprocessed/kn1_train.npy')
        self.train_data, self.val_data = random_split(data, split(len(data)),
                                                      generator=torch.Generator().manual_seed(42))
        self.test_data = dl.T5Dataset('datasets/preprocessed/kn1_ua.npy')
        self.save_hyperparameters()

    def forward(self, tok_seq, attn_seq):
        return self.tokenizer.decode(self.model.generate(input_ids=tok_seq, attention_mask=attn_seq, min_length=11,
                                                         max_length=128)[0],
                                     skip_special_tokens=True)

    def training_step(self, batch, batch_idx):
        text, text_attn, answer, lab = batch
        return self.model(input_ids=text, attention_mask=text_attn, labels=answer)[0].mean()

    def validation_step(self, batch, batch_idx):
        text, text_attn, answer, lab = batch
        return {'prediction': self(text, text_attn),
                'truth': self.tokenizer.decode(answer.squeeze(), skip_special_tokens=True),
                'label': self.tokenizer.decode(lab.squeeze(), skip_special_tokens=True),
                'original': self.tokenizer.decode(text.squeeze(), skip_special_tokens=True),
                }

    def validation_epoch_end(self, outputs):
        # validation array, first entry are all full text predictions, second entry gold standard, third entry label
        # and fourth entry label prediction
        val_data = [[x['prediction'] for x in outputs], [x['truth'] for x in outputs],
                    [x['label'] for x in outputs], [x['prediction'].split(' ', 1)[0] for x in outputs]]
        pred = [x.split(' ', 2)[2] for x in val_data[0]]
        truth = [x.split(' ', 2)[2] for x in val_data[1]]
        text = [x['original'] for x in outputs]
        acc_data = np.array(val_data[2:])
        val_acc = np.sum(acc_data[0] == acc_data[1]) / acc_data.shape[1]
        val_weighted = weighted_f1(acc_data[1], acc_data[0])
        val_macro = macro_f1(acc_data[1], acc_data[0])
        sacrebleu_score = sacrebleu.compute(predictions=pred,
                                            references=[[x] for x in truth])['score']
        rouge_score = rouge.compute(predictions=pred, references=truth)['rouge2'].mid.fmeasure
        meteor_score = meteor.compute(predictions=pred, references=truth)['meteor']
        if len(acc_data[1]) > 0:
            mse_val, invalid = mse(acc_data[1], acc_data[0])
            self.log('my_metric', (sacrebleu_score / 100 + rouge_score + meteor_score) / 3 * val_macro * (1 - mse_val) *
                     (1 - invalid / len(acc_data[1])))
        else:
            print('\nInvalid mse')
            mse_val, invalid = 1
            self.log('my_metric', 0)
        self.log('bleu', sacrebleu_score)
        self.log('val_macro', val_macro)
        self.log('rouge', rouge_score)
        self.log('meteor', meteor_score)
        print('Acc = {:.4f}, M-F1 = {:.4f}, W-F1 = {:.4f}, MSE = {:.4f}, BLEU = {:.4f}, Rouge = {:.4f}, Meteor = {:.4f}'
              .format(val_acc, val_macro, val_weighted, mse_val, sacrebleu_score, rouge_score, meteor_score))

    def test_step(self, batch, batch_idx):
        text, text_attn, answer, lab = batch
        return {'prediction': self(text, text_attn),
                'truth': self.tokenizer.decode(answer.squeeze(), skip_special_tokens=True),
                'label': self.tokenizer.decode(lab.squeeze(), skip_special_tokens=True),
                'original': self.tokenizer.decode(text.squeeze(), skip_special_tokens=True),
                }

    def test_epoch_end(self, outputs):
        val_data = [[x['prediction'] for x in outputs], [x['truth'] for x in outputs], [x['original'] for x in outputs],
                    [x['label'] for x in outputs], [x['prediction'].split(' ', 1)[0] for x in outputs]]
        pred = [x.split(' ', 2)[2] for x in val_data[0]]
        truth = [x.split(' ', 2)[2] for x in val_data[1]]
        acc_data = np.array(val_data[3:])
        val_acc = np.sum(acc_data[0] == acc_data[1]) / acc_data.shape[1]
        val_weighted = weighted_f1(acc_data[1], acc_data[0])
        val_macro = macro_f1(acc_data[1], acc_data[0])
        sacrebleu_score = sacrebleu.compute(predictions=pred,
                                            references=[[x] for x in truth])['score']
        rouge_score = rouge.compute(predictions=pred, references=truth)['rouge2'].mid.fmeasure
        meteor_score = meteor.compute(predictions=pred, references=truth)['meteor']
        if len(acc_data[1]) > 0:
            mse_val, invalid = mse(acc_data[1], acc_data[0])
            self.log('mse', mse_val)
        else:
            print('\nInvalid mse')
            self.log('mse', 0)
        self.log('bleu', sacrebleu_score)
        self.log('macro_f1', val_macro)
        self.log('rouge', rouge_score)
        self.log('meteor', meteor_score)
        self.log('acc', val_acc)
        self.log('weighted', val_weighted)
        print('Acc = {:.4f}, M-F1 = {:.4f}, W-F1 = {:.4f}, MSE = {:.4f}, BLEU = {:.4f}, Rouge = {:.4f}, Meteor = {:.4f}'
              .format(val_acc, val_macro, val_weighted, mse_val, sacrebleu_score, rouge_score, meteor_score))
        np.save('kn1_data_for_bertscore.npy', np.array(val_data[:3]), allow_pickle=True)


    def configure_optimizers(self):
        return Adafactor(self.model.parameters(), lr=None, warmup_init=True, relative_step=True)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=0, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1, num_workers=0, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, num_workers=0, shuffle=False)


class LitAsagFineT5(pl.LightningModule):

    def __init__(self, batch_size):
        super(LitAsagFineT5, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.batch_size = batch_size
        data = dl.T5Dataset('datasets/preprocessed/asag_kn1_train.npy')
        self.train_data, self.val_data = random_split(data, split(len(data)),
                                                      generator=torch.Generator().manual_seed(42))
        self.test_data = dl.T5Dataset('datasets/preprocessed/asag_kn1_ua.npy')
        self.save_hyperparameters()

    def forward(self, tok_seq, attn_seq):
        return self.tokenizer.decode(self.model.generate(input_ids=tok_seq, attention_mask=attn_seq, min_length=10,
                                                         max_length=128)[0],
                                     skip_special_tokens=True)

    def training_step(self, batch, batch_idx):
        text, text_attn, answer, lab = batch
        return self.model(input_ids=text, attention_mask=text_attn, labels=answer)[0].mean()

    def validation_step(self, batch, batch_idx):
        text, text_attn, answer, lab = batch
        return {'prediction': self(text, text_attn),
                'truth': self.tokenizer.decode(answer.squeeze(), skip_special_tokens=True),
                'label': self.tokenizer.decode(lab.squeeze(), skip_special_tokens=True),
                'original': self.tokenizer.decode(text.squeeze(), skip_special_tokens=True),
                }

    def validation_epoch_end(self, outputs):

        # validation array, first entry are all full text predictions, second entry gold standard, third entry label
        # and fourth entry label prediction
        val_data = [[x['prediction'] for x in outputs], [x['truth'] for x in outputs],
                    [x['label'] for x in outputs]]
        pred = extract_pred(val_data[0])
        truth = [x.split(':', 1)[1] for x in val_data[1]]
        text = [x['original'] for x in outputs]
        label_pred = extract_label(val_data[0])
        acc_data = np.array([val_data[2], label_pred])
        val_acc = np.sum(acc_data[0] == acc_data[1]) / acc_data.shape[1]
        val_weighted = weighted_f1(acc_data[1], acc_data[0])
        val_macro = macro_f1(acc_data[1], acc_data[0])
        sacrebleu_score = sacrebleu.compute(predictions=pred,
                                            references=[[x] for x in truth])['score']
        rouge_score = rouge.compute(predictions=pred, references=truth)['rouge2'].mid.fmeasure
        meteor_score = meteor.compute(predictions=pred, references=truth)['meteor']

        self.log('my_metric', (sacrebleu_score / 100 + rouge_score + meteor_score) / 3 * val_macro)

        self.log('bleu', sacrebleu_score)
        self.log('val_macro', val_macro)
        self.log('rouge', rouge_score)
        self.log('meteor', meteor_score)
        print('Acc = {:.4f}, M-F1 = {:.4f}, W-F1 = {:.4f}, BLEU = {:.4f}, Rouge = {:.4f}, Meteor = {:.4f}'
              .format(val_acc, val_macro, val_weighted, sacrebleu_score, rouge_score, meteor_score))

    def test_step(self, batch, batch_idx):
        text, text_attn, answer, lab = batch
        return {'prediction': self(text, text_attn),
                'truth': self.tokenizer.decode(answer.squeeze(), skip_special_tokens=True),
                'label': self.tokenizer.decode(lab.squeeze(), skip_special_tokens=True),
                'original': self.tokenizer.decode(text.squeeze(), skip_special_tokens=True),
                }

    def test_epoch_end(self, outputs):
        # validation array, first entry are all full text predictions, second entry gold standard, third entry label
        # and fourth entry label prediction
        val_data = [[x['prediction'] for x in outputs], [x['truth'] for x in outputs], [x['original'] for x in outputs],
                    [x['label'] for x in outputs]]
        pred = extract_pred(val_data[0])
        truth = [x.split(':', 1)[1] for x in val_data[1]]
        label_pred = extract_label(val_data[0])
        acc_data = np.array([val_data[3], label_pred])
        val_acc = np.sum(acc_data[0] == acc_data[1]) / acc_data.shape[1]
        val_weighted = weighted_f1(acc_data[1], acc_data[0])
        val_macro = macro_f1(acc_data[1], acc_data[0])
        sacrebleu_score = sacrebleu.compute(predictions=pred,
                                            references=[[x] for x in truth])['score']
        rouge_score = rouge.compute(predictions=pred, references=truth)['rouge2'].mid.fmeasure
        meteor_score = meteor.compute(predictions=pred, references=truth)['meteor']

        self.log('bleu', sacrebleu_score)
        self.log('macro_f1', val_macro)
        self.log('rouge', rouge_score)
        self.log('meteor', meteor_score)
        self.log('acc', val_acc)
        self.log('weighted', val_weighted)
        print('Acc = {:.4f}, M-F1 = {:.4f}, W-F1 = {:.4f}, MSE = {:.4f}, BLEU = {:.4f}, Rouge = {:.4f}, Meteor = {:.4f}'
              .format(val_acc, val_macro, val_weighted, mse_val, sacrebleu_score, rouge_score, meteor_score))
        np.save('kn1_data_for_bertscore.npy', np.array(val_data[:3]), allow_pickle=True)


    def configure_optimizers(self):
        return Adafactor(self.model.parameters(), lr=None, warmup_init=True, relative_step=True)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=0, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1, num_workers=0, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, num_workers=0, shuffle=False)


class LitMultiT5(pl.LightningModule):

    def __init__(self, batch_size, mode):
        super(LitMultiT5, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.batch_size = batch_size
        # multitasking:
        # datasets
        seb = dl.T5Dataset("datasets/preprocessed/sciEntsBank_train.npy")
        cose = dl.T5Dataset('datasets/preprocessed/cose_train.npy')
        glucose = dl.T5Dataset('datasets/preprocessed/glucose_train.npy')
        # SEB data set
        self.seb_train_data, self.seb_val_data = random_split(seb, split(len(seb)))
        self.seb_test_data = dl.T5Dataset("datasets/preprocessed/sciEntsBank_test_ua.npy")
        # ESNLI data set
        self.esnli_train_data = dl.T5Dataset('datasets/preprocessed/esnli_train.npy')
        self.esnli_val_data = dl.T5Dataset('datasets/preprocessed/esnli_val.npy')
        self.esnli_test_data = dl.T5Dataset('datasets/preprocessed/esnli_test.npy')
        # CosE data set
        self.cose_train_data, self.cose_val_data = random_split(cose, split(len(cose)))
        self.cose_test_data = dl.T5Dataset('datasets/preprocessed/cose_test.npy')
        # GLUCOSE data set
        self.glucose_train_data, self.glucose_val_data = random_split(glucose, split(len(glucose)))
        self.glucose_test_data = dl.T5Dataset('datasets/preprocessed/glucose_test.npy')
        # Combine data sets
        self.train_data = ConcatDataset([self.seb_train_data, self.esnli_train_data, self.cose_train_data,
                                         self.glucose_train_data])
        self.val_data = ConcatDataset([self.seb_val_data, self.esnli_val_data, self.cose_val_data,
                                       self.glucose_val_data])
        self.test_data = ConcatDataset([self.seb_test_data, self.esnli_test_data, self.cose_test_data,
                                        self.glucose_test_data])
        self.save_hyperparameters()

    def forward(self, tok_seq, attn_seq):
        return self.tokenizer.decode(self.model.generate(input_ids=tok_seq, attention_mask=attn_seq, min_length=11,
                                                         max_length=128)[0],
                                     skip_special_tokens=True)

    def training_step(self, batch, batch_idx):
        text, text_attn, answer, lab = batch
        return self.model(input_ids=text, attention_mask=text_attn, labels=answer)[0].mean()

    def validation_step(self, batch, batch_idx):
        text, text_attn, answer, lab = batch
        return {'prediction': self(text, text_attn),
                'truth': self.tokenizer.decode(answer.squeeze(), skip_special_tokens=True),
                'label': self.tokenizer.decode(lab.squeeze(), skip_special_tokens=True),
                'original': self.tokenizer.decode(text.squeeze(), skip_special_tokens=True),
                }

    def validation_epoch_end(self, outputs):
        # separate outputs
        val_data = [[x['prediction'] for x in outputs], [x['truth'] for x in outputs], [x['label'] for x in outputs]]
        text = [x['original'] for x in outputs]
        # separate outputs to tasks
        seb_val = np.array(sep_val(val_data, [i for i, j in enumerate(text) if 'asag:' in j]))
        esnli_val = np.array(sep_val(val_data, [i for i, j in enumerate(text) if ' esnli:' in j]))
        cose_val = np.array(sep_val(val_data, [i for i, j in enumerate(text) if ' cose:' in j]))
        glucose_val = np.array(sep_val(val_data, [i for i, j in enumerate(text) if ' glucose:' in j]))
        # validate seb
        seb_acc_stack = seb_val[:, :2]
        seb_acc = np.sum(seb_acc_stack[:, 0] == seb_acc_stack[:, 1]) / seb_acc_stack.shape[0]
        seb_m_f1 = macro_f1(seb_acc_stack[:, 0], seb_acc_stack[:, 1])
        seb_w_f1 = weighted_f1(seb_acc_stack[:, 0], seb_acc_stack[:, 1])

        print('\nSEB: Accuracy = {:.4f}, M-F1 = {:.4f}, W-F1 = {:.4f}'.format(seb_acc, seb_m_f1, seb_w_f1))
        # validate esnli
        esnli_acc_stack = esnli_val[:, [0, 2]]
        esnli_acc = np.sum(esnli_acc_stack[:, 0] == esnli_acc_stack[:, 1]) / esnli_acc_stack.shape[0]
        esnli_m_f1 = macro_f1(esnli_acc_stack[:, 0], esnli_acc_stack[:, 1])
        esnli_w_f1 = weighted_f1(esnli_acc_stack[:, 0], esnli_acc_stack[:, 1])
        esnli_bleu = sacrebleu.compute(predictions=esnli_val[:, 0], references=[[x] for x in esnli_val[:, 1]])
        # esnli_bert = bert_score.compute(predictions=esnli_val[:, 0], references=[[x] for x in esnli_val[:, 1]],
        #                                lang='en')
        print('ESNLI: Accuracy = {:.4f}, M-F1 = {:.4f}, W-F1 = {:.4f}, BLEU = {:.4f}'  # , bertscore = {:.4f}\n'
            .format(
            esnli_acc, esnli_m_f1, esnli_w_f1, esnli_bleu['score'],  # torch.mean(esnli_bert['f1']).item()
        ))
        # validate cose
        cose_acc_stack = cose_val[:, [0, 2]]
        cose_acc = np.sum(cose_acc_stack[:, 0] == cose_acc_stack[:, 1]) / esnli_acc_stack.shape[0]
        cose_bleu = sacrebleu.compute(predictions=cose_val[:, 0], references=[[x] for x in cose_val[:, 1]])
        # cose_bert = torch.mean(bert_score.compute(predictions=cose_val[:, 0], references=[[x] for x in cose_val[:,
        # 1]], lang='en')['f1']).item()
        print('Cos-E: Accuracy = {:.4f}, BLEU = {:.4f}'  # , bertscore = {:.4f}\n'
            .format(
            cose_acc, cose_bleu['score'],  # cose_bert
        ))
        # validate glucose

        glucose_bleu = sacrebleu.compute(predictions=glucose_val[:, 0], references=[[x] for x in glucose_val[:, 1]])
        # glucose_bert = torch.mean(bert_score.compute(predictions=glucose_val[:, 0],
        #                                        references=[[x] for x in glucose_val[:, 1]], lang='en')['f1']).item()
        print('GLUCOSE: BLEU = {:.4f}'  # , bertscore = {:.4f}\n'
              .format(glucose_bleu['score']
                      # , glucose_bert
                      ))

        self.log("val_macro", seb_m_f1)

    # prepared for later use
    def test_step(self, batch, batch_idx):
        text, text_attn, answer, lab = batch
        return {'prediction': self(text, text_attn),
                'truth': self.tokenizer.decode(answer.squeeze(), skip_special_tokens=True),
                'label': self.tokenizer.decode(lab.squeeze(), skip_special_tokens=True),
                'original': self.tokenizer.decode(text.squeeze(), skip_special_tokens=True),
                }

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
        self.log("test_macro", m_f1)
        self.log("test_weighted", w_f1)
        self.log("test_acc", acc)
        print("Accuracy: " + str(acc)[:6] + ", Macro-F1: " + str(m_f1)[:6] + ", Weighted-F1 " + str(w_f1)[:6])

    def configure_optimizers(self):
        return Adafactor(self.model.parameters(), lr=None, warmup_init=True, relative_step=True)

    def train_dataloader(self):
        train_length = len(self.seb_train_data)
        train_set = ConcatDataset([
            get_subset(self.esnli_train_data, train_length),
            get_subset(self.cose_train_data, train_length),
            get_subset(self.glucose_train_data, train_length),
            self.seb_train_data])
        return DataLoader(train_set, batch_size=self.batch_size, num_workers=0, shuffle=False)

    def val_dataloader(self):

        val_length = len(self.seb_val_data)
        val_set = ConcatDataset([
            get_subset(self.esnli_val_data, val_length),
            get_subset(self.cose_val_data, val_length),
            get_subset(self.glucose_val_data, val_length),
            self.seb_val_data
        ])
        return DataLoader(val_set, batch_size=1, num_workers=0, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, num_workers=0, shuffle=False)
