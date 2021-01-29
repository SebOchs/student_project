import numpy as np
import datasets
from utils import *
sacrebleu = datasets.load_metric('sacrebleu')
rouge = datasets.load_metric('rouge')
meteor = datasets.load_metric('meteor')
bert_score = datasets.load_metric('bertscore')

data = np.load('nik2.npy', allow_pickle=True)
feedback_data = np.array([
    [x['Pred_Feed'] for x in data],
    [x['Gold_Feed'] for x in data]
    ])
label_data = np.array([
    [float(x['Pred_Score']) for x in data],
    [float(x['Gold_Score']) for x in data]
    ])

val_acc = np.sum(label_data[0] == label_data[1]) / label_data.shape[1]
val_weighted = weighted_f1(label_data[0], label_data[1])
val_macro = macro_f1(label_data[0], label_data[1])
sacrebleu_score = sacrebleu.compute(predictions=feedback_data[0],
                                    references=[[x] for x in feedback_data[1]])['score']
rouge_score = rouge.compute(predictions=feedback_data[0], references=feedback_data[1])['rouge2'].mid.fmeasure
meteor_score = meteor.compute(predictions=feedback_data[0], references=feedback_data[1])['meteor']
bert = bert_score.compute(predictions=feedback_data[0], references=feedback_data[1], lang='en', rescale_with_baseline=True)['f1'].mean().item()


print('Acc = {:.6f}, M-F1 = {:.6f}, W-F1 = {:.6f}, BLEU = {:.6f}, Rouge = {:.6f}, Meteor = {:.6f}, Bert Score = {:.6f}'
      .format(val_acc, val_macro, val_weighted, sacrebleu_score, rouge_score, meteor_score, bert))
