import os
import xml.etree.ElementTree as et
from transformers import T5Tokenizer
import numpy as np

tokenizer = T5Tokenizer.from_pretrained('t5-base')
PATH = "datasets/raw/sciEntsBank_training"
array = []
files = os.listdir(PATH)
for file in files:
    root = et.parse(PATH + '/' + file).getroot()
    for ref_answer in root[1]:
        for stud_answer in root[2]:
            text = "asag: reference: " + ref_answer.text[:-1] + tokenizer.eos_token + " student: " + stud_answer.text[:-1]
            label = stud_answer.get('accuracy')
            if label == 'contradictory':
                label = 'contradict'
            array.append([tokenizer(text, padding='max_length').input_ids, tokenizer(label).input_ids])

np.save("datasets/preprocessed/sciEntsBank_train.npy", np.array(array), allow_pickle=True)