import os
import xml.etree.ElementTree as et
from transformers import T5Tokenizer
import numpy as np


def preprocessing(folder_path, file_path):
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    PATH = folder_path
    array = []
    files = os.listdir(PATH)
    for file in files:
        root = et.parse(PATH + '/' + file).getroot()
        for ref_answer in root[1]:
            for stud_answer in root[2]:
                text = "asag: reference: " + ref_answer.text[
                                             :-1] + tokenizer.eos_token + " student: " + stud_answer.text[:-1]
                label = stud_answer.get('accuracy')
                if label == 'contradictory':
                    label = 'contradict'
                array.append(
                    [tokenizer(text, max_length=128, padding='max_length').input_ids[:128], tokenizer(label).input_ids])

    np.save(file_path + ".npy", np.array(array), allow_pickle=True)


preprocessing("datasets/raw/sciEntsBank_training", "datasets/preprocessed/sciEntsBank_train")
preprocessing("datasets/raw/sciEntsBank_testing/test-unseen-answers", "datasets/preprocessed/sciEntsBank_test_ua")
preprocessing("datasets/raw/sciEntsBank_testing/test-unseen-domains", "datasets/preprocessed/sciEntsBank_test_ud")
preprocessing("datasets/raw/sciEntsBank_testing/test-unseen-questions", "datasets/preprocessed/sciEntsBank_test_uq")