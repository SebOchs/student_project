import os
import xml.etree.ElementTree as et
from transformers import T5Tokenizer
import numpy as np
from datasets import load_dataset
import pandas
import random

tokenizer = T5Tokenizer.from_pretrained('t5-base')
random.seed(123)


def save(filepath, data):
    np.save(filepath + ".npy", np.array(data), allow_pickle=True)


def preprocessing_semeval(folder_path, file_path, tokenizer):
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
                array.append({
                    "input": tokenizer(text, max_length=128, padding='max_length').input_ids[:128],
                    "label": tokenizer(label).input_ids})
    save(file_path, array)


def preprocessing_esnli(file_path, tokenizer, mode):
    dataset = load_dataset("esnli")[mode]
    array = []

    for i in dataset:
        text = "justify: esnli: premise: " + i['premise'] + ' hypothesis: ' + i['hypothesis']
        answer = ['neutral', 'contradictory', 'entailment'][int(i['label'])] + ' explanation: '
        for j in [x for x in (i['explanation_1'], i['explanation_2'], i['explanation_3']) if len(x) > 0]:
            answer += j
            array.append({
                "input": tokenizer(text.lower()).input_ids,
                "answer": tokenizer(answer.lower()).input_ids,
                "label": answer.split(" explanation:", 1)[0]
            })
    save(file_path, array)


def preprocessing_cose(file_path, tokenizer, mode):
    dataset = load_dataset("cos_e", "v1.11")[mode]
    array = []

    for i in dataset:
        text = "justify: cose: question: " + i['question'] + ' '.join(' choice: ' + x for x in i['choices'])
        answer = i['answer'] + ' explanation: ' + i['abstractive_explanation']
        array.append({
            "input": tokenizer(text.lower()).input_ids,
            "answer": tokenizer(answer.lower()).input_ids,
            "label": answer.split(" explanation:", 1)[0]
        })
    save(file_path, array)


def preprocessing_glucose(folder_path, file_path, tokenizer):
    def remove_relation(specific_answer):
        def replacement(text, to_replace, replacements):
            return text.replace(to_replace, random.choice(replacements), 1).lower()

        if ">Causes/Enables>" in specific_answer:
            modified_answer = replacement(specific_answer, ">Causes/Enables>",
                                          ["and", ", so", ", which causes that", ", causing that"])
        elif ">Enables>":
            modified_answer = replacement(specific_answer, ">Enables>", ["and", ", which enables that", ", so"])
        elif ">Results in>" in specific_answer:
            modified_answer = replacement(specific_answer, ">Results in>", ["and", ", resulting in that", ", so"])
        elif ">Motivates>" in specific_answer:
            modified_answer = replacement(specific_answer, ">Motivates>", ["and", ", motivating that"])
        elif ">Causes>":
            modified_answer = replacement(specific_answer, ">Causes>", [", which causes that", ", causing that"])
        else:
            raise ValueError("No known relation in this sentence: " + specific_answer)
        return modified_answer.lower()

    dataset = pandas.read_csv(folder_path)
    array = []
    for i in range(len(dataset)):
        date = dataset.iloc[i]
        text = "justify: glucose: " + date.story.lower()
        usable_answers = [date[x] for x in [8, 12, 16, 20, 24, 28, 32, 36, 40, 44] if date[x] != 'escaped']
        for j in usable_answers:
            answer = "explanation: " + remove_relation(j)
            array.append({
                "input": tokenizer(text).input_ids,
                "answer": tokenizer(answer).input_ids
            })
    save(file_path, array)


# preprocessing_glucose('datasets/raw/GLUCOSE_training_data_final.csv', '', tokenizer)
# preprocessing_cose("", tokenizer, 'train')
# preprocessing_esnli("", tokenizer, 'train')
# preprocessing_semeval("datasets/raw/sciEntsBank_training", "datasets/preprocessed/sciEntsBank_train", tokenizer)
# preprocessing_semeval("datasets/raw/sciEntsBank_testing/test-unseen-answers", "datasets/preprocessed/sciEntsBank_test_ua", tokenizer)
# preprocessing_semeval("datasets/raw/sciEntsBank_testing/test-unseen-domains", "datasets/preprocessed/sciEntsBank_test_ud", tokenizer)
# preprocessing_semeval("datasets/raw/sciEntsBank_testing/test-unseen-questions", "datasets/preprocessed/sciEntsBank_test_uq", tokenizer)
