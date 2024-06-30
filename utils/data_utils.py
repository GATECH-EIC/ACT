import json
import os
import time 
import sys
import argparse
import torch
import math
import torch.nn.functional as F
import copy
import numpy as np
from datasets import load_dataset

from tqdm import tqdm
from torch import nn
import random
import warnings
import transformers
from typing import List, Optional, Tuple, Union

choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

def get_classification_dataset(dataset_name: str):
    if dataset_name == "sst2":
        sst2_dataset = load_dataset('glue', 'sst2')
        train_dataset = sst2_dataset['train']
        val_dataset = sst2_dataset['validation']

        instruction =  "Classify the sentiment of the user's message into one of the following categories:'positive' or 'negative'.\n\n"
        template = "Sentence: {sentence} \nSentiment: "
        labels = [' negative', ' positive']

    elif dataset_name == "sst5":
        sst5_dataset = load_dataset('SetFit/sst5')
        train_dataset = sst5_dataset['train']
        val_dataset = sst5_dataset['validation']

        instruction =   "Classify the sentiment of the user's message into one of the following categories:'terrible', 'negative', 'neutral', 'positive', or 'great'.\n\n"
        template = "Sentence: {text} \nSentiment: "
        labels = [' terrible', " negative", " neutral", " positive", " great"]

    elif dataset_name == "MR":
        MR_dataset = load_dataset("rotten_tomatoes")
        train_dataset = MR_dataset['train']
        val_dataset = MR_dataset['validation']

        instruction = "Classify the sentiment of the movie's review into one of the following categories:'positive' or 'negative'.\n\n"
        template = "Review: {text} \nSentiment: "
        labels = [' negative', ' positive']

    elif dataset_name == "SUBJ":
        SUBJ_dataset = load_dataset("SetFit/subj")
        train_dataset = SUBJ_dataset['train']
        val_dataset = SUBJ_dataset['test']

        instruction =  "Classify the sentiment polarity of the movie's review into one of the following categories: 'subjective' or 'object'.\n\n"
        template = "Input: {text} \nType: "
        labels = [" objective", " subjective"]

    elif dataset_name == "DBPedia":
        DBPedia_dataset = load_dataset("dbpedia_14")
        train_dataset = DBPedia_dataset['train']
        val_dataset = DBPedia_dataset['test']

        instruction = "Classify the given text into one of the most relevant categories: 'company', 'school', 'artist', 'sport', 'politics', 'transportation', 'building', 'nature', 'village', 'animal', 'plant', 'album', 'film', or 'book'.\n\n"
        template = "Input: {content} \nType: "
        labels = [" company", " school", " artist", " sport", " politics", " transportation", " building", " nature", " village", " animal", " plant", " album", " film", " book"]

    elif dataset_name == "AGNews":
        AGNews_dataset = load_dataset('ag_news')
        train_dataset = AGNews_dataset['train']
        val_dataset = AGNews_dataset['test']

        instruction = "Classify the news articles into the categories of 'World', 'Sports', 'Business', or 'Technology'.\n\n"
        template = "Article: {text} \nCategory: "
        labels = [" World", " Sports", " Business", " Technology"]
    
    elif dataset_name == "TREC":
        TREC_dataset = load_dataset('trec')
        train_dataset = TREC_dataset['train']
        val_dataset = TREC_dataset['test']

        instruction =  "Classify the given questions into the following categories of 'Description', 'Entity', 'Expression', 'Person', 'Number', or 'Location'.\n\n"
        template = "Question: {text} \nType: "
        labels = [" Description", " Entity", " Expression", " Person", " Number", " Location"]
    
    elif dataset_name == "CB":
        CB_dataset = load_dataset('super_glue', 'cb')
        train_dataset = CB_dataset['train']
        val_dataset = CB_dataset['validation']

        instruction = "Read the following paragraph and determine if the hypothesis is true.\n\n"
        template = "Premise: {premise} Hypothesis: {hypothesis}. Answer: "
        labels = ["Yes", "No", "Maybe"]

    elif dataset_name == "BoolQ":
        BoolQ_dataset = load_dataset('super_glue', 'boolq')
        train_dataset = BoolQ_dataset['train']
        val_dataset = BoolQ_dataset['validation']

        instruction = "Read the text and answer the question by True or False.\n\n"
        template = "Text: {passage} Question: {question}? \nAnswer: "
        labels = [" False", " True"]
    else:
        raise NotImplementedError
    
    return {
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "instruction": instruction,
        "template": template,
        "label_choices": labels,
        "name": dataset_name
    }

def get_multiple_choice_dataset(dataset_name: str):
    if dataset_name == "hellaswag":
        hellaswag_dataset = load_dataset('hellaswag')
        train_dataset = hellaswag_dataset['train']
        val_dataset = hellaswag_dataset['validation']
        instruction =  "Complete the following sentence with an appropriate ending."
    elif dataset_name == "ARCE":
        ARCE_dataset = load_dataset('ai2_arc', 'ARC-Easy')
        train_dataset = ARCE_dataset['train']
        val_dataset = ARCE_dataset['validation']
        instruction = "Generate the correct answer to the following question."
    elif dataset_name == "ARCC":
        ARCC_dataset = load_dataset('ai2_arc', 'ARC-Challenge')
        train_dataset = ARCC_dataset['train']
        val_dataset = ARCC_dataset['validation']
        instruction = "Generate the correct answer to the following question."
    elif dataset_name == "PIQA":
        PIQA_dataset = load_dataset('piqa')
        train_dataset = PIQA_dataset['train']
        val_dataset = PIQA_dataset['validation']
        instruction = "Generate the correct solution to accomplish the following goal."
    elif dataset_name == "OB":
        OB_dataset = load_dataset('openbookqa', 'main')
        train_dataset = OB_dataset['train']
        val_dataset = OB_dataset['validation']
        instruction = "Generate the most appropriate answer to the following question."
    elif dataset_name == "COPA":
        COPA_dataset = load_dataset('super_glue', 'copa')
        train_dataset = COPA_dataset['train']
        val_dataset = COPA_dataset['validation']
        instruction = "Generate the correct answer to the following question."
    elif dataset_name == "CQA":
        CQA_dataset = load_dataset("commonsense_qa")
        train_dataset = CQA_dataset['train']
        val_dataset = CQA_dataset['validation']
        instruction = "Generate the correct answer to the following question."
    else:
        NotImplementedError
    
    return {
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "instruction": instruction,
        "name": dataset_name
    }

def form_choices(sample, dataset_name, few_shot=False):
    if dataset_name == 'hellaswag':
        question = sample['ctx']
        choice_texts = sample['endings']
        prompt = question
        for i, choice in enumerate(choice_texts):
            prompt += '\n' + choices[i] + '. ' + choice
        prompt += "\nAnswer:"
        if few_shot:
            prompt += " "+choices[int(sample['label'])]
        label = sample['label']
    elif dataset_name == 'ARCE':
        question = sample['question']
        choice_texts = sample['choices']['text']
        prompt = question
        for i, choice in enumerate(choice_texts):
            prompt += '\n' + choices[i] + '. ' + choice
        prompt += "\nAnswer:"
        if few_shot:
            prompt += " " + sample["answerKey"]
        if sample["answerKey"] == 'A':
            label = 0
        elif sample["answerKey"] == 'B':
            label = 1
        elif sample["answerKey"] == 'C':
            label = 2
        elif sample["answerKey"] == 'D':
            label = 3
        elif sample["answerKey"] == 'E':
            label = 4
        else:
            label = int(sample["answerKey"]) - 1
    elif dataset_name == "PIQA":
        question = sample['goal']
        sol1 = sample['sol1']
        sol2 = sample['sol2']
        prompt = question + '\n' + choices[0] + '. ' + sol1 + '\n' + choices[1] + '. ' + sol2 + "\nAnswer:"
        if few_shot:
            prompt += " "+choices[int(sample['label'])]
        label = sample['label']
    elif dataset_name == "OB":
        question = sample['question_stem']
        choice_texts = sample['choices']['text']
        prompt = question
        for i, choice in enumerate(choice_texts):
            prompt += '\n' + choices[i] + '. ' + choice
        prompt += "\nAnswer:"
        if few_shot:
            prompt += " " + sample["answerKey"] 
        if sample["answerKey"] == 'A':
            label = 0
        elif sample["answerKey"] == 'B':
            label = 1
        elif sample["answerKey"] == 'C':
            label = 2
        elif sample["answerKey"] == 'D':
            label = 3
    elif dataset_name == "ARCC":
        question = sample['question']
        choice_texts = sample['choices']['text']
        prompt = question
        for i, choice in enumerate(choice_texts):
            prompt += '\n' + choices[i] + '. ' + choice
        prompt += "\nAnswer:"
        if few_shot:
            prompt += " " + sample["answerKey"] 
        # print(sample["answerKey"])
        if sample["answerKey"] == 'A':
            label = 0
        elif sample["answerKey"] == 'B':
            label = 1
        elif sample["answerKey"] == 'C':
            label = 2
        elif sample["answerKey"] == 'D':
            label = 3
        elif sample["answerKey"] == 'E':
            label = 4
        else:
            label = int(sample["answerKey"]) - 1
    elif dataset_name == "COPA":
        premise = sample['premise']
        question = sample['question']
        choice1 = sample['choice1']
        choice2 = sample['choice2']
        prompt = premise + " What is the " +  question + "?" + '\n' + choices[0] + '. ' + choice1 + '\n' + choices[1] + '. ' + choice2  + "\nAnswer:"
        if few_shot:
            prompt += " "+choices[int(sample['label'])]
        label = sample['label']
    elif dataset_name == "CQA":
        question = sample['question']
        choice_texts = sample['choices']['text']
        prompt = question
        for i, choice in enumerate(choice_texts):
            prompt += '\n' + choices[i] + '. ' + choice
        prompt += "\nAnswer:"
        if few_shot:
            prompt += " " + sample["answerKey"] 
        label = choices.index(sample['answerKey'])

    return prompt, label

def get_calibration_single_classification_dataset(dataset, sample_num: int):
    if len(dataset['train_dataset']) < sample_num:
       sample_num = len(dataset['train_dataset'])
       print(f"The number of samples in {dataset['name']} dataset should be less than {len(dataset['train_dataset'])}")
    train_dataset = random.sample(list(dataset['train_dataset']), sample_num)
    instruction = dataset['instruction']
    template = dataset['template']
    label_choices = dataset['label_choices']
    dataset_name = dataset['name']
    label_key = 'coarse_label' if dataset_name == 'TREC' else 'label'

    def format_sample(sample):
        formatted_prompt = instruction + template.format(**sample)
        sample['sentence'] = formatted_prompt
        sample['label_choices'] = label_choices
        sample['name'] = dataset_name
        sample['label'] = sample[label_key]
        
        return sample

    calibration_dataset = list(map(format_sample, train_dataset))

    return calibration_dataset

def get_calibration_single_mc_dataset(dataset, sample_num: int):
    if len(dataset['train_dataset']) < sample_num:
        sample_num = len(dataset['train_dataset'])
        print(f"The number of samples in {dataset['name']} dataset should be less than {len(dataset['train_dataset'])}")
    train_dataset = random.sample(list(dataset['train_dataset']), sample_num)
    instruction = dataset['instruction']
    dataset_name = dataset['name']

    update_dataset = list()
    for sample in train_dataset:
        update_sample = dict()
        question_prompt, label = form_choices(sample, dataset_name, few_shot=False)
        final_prompt = instruction + '\n\n' + question_prompt

        update_sample['sentence'] = final_prompt
        update_sample['label'] = label
        update_sample['name'] = dataset_name
        update_dataset.append(update_sample)

    return update_dataset

def get_calibration_dataset(dataset_type: str, dataset_names: list, sample_num: int):
    comb_calibration_dataset = []
    for name in dataset_names:
        if dataset_type == "classification":
            dataset = get_classification_dataset(name)
            calibration_dataset = get_calibration_single_classification_dataset(dataset, sample_num=sample_num)
        elif dataset_type == "multiple_choice":
            dataset = get_multiple_choice_dataset(name)
            calibration_dataset = get_calibration_single_mc_dataset(dataset, sample_num=sample_num)
        comb_calibration_dataset += calibration_dataset
    random.shuffle(comb_calibration_dataset)

    return comb_calibration_dataset

def get_few_shot_samples(dataset, sample_num: int):
    train_dataset = dataset['train_dataset']
    few_shot_samples = random.sample(list(train_dataset), sample_num)

    template = dataset['template']
    label_choices = dataset['label_choices']
    dataset_name = dataset['name']

    few_shot_string = ''
    for sample in few_shot_samples:
        if dataset_name == 'TREC':
            label = sample['coarse_label']
        else:
            label = sample['label']
        sample = template.format(**sample)
        few_shot_string += sample + label_choices[label] + '\n\n'
    
    return few_shot_string

def get_formatted_evaluation_classification_dataset(dataset, few_shot_num: int):
    few_shot_samples = get_few_shot_samples(dataset, few_shot_num)
    eval_dataset = dataset['eval_dataset']
    instruction = dataset['instruction']
    template = dataset['template']
    label_choices = dataset['label_choices']
    dataset_name = dataset['name']
    label_key = 'coarse_label' if dataset_name == 'TREC' else 'label'

    def format_sample(sample):
        formatted_prompt = instruction + few_shot_samples + template.format(**sample)
        sample['sentence'] = formatted_prompt
        sample['label_choices'] = label_choices
        sample['name'] = dataset_name
        sample['label'] = sample[label_key]
        return sample
    
    formatted_eval_dataset = list(map(format_sample, eval_dataset))

    return formatted_eval_dataset

def get_formatted_evaluation_mc_dataset(dataset, few_shot_num: int):
    update_dataset = list()
    eval_dataset = dataset["eval_dataset"]
    train_dataset = dataset["train_dataset"]
    dataset_name = dataset['name']
    few_shot_samples = random.sample(list(train_dataset), few_shot_num)
    converted_samples = ""
    instruction = dataset['instruction']
    for sample in few_shot_samples:
        question, label = form_choices(sample, dataset_name, few_shot=True)
        converted_samples += question + "\n\n"
    
    for sample in eval_dataset:
        update_sample = dict()
        question_prompt, label = form_choices(sample, dataset_name, few_shot=False)
        final_prompt = instruction + "\n\n" + converted_samples + question_prompt

        update_sample['sentence'] = final_prompt
        update_sample['label'] = label
        update_sample['name'] = dataset_name
        update_dataset.append(update_sample)
        
    return update_dataset

        











    
















    


            

        

        





