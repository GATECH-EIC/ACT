import json
import os
import time 
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
from transformers import LlamaTokenizer,AutoTokenizer,MistralForCausalLM
from models.modelling_llama import LlamaForCausalLM, LlamaAttention
from models.llama_modelling_aug import *
from models.mistral_modelling_aug import *
from utils.data_utils import *
from utils.data_utils import choices

tasks = {
    'classification': ['sst2', 'sst5', 'MR', 'SUBJ', 'AGNews', 'TREC', 'CB', 'BoolQ', 'DBPedia'],
    'multiple choice': ['hellaswag', 'ARCE', 'PIQA', 'ARCC', 'OB', 'COPA', 'CQA'],
}

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load(ckpt_dir, model_type):
    if 'llama' in model_type:
        tokenizer = LlamaTokenizer.from_pretrained(
            ckpt_dir,
            use_fast=False,
            cache_dir="../",
            padding_side="left",
        )
        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
        model = LlamaForCausalLM.from_pretrained(ckpt_dir, low_cpu_mem_usage = True,cache_dir="../", torch_dtype=torch.float16)
        model.half()
        model.to('cuda')
    elif "mistral" in model_type:
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
        model = MistralForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", low_cpu_mem_usage = True,cache_dir="../", torch_dtype=torch.float16)
        model.to('cuda')

    return model, tokenizer

def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts

def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')

    return input_tokens

def compute_metrics(args, results: dict, total_num: int) -> float:
    total_acc = 0
    accs = []
    for name, correct in results.items():
        if args.calibrate:
            if args.task_type == "classification":
                if name != "CB":
                    acc = correct / args.num_samples
                else:
                    acc = correct / 250
            elif args.task_type == "multiple_choice":
                if name != "COPA":
                    acc = correct / args.num_samples
                else:
                    acc = correct / 400
            accs.append(acc)
        else:
            acc = correct / total_num
        total_acc += correct
        print("ACC-%s: %.4f" % (name, acc))
    print("ACC-all: %.4f" % (total_acc/total_num))
    if args.calibrate:
        if args.task_type == "classification":
            with open(args.output_dir, 'a') as file:
                file.write("ACC-all: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n" % \
                        (total_acc/total_num, accs[0], accs[1], accs[2], accs[3], \
                            accs[4], accs[5], accs[6], accs[7], accs[8])) 
        elif args.task_type == "multiple_choice":
            with open(args.output_dir, 'a') as file:
                file.write("ACC-all: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n" % \
                        (total_acc/total_num, accs[0], accs[1], accs[2], accs[3], \
                            accs[4], accs[5], accs[6], accs[7], accs[8])) 
                
    return total_acc/total_num
            
def few_shot_eval(args, dataset, model, tokenizer, k=5):
    total_acc = 0
    for i in range(k):
        set_random_seed(42 * i)
        eval_dataset = get_formatted_evaluation_classification_dataset(dataset, args.few_shot_number)
        final_dataset_prompt = [sample['sentence'] for sample in eval_dataset]
        all_sample_dataset_names = [sample['name'] for sample in eval_dataset]
        correct_counts = {all_sample_dataset_names[0]: 0}

        print(final_dataset_prompt[0])
        batch_size = 1
        count = 0
        model.eval()
        all_input_ids = list()
        with torch.no_grad():
            for batch_input in tqdm(batch_split(final_dataset_prompt, batch_size)):
                choices = eval_dataset[count]['label_choices']
                answer = eval_dataset[count]['label']

                encoded_answer = tokenizer(choices, padding=True, return_tensors='pt')
                encoded_inputs = prepare_input(tokenizer, batch_input)
                all_input_ids.append(encoded_inputs)

                logits = model(**encoded_inputs).logits
                logits = logits[0][-1]
                all_logits = torch.log_softmax(logits[encoded_answer['input_ids'][:, -1].flatten()], dim=-1)
                preds = all_logits.argmax(dim=-1)
                correct_counts[all_sample_dataset_names[count]] += int(preds.item() == answer)
                count += 1
        total_acc += compute_metrics(args, correct_counts, total_num = len(eval_dataset))
    
    print("ACC-avg: %.4f" % (total_acc / k))

def few_shot_mc_eval(args, dataset, model, tokenizer, k=5):
    total_acc = 0
    for i in range(k):
        set_random_seed(42 * i)
        eval_dataset = get_formatted_evaluation_mc_dataset(dataset, args.few_shot_number)
        final_dataset_prompt = [sample['sentence'] for sample in eval_dataset]
        all_sample_dataset_names = [sample['name'] for sample in eval_dataset]
        correct_counts = {all_sample_dataset_names[0]: 0}

        print(final_dataset_prompt[0])
        batch_size = 1
        count = 0
        model.eval()
        with torch.no_grad():
            for batch_input in tqdm(batch_split(final_dataset_prompt, batch_size)):
                encode_inputs = prepare_input(tokenizer, batch_input)
                label = choices[int(eval_dataset[count]['label'])]
                outputs = model.generate(**encode_inputs, max_new_tokens = 1, pad_token_id = tokenizer.pad_token_id, \
                                          do_sample=False, num_beams=1)
                pred = tokenizer.batch_decode(outputs[0][-1].unsqueeze(dim=0), skip_special_tokens=True)[0]
                correct_counts[all_sample_dataset_names[count]] += str(pred) == str(label)
                count += 1

        total_acc += compute_metrics(args, correct_counts, total_num = len(eval_dataset))
    
    print("ACC-avg: %.4f" % (total_acc / k))

def main(args):
    set_random_seed(42)
    model, tokenizer = load(args.ckpt_dir, args.model_type)

    if args.calibrate:
        print("---------------------Calibration---------------------")
        calibration_dataset = get_calibration_dataset(args.task_type, tasks[args.task_type], args.num_samples)
        final_dataset_prompt = [sample['sentence'] for sample in calibration_dataset]
        final_dataset = calibration_dataset
        correct_counts = {key: 0 for key in tasks[args.task_type]}
    else:
        print("---------------------Eval---------------------")
        print(f"-------------------dataset:{args.dataset}-------------------")
        if args.task_type == "classification":
            dataset = get_classification_dataset(args.dataset)
            if args.few_shot_number > 0:
                start_time = time.time()
                few_shot_eval(args, dataset, model, tokenizer, args.k) 
                end_time = time.time()
                print("Total run time: %.2f" % (end_time - start_time))
                return 
            eval_dataset = get_formatted_evaluation_classification_dataset(dataset, args.few_shot_number)
        elif args.task_type == "multiple_choice":
            dataset = get_multiple_choice_dataset(args.dataset)
            if args.few_shot_number > 0:
                start_time = time.time()
                few_shot_mc_eval(args, dataset, model, tokenizer, args.k)
                end_time = time.time()
                print("Total run time: %.2f" % (end_time - start_time))
                return
            eval_dataset = get_formatted_evaluation_mc_dataset(dataset, args.few_shot_number)
        final_dataset_prompt = [sample['sentence'] for sample in eval_dataset]
        final_dataset = eval_dataset
        correct_counts = {eval_dataset[0]['name']: 0}

    all_sample_dataset_names = [sample['name'] for sample in final_dataset]

    print(final_dataset_prompt[0])
    start_time = time.time()
    batch_size = 1
    count = 0
    model.eval()
    all_input_ids = list()
    with torch.no_grad():
        if args.task_type == "classification":
            for batch_input in tqdm(batch_split(final_dataset_prompt, batch_size)):
                choices1 = final_dataset[count]['label_choices']
                answer = final_dataset[count]['label']

                encoded_answer = tokenizer(choices1, padding=True, return_tensors='pt')
                encoded_inputs = prepare_input(tokenizer, batch_input)
                all_input_ids.append(encoded_inputs)

                logits = model(**encoded_inputs).logits
                logits = logits[0][-1]
                all_logits = torch.log_softmax(logits[encoded_answer['input_ids'][:, -1].flatten()], dim=-1)
                preds = all_logits.argmax(dim=-1)
                correct_counts[all_sample_dataset_names[count]] += int(preds.item() == answer)
                count += 1
        elif args.task_type == "multiple_choice":
            answers = []
            for batch_input in tqdm(batch_split(final_dataset_prompt, batch_size)):
                encode_inputs = prepare_input(tokenizer, batch_input)
                label = choices[int(final_dataset[count]['label'])]
                outputs = model.generate(**encode_inputs, max_new_tokens = 1, pad_token_id = tokenizer.pad_token_id, \
                                          do_sample=False, num_beams=1)
                pred = tokenizer.batch_decode(outputs[0][-1].unsqueeze(dim=0), skip_special_tokens=True)[0]
                answers.append(pred)
                correct_counts[all_sample_dataset_names[count]] += str(pred) == str(label)
                count += 1
            print(set(answers))

    end_time = time.time()
    compute_metrics(args, correct_counts, total_num = len(final_dataset))
    print("Total run time: %.2f" % (end_time - start_time))
    return 

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--few_shot_number', type=int, default=0)
    parser.add_argument('--calibrate', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='sst2')
    parser.add_argument('--do_augmentation', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=900)
    parser.add_argument('--output_dir', type=str, default='./multiple_choice/cal.log')
    parser.add_argument('--task_type', type=str, default='multiple_choice')
    parser.add_argument('--k', type=int, default=5)

    args = parser.parse_args()
    print(args)
    if args.do_augmentation:
        print("-------------------Attention Aug-----------------")
        if args.model_type == "llama":
            if args.calibrate:
                LlamaAttention.forward = atten_aug_forward_cal_llama
            else:
                LlamaAttention.forward = atten_aug_forward_eval_llama
        elif args.model_type == "mistral":
            if args.calibrate:
                LlamaAttention.forward = atten_aug_forward_cal_mistral
            else:
                LlamaAttention.forward = atten_aug_forward_eval_mistral

    else:
        print("-------------------No Attention Aug-----------------")
    
    main(args)





