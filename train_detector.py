
from transformers import ElectraForPreTraining, AutoTokenizer
import torch
import evaluate
import os
import re
import json
import numpy as np
from datasets import load_from_disk, load_dataset, load_metric, concatenate_datasets
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer
from difflib import SequenceMatcher
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def padding(examples, max_len):
    error_list = []
    for an_error in examples:
        if len(an_error)<max_len:
            an_error += np.zeros(max_len-len(an_error), dtype=int).tolist()
        else:
            an_error=an_error[:max_len]
        error_list.append(an_error)
    return error_list

def process_data(examples):
    source_max_length = 512
    target_max_length = 512
    model_inputs = tokenizer(examples['src'],
                             max_length=source_max_length,
                             padding=True,
                             truncation=True)
    model_inputs["input_ids"] = padding(model_inputs["input_ids"], 256)
    

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['tgt'], 
                           max_length=target_max_length,
                           padding=True, 
                           truncation=True)
    labels["input_ids"] = padding(labels["input_ids"], 256)
    model_inputs['labels'] = Get_error(model_inputs["input_ids"], labels["input_ids"])
    model_inputs["labels"] = padding(model_inputs["labels"], 256)
    model_inputs["attention_mask"] = padding(model_inputs["attention_mask"], 256)
    model_inputs.pop("token_type_ids")
    return model_inputs

def eval_pr(ress, tgt):
    p = 0.0
    tp = 0.0001
    r = 0.0
    tr = 0.0001
    acc = 0
    total = 0.0001
    for aress, atgt in zip(ress, tgt):
        total+=1
        if aress==atgt:
            acc+=1
        for item in aress:
            tp+=1
            if item in atgt:
                p+=1
        for item2 in atgt:
            tr+=1
            if item2 in aress:
                r+=1
    print(p/tp)
    print(r/tr)
    print(acc/total)
    return p/tp, r/tr

def compute(predictions, labels):
    total = 0
    acc = 0
    ress = []
    tgt_l = []
    for pre,tgt,length_ in zip(predictions, labels, length):
        temp1 = pre[:length_].astype(np.int32)
        temp2 = tgt[:length_].astype(np.int32)
        ress_ = []
        tgt_ = []
        for i in range(length_):
            if temp1[i] == 1:
                ress_.append(i)
            if temp2[i] == 1:
                tgt_.append(i)

        ress.append(ress_)
        tgt_l.append(tgt_)
        
        if (pre[:length_].astype(np.int32)==tgt[:length_].astype(np.int32)).all():
            acc = acc + 1
            total = total + 1
        else:
            total = total + 1
    eval_pr(ress, tgt_l)
    return acc/total

def compute_metrics(eval_pred):
    # calculate chinese rouge metric
    pre_list = []
    tgt_list = []
    predictions, labels = eval_pred
    
    predictions = np.round((np.sign(predictions) + 1) / 2).astype(np.int32)
    labels = labels.astype(np.int32)
    print("\n")
    print(compute(predictions, labels))
    for pre,tgt in zip(predictions, labels):
        p = ""
        t = ""
        for i in pre:
            p+=str(i)
        for j in tgt:
            t+=str(j)
        pre_list.append(p)
        tgt_list.append(t)
    
    return metric.compute(predictions=pre_list, references=tgt_list)

def Get_error(str1_list, str2_list):
    list_out = []
    for str1, str2 in zip(str1_list, str2_list):
        errpr_pos = np.zeros_like(str1, dtype=int)
        for i in range(len(str2)):
            if str1[i]!=str2[i]:
                errpr_pos[i]=1
        error_list = errpr_pos.tolist()
        list_out.append(error_list)
    return list_out

discriminator = ElectraForPreTraining.from_pretrained("./checkpoint/ELECTRA")
tokenizer = AutoTokenizer.from_pretrained("./checkpoint/ELECTRA")

test_file = "./data/test_data.json"
train_correction_dataset = load_dataset("json",data_files="./data/train_data.json", split="train")
eval_correction_dataset = load_dataset("json",data_files=test_file,split="train")

metric = evaluate.load("./metrics/rouge")

train_correction_dataset = train_correction_dataset.map(process_data,batch_size=1000,
                                                        batched=True)
eval_correction_dataset = eval_correction_dataset.map(process_data,batch_size=1000,
                                                      batched=True)

length=[]
with open(test_file, "r", encoding="utf-8") as data:
    data_ = json.load(data)
    for item in data_:
        length.append(len(item["src"]))

saved_dir = "./models/detector_law"
train_batch_size = 64
eval_batch_size = 64
args = TrainingArguments(
    output_dir=saved_dir, #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    # num_train_epochs=10, # number of training epochs
    per_device_train_batch_size=train_batch_size, # batch size for training
    per_device_eval_batch_size=eval_batch_size,  # batch size for evaluation
    eval_steps=50, # Number of update steps between two evaluations.
    save_steps=50, # after # steps model is saved 
    warmup_steps=500,# number of warmup steps for learning rate scheduler
    max_steps=2000,
    # prediction_loss_only=True,
    # warmup_ratio=0.1,
    save_total_limit=5,
    load_best_model_at_end=True,
    save_strategy="steps",
    evaluation_strategy="steps",
    gradient_accumulation_steps=2,
    learning_rate=3e-6,
    weight_decay=1e-2,
    fp16=True,
    report_to="tensorboard"
)
trainer = Trainer(
    discriminator,
    args=args,
    train_dataset=train_correction_dataset,
    eval_dataset=eval_correction_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("start train model...")
trainer.train()
trainer.save_model()
