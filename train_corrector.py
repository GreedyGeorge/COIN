from __future__ import absolute_import, division, print_function
import argparse
import logging
import glob
import os
import random
import math
import copy
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
from tqdm.auto import tqdm, trange
from transformers import AutoTokenizer
from accelerate import Accelerator
from autocsc_ import *
import random


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, src, trg):
        self.guid = guid
        self.src = src
        self.trg = trg

class InputExample_err(object):
    def __init__(self, guid, src, trg, err_p, err_r):
        self.guid = guid
        self.src = src
        self.trg = trg
        self.err_p = err_p
        self.err_r = err_r

class InputFeatures(object):
    def __init__(self, src_ids, attention_mask, trg_ids, trg_ref_ids=None, error_pos=None, mask=None):
        self.src_ids = src_ids
        self.attention_mask = attention_mask
        self.trg_ids = trg_ids
        self.trg_ref_ids = trg_ref_ids
        self.error_pos = error_pos
        self.mask = mask


class DataProcessor:
    """
    Processor for the data set:
    a) in a .tsv format, i.e. src\ttrg; b) separate Chinese characters from each other by spaces; c) without headlines.
    """

    def get_train_examples(self, data_dir, filename):
        return self._create_examples(self._read(os.path.join(data_dir, filename)), "train")

    def get_dev_examples(self, data_dir, filename):
        return self._create_examples(self._read(os.path.join(data_dir, filename)), "dev")

    def get_test_examples(self, data_dir, filename):
        return self._create_examples(self._read(os.path.join(data_dir, filename)), "test")
    
    def get_json_examples(self, data_dir, filename, split=None):
        examples = []
        data = open(os.path.join(data_dir, filename), "r", encoding="utf-8")
        dataset = json.load(data)
        def add_space(sentence):
            sent=""
            for i in range(len(sentence)):
                sent+=sentence[i]+" "
            return sent[:-1]
        if "err_p" in dataset[0].keys():
            for i, item in enumerate(dataset):
                guid = "%s-%s" % (split, i)
                if len(item["src"]) == len(item["tgt"]):
                    examples.append(InputExample_err(guid=guid, src=add_space(item["src"]).split(), trg=add_space(item["tgt"]).split(), err_p=item["err_p"], err_r=item["err_r"]))
        else:
            for i, item in enumerate(dataset):
                guid = "%s-%s" % (split, i)
                if len(item["src"]) == len(item["tgt"]):
                    examples.append(InputExample(guid=guid, src=add_space(item["src"]).split(), trg=add_space(item["tgt"]).split()))
        return examples



    @staticmethod
    def _read(input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                src, trg = line.strip().split("\t")
                lines.append((src.split(), trg.split()))
            return lines

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (src, trg) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if len(src) == len(trg):
                examples.append(InputExample(guid=guid, src=src, trg=trg))
        return examples


class DataProcessorForRephrasing(DataProcessor):
    @staticmethod
    def convert_examples_to_features(examples, max_seq_length, tokenizer, verbose=True):
        features = []
        for i, example in enumerate(examples):
            src_ids = tokenizer(example.src,
                                max_length=max_seq_length // 2 - 2,
                                truncation=True,
                                is_split_into_words=True,
                                add_special_tokens=False).input_ids
            trg_ids = tokenizer(example.trg,
                                max_length=max_seq_length // 2 - 2,
                                truncation=True,
                                is_split_into_words=True,
                                add_special_tokens=False).input_ids
            mask = [tokenizer.cls_token_id] + src_ids + [tokenizer.sep_token_id] + [tokenizer.mask_token_id for _ in trg_ids] + [tokenizer.sep_token_id]
            err_mask = src_ids.copy()

            error_pos = [0]*len(src_ids)
            arand = random.random()
            if arand>0.95:              #0.95精确度错误指示有0.05的概率出错
                error_pos[int(arand*10000)%len(src_ids)] = 1
            for j in range(len(src_ids)):
                if src_ids[j]!=trg_ids[j]:
                    brand = random.random()
                    if brand>0.1:       #错误指示有0.1的概率缺失
                        error_pos[j]=1
                    crand = random.random()
                    if crand > 0.05:    #0.95召回率错误mask有0.05的概率缺失
                        for k in range(5):
                            if (j+k<len(err_mask)):err_mask[k+j]=tokenizer.mask_token_id
                            if (j-k>=0):err_mask[j-k]=tokenizer.mask_token_id
            drand = random.random()
            if drand>0.8:               #错误mask有0.2的概率出错
                fake_err = int(drand*10000)%len(err_mask)
                for k in range(5):
                    if (fake_err+k<len(err_mask)):err_mask[k+fake_err]=tokenizer.mask_token_id
                    if (fake_err-k>=0):err_mask[fake_err-k]=tokenizer.mask_token_id
            input_ids = [tokenizer.cls_token_id] + src_ids + [tokenizer.sep_token_id] + err_mask + [tokenizer.sep_token_id]
            label_ids = [tokenizer.cls_token_id] + src_ids + [tokenizer.sep_token_id] + trg_ids + [tokenizer.sep_token_id]
            attention_mask = [1] * len(input_ids)
            ref_ids = [tokenizer.cls_token_id] + trg_ids + [tokenizer.sep_token_id] + trg_ids + [tokenizer.sep_token_id]

            error_pos = [0] + error_pos+[0]*(len(src_ids)+2)

            offset_length = max_seq_length - len(input_ids)
            if offset_length > 0:
                input_ids = input_ids + [tokenizer.pad_token_id] * offset_length
                attention_mask = attention_mask + [0] * offset_length
                label_ids = label_ids + [tokenizer.pad_token_id] * offset_length
                ref_ids = ref_ids + [tokenizer.pad_token_id] * offset_length
                error_pos = error_pos + [0] * offset_length
                mask = mask + [tokenizer.pad_token_id] * offset_length
            input_ids, attention_mask, label_ids, ref_ids, error_pos, mask = input_ids[:max_seq_length], attention_mask[:max_seq_length], label_ids[:max_seq_length], ref_ids[:max_seq_length], error_pos[:max_seq_length], mask[:max_seq_length]

            assert len(input_ids) == max_seq_length
            assert len(attention_mask) == max_seq_length
            assert len(label_ids) == max_seq_length

            if verbose and i < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("src_tokens: %s" % " ".join(tokenizer.convert_ids_to_tokens(input_ids)))
                logger.info("trg_tokens: %s" % " ".join(tokenizer.convert_ids_to_tokens(label_ids)))
                logger.info("src_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("trg_ids: %s" % " ".join([str(x) for x in label_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))

            features.append(
                    InputFeatures(src_ids=input_ids,
                                  attention_mask=attention_mask,
                                  trg_ids=label_ids,
                                  trg_ref_ids=ref_ids,
                                  error_pos=error_pos,
                                  mask=mask)
            )
        return features
    @staticmethod
    def convert_examples_to_features_eval(examples, max_seq_length, tokenizer, verbose=True):
        features = []
        for i, example in enumerate(examples):
            src_ids = tokenizer(example.src,
                                max_length=max_seq_length // 2 - 2,
                                truncation=True,
                                is_split_into_words=True,
                                add_special_tokens=False).input_ids
            trg_ids = tokenizer(example.trg,
                                max_length=max_seq_length // 2 - 2,
                                truncation=True,
                                is_split_into_words=True,
                                add_special_tokens=False).input_ids
            mask = [tokenizer.cls_token_id] + src_ids + [tokenizer.sep_token_id] + [tokenizer.mask_token_id for _ in trg_ids] + [tokenizer.sep_token_id]
            err_mask = src_ids.copy()

            error_pos = [0]*len(src_ids)
            for k in example.err_p:
                if k<len(error_pos):
                    error_pos[k] = 1

            for j in example.err_r:
                if j < len(err_mask):
                    for k in range(5):
                        if (j+k<len(err_mask)):err_mask[k+j]=tokenizer.mask_token_id
                        if (j-k>=0):err_mask[j-k]=tokenizer.mask_token_id
            input_ids = [tokenizer.cls_token_id] + src_ids + [tokenizer.sep_token_id] + err_mask + [tokenizer.sep_token_id]
            label_ids = [tokenizer.cls_token_id] + src_ids + [tokenizer.sep_token_id] + trg_ids + [tokenizer.sep_token_id]
            attention_mask = [1] * len(input_ids)
            ref_ids = [tokenizer.cls_token_id] + trg_ids + [tokenizer.sep_token_id] + trg_ids + [tokenizer.sep_token_id]

            error_pos = [0] + error_pos+[0]*(len(src_ids)+2)

            offset_length = max_seq_length - len(input_ids)
            if offset_length > 0:
                input_ids = input_ids + [tokenizer.pad_token_id] * offset_length
                attention_mask = attention_mask + [0] * offset_length
                label_ids = label_ids + [tokenizer.pad_token_id] * offset_length
                ref_ids = ref_ids + [tokenizer.pad_token_id] * offset_length
                error_pos = error_pos + [0] * offset_length
                mask = mask + [tokenizer.pad_token_id] * offset_length
            input_ids, attention_mask, label_ids, ref_ids, error_pos, mask = input_ids[:max_seq_length], attention_mask[:max_seq_length], label_ids[:max_seq_length], ref_ids[:max_seq_length], error_pos[:max_seq_length], mask[:max_seq_length]

            assert len(input_ids) == max_seq_length
            assert len(attention_mask) == max_seq_length
            assert len(label_ids) == max_seq_length

            if verbose and i < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("src_tokens: %s" % " ".join(tokenizer.convert_ids_to_tokens(input_ids)))
                logger.info("trg_tokens: %s" % " ".join(tokenizer.convert_ids_to_tokens(label_ids)))
                logger.info("src_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("trg_ids: %s" % " ".join([str(x) for x in label_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))

            features.append(
                    InputFeatures(src_ids=input_ids,
                                  attention_mask=attention_mask,
                                  trg_ids=label_ids,
                                  trg_ref_ids=ref_ids,
                                  error_pos=error_pos,
                                  mask=mask)
            )
        return features
   
class Metrics:
    @staticmethod
    def compute(src_sents, trg_sents, prd_sents):
        def difference(src, trg):
            ret = copy.deepcopy(src)
            for i, (src_char, trg_char) in enumerate(zip(src, trg)):
                if src_char!= trg_char:
                    ret[i] = "(" + src_char + "->" + trg_char + ")"

            return "".join(ret)

        pos_sents, neg_sents, tp_sents, fp_sents, fn_sents, prd_pos_sents, prd_neg_sents = [], [], [], [], [], [], []
        for s, t, p in zip(src_sents, trg_sents, prd_sents):
            if s != t:
                pos_sents.append(difference(s, t))
                if p == t:
                    tp_sents.append(difference(s, t))
                if p == s:
                    fn_sents.append(difference(s, t))

            else:
                neg_sents.append(difference(s, t))
                if p != t:
                    fp_sents.append(difference(t, p))

            if s != p:
                prd_pos_sents.append(difference(s, p))
            if s == p:
                prd_neg_sents.append(difference(s, p))

        p = 1.0 * len(tp_sents) / len(prd_pos_sents)
        r = 1.0 * len(tp_sents) / len(pos_sents)
        f1 = 2.0 * (p * r) / (p + r + 1e-12)
        fpr = 1.0 * (len(fp_sents) + 1e-12) / (len(neg_sents) + 1e-12)

        return p, r, f1, fpr, tp_sents, fp_sents, fn_sents

def rand_mask(rand_):
    rand = random.random()
    if rand > rand_:
        return False
    else:
        return True

def find_mask(token_line, tokenizer):
    random.seed()
    rand_ = 0.3 + random.random()*0.4
    mask = [rand_mask(rand_) if token_line[i]==tokenizer.convert_tokens_to_ids(tokenizer.mask_token) else False for i in range(len(token_line))]

    return mask


def mask_tokens(inputs, input_mask, tokenizer):
    inputs = inputs.clone()
    input_mask = input_mask.clone()
    
    mask_tokens = [
        find_mask(val, tokenizer) for val in input_mask.tolist()
    ]
    tokens_mask = torch.tensor(mask_tokens, dtype=torch.bool)
    inputs[tokens_mask] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs

def new_mask_tokens_only_neg(inputs, input_mask, labels, tokenizer, noise_probability=0.2):
    inputs = inputs.clone()
    input_mask = input_mask.clone()
    probability_matrix = torch.full(input_mask.shape, noise_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in input_mask.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    neq_tokens_mask = (input_mask != labels).cpu()

    probability_matrix.masked_fill_(special_tokens_mask + neq_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs


def mask_tokens_only_neg(inputs, labels, tokenizer, noise_probability=0.2):
    inputs = inputs.clone()
    probability_matrix = torch.full(inputs.shape, noise_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    neq_tokens_mask = (inputs != labels).cpu()

    probability_matrix.masked_fill_(special_tokens_mask + neq_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs

def mask_tokens_random(inputs, labels, tokenizer):
    inputs = inputs.clone()
    probability_matrix = torch.full(inputs.shape, 0)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    neq_tokens_mask = (inputs != labels).cpu()

    probability_matrix.masked_fill_(special_tokens_mask + neq_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs

def adjust_learning_rate(optimizer, now_steps, total_steps, start_lr):
    if now_steps<0.25*total_steps:
        lr = start_lr * (float(now_steps)/total_steps)*4
    else:
        lr = start_lr * (4.0/3.0-(4.0/3.0)*(float(now_steps)/total_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    parser = argparse.ArgumentParser()

    # Data config
    parser.add_argument("--data_dir", type=str, default="../data",
                        help="Directory to contain the input data for all tasks.")
    parser.add_argument("--train_on", type=str, default="",
                        help="Specify a training set.")
    parser.add_argument("--eval_on", type=str, default="",
                        help="Specify a dev set.")
    parser.add_argument("--test_on_lemon", type=str, default="",
                        help="Specify the directory to LEMON.")
    parser.add_argument("--load_model_path", type=str, default="bert-base-chinese",
                        help="Pre-trained model path to load.")
    parser.add_argument("--model_type", type=str, default="relm",
                        help="Model architecture to load.")
    parser.add_argument("--cache_dir", type=str, default="../cache/",
                        help="Directory to store the pre-trained language models downloaded from s3.")
    parser.add_argument("--output_dir", type=str, default="model/",
                        help="Directory to output predictions and checkpoints.")
    parser.add_argument("--load_state_dict", type=str, default="",
                        help="Trained model weights to load for evaluation if needed.")

    # Training config
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to evaluate on the dev set.")
    parser.add_argument("--use_slow_tokenizer", action="store_true",
                        help="A slow tokenizer will be used if passed.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum total input sequence length after word-piece tokenization.")
    parser.add_argument("--train_batch_size", type=int, default=128,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=128,
                        help="Total batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", type=float, default=1000.0,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=1000,
                        help="Total number of training steps to perform. If provided, overrides training epochs.")
    parser.add_argument("--weight_decay", type=float, default=0.,
                        help="L2 weight decay for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward pass.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use mixed precision.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for initialization.")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="How many steps to save the checkpoint once.")
    parser.add_argument("--noise_probability", type=float, default=0.2,
                        help="Mask rate for masked-fine-tuning.")
    parser.add_argument("--mft", action="store_true",
                        help="Training with masked-fine-tuning.")

    args = parser.parse_args()

    relm = args.model_type.startswith("relm")
    coin = args.model_type.startswith("coin")

    AutoCSC = {
        "finetune": AutoCSCfinetune,
        "softmasked": AutoCSCSoftMasked,
        "mdcspell": AutoCSCMDCSpell,
        "relm": AutoCSCReLM,
        "coin": AutoCSCCOIN,
    }

    processor = DataProcessorForRephrasing() if relm or coin else DataProcessorForTagging()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, "-accelerate", args.fp16))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.do_train:
        torch.save(args, os.path.join(args.output_dir, "train_args.bin"))

    cache_dir = args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(args.load_model_path,
                                              do_lower_case=args.do_lower_case,
                                              cache_dir=cache_dir,
                                              use_fast=not args.use_slow_tokenizer,
                                              add_prefix_space=True)

    if args.do_train:
        accelerator = Accelerator(cpu=args.no_cuda, mixed_precision="fp16" if args.fp16 else "no")
        device = accelerator.device

        train_examples = processor.get_train_examples(args.data_dir, args.train_on)
        train_features = processor.convert_examples_to_features(train_examples, args.max_seq_length, tokenizer)

        all_input_ids = torch.LongTensor([f.src_ids for f in train_features])
        all_input_mask = torch.LongTensor([f.attention_mask for f in train_features])
        all_label_ids = torch.LongTensor([f.trg_ids for f in train_features])
        if coin:
            all_ref_ids = torch.LongTensor([f.trg_ref_ids for f in train_features])
            all_error_pos = torch.LongTensor([f.error_pos for f in train_features])
            mask = torch.LongTensor([f.mask for f in train_features])
            train_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_ref_ids, all_error_pos, mask)
        else:
            train_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)

        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.train_batch_size)
        train_dataloader = accelerator.prepare(train_dataloader)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        model = AutoCSC[args.model_type].from_pretrained(args.load_model_path,
                                                         cache_dir=cache_dir)

        
        if args.load_state_dict:
            model.load_state_dict(torch.load(args.load_state_dict))

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        model, optimizer = accelerator.prepare(model, optimizer)
        
        if args.do_eval:
            eval_examples = processor.get_json_examples(args.data_dir, args.eval_on)
            eval_features = processor.convert_examples_to_features_eval(eval_examples, args.max_seq_length, tokenizer)

            all_input_ids = torch.LongTensor([f.src_ids for f in eval_features])
            all_input_mask = torch.LongTensor([f.attention_mask for f in eval_features])
            all_label_ids = torch.LongTensor([f.trg_ids for f in eval_features])
            if coin:
                all_ref_ids = torch.LongTensor([f.trg_ref_ids for f in eval_features])
                all_error_pos = torch.LongTensor([f.error_pos for f in eval_features])
                mask = torch.LongTensor([f.mask for f in eval_features])
                eval_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_ref_ids, all_error_pos, mask)
            else:
                eval_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
            
            eval_dataloader = DataLoader(eval_data, shuffle=False, batch_size=args.eval_batch_size)
            eval_dataloader = accelerator.prepare(eval_dataloader)

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", args.train_batch_size * accelerator.num_processes)
        logger.info("  Num steps = %d", args.max_train_steps)

        progress_bar = tqdm(range(args.max_train_steps), desc="Iteration")
        global_step = 0
        best_result = list()
        wrap = False
        for epoch in range(int(args.num_train_epochs)):
            if wrap: break
            train_loss = 0
            num_train_examples = 0

            for step, batch in enumerate(train_dataloader):
                adjust_learning_rate(optimizer, global_step, args.max_train_steps, args.learning_rate)
                model.train()
                batch = tuple(t.to(device) for t in batch)
                if coin:
                    src_ids, attention_mask, trg_ids, trg_ref_ids, error_pos, mask = batch
                    if args.mft:
                        src_ids = new_mask_tokens_only_neg(src_ids, mask, trg_ref_ids, tokenizer, args.noise_probability)
                        src_ids = mask_tokens(src_ids, mask, tokenizer)
                else:
                    src_ids, attention_mask, trg_ids = batch
                    if args.mft:
                        src_ids = mask_tokens_only_neg(src_ids, trg_ids, tokenizer, args.noise_probability)

                if coin:
                    outputs = model(src_ids=src_ids,
                                    attention_mask=attention_mask,
                                    trg_ids=trg_ids,
                                    error_pos=error_pos)
                else:
                    outputs = model(src_ids=src_ids,
                                    attention_mask=attention_mask,
                                    trg_ids=trg_ids)
                loss = outputs["loss"]

                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)

                train_loss += loss.item()
                num_train_examples += src_ids.size(0)
                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    progress_bar.update(1)

                if args.do_eval and global_step % args.save_steps == 0 and (step + 1) % args.gradient_accumulation_steps == 0:
                    logger.info("***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size * accelerator.num_processes)

                    def decode(input_ids):
                        return tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)

                    model.eval()
                    all_inputs, all_labels, all_predictions = [], [], []
                    for batch in tqdm(eval_dataloader):
                        batch = tuple(t.to(device) for t in batch)
                        src_ids, attention_mask, trg_ids = batch[:3]
                        mask = batch[-1]
                        with torch.no_grad():
                            if coin:
                                error_pos = batch[-2]
                                outputs = model(src_ids=src_ids,
                                                attention_mask=attention_mask,
                                                trg_ids=trg_ids,
                                                error_pos=error_pos)
                            else:
                                outputs = model(src_ids=src_ids,
                                                attention_mask=attention_mask,
                                                trg_ids=trg_ids)
                            prd_ids = outputs["predict_ids"]

                        src_ids, trg_ids, prd_ids, mask = accelerator.gather_for_metrics((src_ids, trg_ids, prd_ids, mask))
                        for s, t, p in zip(mask.tolist(), trg_ids.tolist(), prd_ids.tolist()):
                            if relm or coin:
                                _t = [tt for tt, st in zip(t, s) if st == tokenizer.mask_token_id]
                                _p = [pt for pt, st in zip(p, s) if st == tokenizer.mask_token_id]

                                all_inputs += [decode(s)]
                                all_labels += [decode(_t)]
                                all_predictions += [decode(_p)]

                            else:
                                all_inputs += [decode(s)]
                                all_labels += [decode(t)]
                                all_predictions += [decode(p)]

                    loss = train_loss / global_step
                    p, r, f1, fpr, tp, fp, fn = Metrics.compute(all_inputs, all_labels, all_predictions)
    
                    output_tp_file = os.path.join(args.output_dir, "sents.tp")
                    with open(output_tp_file, "w") as writer:
                        for line in tp:
                            writer.write(line + "\n")
                    output_fp_file = os.path.join(args.output_dir, "sents.fp")
                    with open(output_fp_file, "w") as writer:
                        for line in fp:
                            writer.write(line + "\n")
                    output_fn_file = os.path.join(args.output_dir, "sents.fn")
                    with open(output_fn_file, "w") as writer:
                        for line in fn:
                            writer.write(line + "\n")

                    result = {
                        "global_step": global_step,
                        "loss": loss,
                        "eval_p": p * 100,
                        "eval_r": r * 100,
                        "eval_f1": f1 * 100,
                        "eval_fpr": fpr * 100,
                    }
                    if accelerator.is_local_main_process:
                        model_to_save = model.module if hasattr(model, "module") else model
                        output_model_file = os.path.join(args.output_dir, "step-%s_f1-%.2f.bin" % (str(global_step), result["eval_f1"]))
                        torch.save(model_to_save.state_dict(), output_model_file)
                        best_result.append((result["eval_f1"], output_model_file))
                        best_result.sort(key=lambda x: x[0], reverse=True)
                        if len(best_result) > 3:
                            _, model_to_remove = best_result.pop()
                            os.remove(model_to_remove)

                        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                        with open(output_eval_file, "a") as writer:
                            logger.info("***** Eval results *****")
                            writer.write(
                                "Global step = %s | loss = %.3f | eval precision = %.2f | eval recall = %.2f | eval f1 = %.2f | eval fpr = %.2f\n"
                                % (str(result["global_step"]),
                                result["loss"],
                                result["eval_p"],
                                result["eval_r"],
                                result["eval_f1"],
                                result["eval_fpr"]))
                            for key in sorted(result.keys()):
                                logger.info("Global step: %s,  %s = %s", str(global_step), key, str(result[key]))

                if global_step >= args.max_train_steps:
                    wrap = True
                    break

    if args.test_on_lemon:
        accelerator = Accelerator(cpu=args.no_cuda, mixed_precision="fp16" if args.fp16 else "no")
        device = accelerator.device

        model = AutoCSC[args.model_type].from_pretrained(args.load_model_path,
                                                         state_dict=torch.load(args.load_state_dict),
                                                         cache_dir=cache_dir)
        model = accelerator.prepare(model)

        avg = 0
        for cat in ["gam", "car", "nov", "enc", "new", "cot", "mec", "sig"]:
            eval_examples = processor.get_test_examples(args.test_on_lemon, cat + ".txt")
            eval_features = processor.convert_examples_to_features(eval_examples, args.max_seq_length, tokenizer, False)

            all_input_ids = torch.tensor([f.src_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.attention_mask for f in eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.trg_ids for f in eval_features], dtype=torch.long)
            if relm:
                all_ref_ids = torch.LongTensor([f.trg_ref_ids for f in eval_features])
                eval_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_ref_ids)
            else:
                eval_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)

            eval_dataloader = DataLoader(eval_data, shuffle=False, batch_size=args.eval_batch_size)
            eval_dataloader = accelerator.prepare(eval_dataloader)

            def decode(input_ids):
                return tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)

            model.eval()
            all_inputs, all_labels, all_predictions = [], [], []
            for batch in tqdm(eval_dataloader, leave=False):
                batch = tuple(t.to(device) for t in batch)
                src_ids, attention_mask, trg_ids = batch[:3]
                with torch.no_grad():
                    outputs = model(src_ids=src_ids,
                                    attention_mask=attention_mask,
                                    trg_ids=trg_ids)
                    prd_ids = outputs["predict_ids"]

                src_ids, trg_ids, prd_ids = accelerator.gather_for_metrics((src_ids, trg_ids, prd_ids))
                for s, t, p in zip(src_ids.tolist(), trg_ids.tolist(), prd_ids.tolist()):
                    if relm:
                        _t = [tt for tt, st in zip(t, s) if st == tokenizer.mask_token_id]
                        _p = [pt for pt, st in zip(p, s) if st == tokenizer.mask_token_id]

                        all_inputs += [decode(s)]
                        all_labels += [decode(_t)]
                        all_predictions += [decode(_p)]

                    else:
                        all_inputs += [decode(s)]
                        all_labels += [decode(t)]
                        all_predictions += [decode(p)]

            p, r, f1, fpr, tp, fp, fn = Metrics.compute(all_inputs, all_labels, all_predictions)

            result = {
                "eval_p": p * 100,
                "eval_r": r * 100,
                "eval_f1": f1 * 100,
                "eval_fpr": fpr * 100,
            }
            avg += f1 * 100
            logger.info("{}: F1 = {}".format(cat.upper(), f1 * 100))

        avg /= 8
        logger.info("AVG: F1 = {}".format(avg))


if __name__ == "__main__":
    main()
