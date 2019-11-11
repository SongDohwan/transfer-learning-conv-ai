# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings

import torch
import torch.nn.functional as F

from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset_personalities, download_pretrained_model
import json

import sys
import numpy as np
import time
import math

sys.path.append('./src')
import data_io, params, SIF_embedding
params = params.params()

def top_filtering(logits, words, weight4ind, We, tokenizer, history, args, params, embedding1, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf'), current_output=None):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    if current_output is None:
        current_output = []
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))

    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        indices_to_use = sorted_indices[~sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        if len(indices_to_use) > 1:
            cands = [current_output + [idx] for idx in indices_to_use.tolist()]
            raw_cands = [tokenizer.decode(cand, skip_special_tokens=True) for cand in cands]
            scores = []
            for i in raw_cands:
                sentences = [i]
                x, m = data_io.sentences2idx(sentences,
                                             words)
                w = data_io.seq2weight(x, m, weight4ind)
                embedding2 = SIF_embedding.SIF_embedding(We, x, w, params)
                inn = (embedding1 * embedding2).sum(axis=1)
                emb1norm = np.sqrt((embedding1 * embedding1).sum(axis=1))
                emb2norm = np.sqrt((embedding2 * embedding2).sum(axis=1))
                scores.append(inn / emb1norm / emb2norm)
                #print(sentences)
            for idx, sim in zip(indices_to_use, scores):
                logits[idx] += sim.item()
            """
            probs = F.softmax(logits, dim=-1)
            index = []
            for i in probs:
                if i > 0:
                    index.append(i)
            prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, len(index))
            text = []
            last_utt = history[-1]
            last = tokenizer.decode(last_utt, skip_special_tokens=True)
            sentences = [last]
            # load sentences
            x, m = data_io.sentences2idx(sentences,
                                            words)  # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
            w = data_io.seq2weight(x, m, weight4ind)  # get word weights

            # set parameters
            global params
            params.rmpc = rmpc
            # get SIF embedding
            embedding1 = SIF_embedding.SIF_embedding(We, x, w, params)  # embedding[i,:] is the embedding for sentence i
            for i in prev:
                text.append(i.item())
            for i in text:
                cand = current_output.copy()
                cand.append(i)
                indice = i
                raw_text=tokenizer.decode(cand, skip_special_tokens=True)
                sentences = [raw_text]
                x, m= data_io.sentences2idx(sentences,
                                                words)
                w = data_io.seq2weight(x, m, weight4ind)
                embedding2 = SIF_embedding.SIF_embedding(We, x, w, params)
                inn = (embedding1 * embedding2 ).sum(axis=1)
                emb1norm = np.sqrt((embedding1 * embedding1).sum(axis=1))
                emb2norm = np.sqrt((embedding2  * embedding2 ).sum(axis=1))
                scores = inn / emb1norm / emb2norm
                #print(scores)
                logits[indice] += scores.item()
                cand.clear()
                """

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

def normal_top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history, tokenizer, model, args, words, weight4ind, We, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []
    last_utt = history[-1]
    last = tokenizer.decode(last_utt, skip_special_tokens=True)
    sentences = [last]
    # load sentences
    x, m = data_io.sentences2idx(sentences,
                                 words)  # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
    w = data_io.seq2weight(x, m, weight4ind)  # get word weights

    rmpc = 1  # number of principal components to remove in SIF weighting scheme

    # set parameters
    global params
    params.rmpc = rmpc
    # get SIF embedding
    embedding1 = SIF_embedding.SIF_embedding(We, x, w, params)  # embedding[i,:] is the embedding for sentence i

    for i in range(args.max_length):
        instance, _ = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
        temperature = 1.0
        top_k = 0
        top_p = 0.9

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, words, weight4ind, We, tokenizer, history, args, params, embedding1, top_k=top_k, top_p=top_p, current_output=current_output)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

def normal_sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance, _ = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)
        temperature = 1.0
        top_k = 0
        top_p = 0.9

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = normal_top_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="gpt", help="Model type (gpt or gpt2)")
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        args.model_checkpoint = download_pretrained_model()

    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class = GPT2Tokenizer if "gpt2" == args.model else OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model_class = GPT2LMHeadModel if "gpt2" == args.model else OpenAIGPTLMHeadModel
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    add_special_tokens_(model, tokenizer)


    logger.info("Sample a personality")
    #personalities = get_dataset_personalities(tokenizer, args.dataset_path, args.dataset_cache)
    #personality = random.choice(personalities)
    #logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))
    wordfile = './data/truncate.txt'  # word vector file, can be downloaded from GloVe website
    weightfile = './auxiliary_data/enwiki_vocab_min200.txt'  # each line is a word and its frequency
    weightpara = 1e-3  # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
    # load word vectors
    (words, We) = data_io.getWordmap(wordfile)
    # load word weights
    word2weight = data_io.getWordWeight(weightfile, weightpara)  # word2weight['str'] is the weight for the word 'str'
    weight4ind = data_io.getWeight(words, word2weight)  # weight4ind[i] is the weight for the i-th word

    p = 0
    start_time = time.time()
    with open('data_volunteers.json') as json_file:
        json_data = json.load(json_file)
        for i in json_data:
            p += 1
            #if p <1100:
            #    continue
            history = []
            personality = []
            query_set = []
            json_dialog = i["dialog"]
            json_bot = i["bot_profile"]
            for j in json_bot:
                personality.append(tokenizer.encode(j))
            #logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))
            persona = tokenizer.decode(chain(*personality))
            row = {"Personality": persona}
            text = []
            for j in json_dialog:
                if j["sender_class"] == "Human":
                    json_text = j["text"]
                    raw_text = json_text
                    check = tokenizer.decode(tokenizer.encode(raw_text), skip_special_tokens=True)
                    if check == "":
                        history.append(tokenizer.encode(raw_text))
                        with torch.no_grad():
                            out_ids = normal_sample_sequence(personality, history, tokenizer, model, args)
                        # history.append(out_ids)
                        history = history[-(2 * args.max_history + 1):]
                        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
                        text.append({"evaluation_score": j["evaluation_score"], "id": j["id"], "sender": j["sender"],
                                     "sender_class": j["sender_class"], "text": raw_text, "generated_text": out_text})
                        continue
                    history.append(tokenizer.encode(raw_text))
                    with torch.no_grad():
                        out_ids = sample_sequence(personality, history, tokenizer, model, args, words, weight4ind, We)
                    # history.append(out_ids)
                    history = history[-(2 * args.max_history + 1):]
                    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
                    text.append({"evaluation_score": j["evaluation_score"], "id": j["id"], "sender": j["sender"],
                                 "sender_class": j["sender_class"], "text": raw_text, "generated_text": out_text})
                else:
                    json_text = j["text"]
                    raw_text = json_text
                    history.append(tokenizer.encode(raw_text))
                    text.append({"evaluation_score": j["evaluation_score"], "id": j["id"], "sender": j["sender"],
                                 "sender_class": j["sender_class"], "text": raw_text})
            row["dialog"] = text
            query_set.append(row)
            #print(query_set)
            with open('./sif_set/sif' + str(p) + '.json', 'w', encoding='utf-8') as make_file:
                json.dump(query_set, make_file)
            if not p % 10:
                print(str(p * 100 / 1111) + '%, ' + str(time.time() - start_time) + 'sec')
    '''
    while True:
        raw_text = input(">>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        history.append(tokenizer.encode(raw_text))
        with torch.no_grad():
            out_ids = sample_sequence(personality, history, tokenizer, model, args)
        history.append(out_ids)
        history = history[-(2*args.max_history+1):]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        print(out_text)
    '''


if __name__ == "__main__":
    run()
