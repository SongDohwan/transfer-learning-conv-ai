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
import sys
import numpy as np

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import torch.nn.functional as F

from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset_personalities, download_pretrained_model
from sklearn.metrics.pairwise import cosine_similarity

import os
import time
import matplotlib
import matplotlib.pyplot as plt


def top_k_top_p_filtering(logits, tokenizer, history, args, B_tokenizer, B_model, top_k=0, top_p=0.0, filter_value=-float('Inf'),
                          current_output=None):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    if current_output is None:
        current_output = []
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

        probs = F.softmax(logits, dim=-1)
        index = []
        for i in probs:
            if i > 0:
                index.append(i)
        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, len(index))
        text = []
        last_utt = history[-1]
        last = tokenizer.decode(last_utt, skip_special_tokens=True)

        marked_text2 = "[CLS] " + last + " [SEP]"
        tokenized_text2 = B_tokenizer.tokenize(marked_text2)
        indexed_tokens2 = B_tokenizer.convert_tokens_to_ids(tokenized_text2)
        segments_ids2 = [1] * len(tokenized_text2)
        tokens_tensor2 = torch.tensor([indexed_tokens2])
        segments_tensors2 = torch.tensor([segments_ids2])
        with torch.no_grad():
            encoded_layers2, _ = B_model(tokens_tensor2, segments_tensors2)
        sentence_embedding2 = torch.mean(encoded_layers2[11], 1)
        #print(sentence_embedding2[0])

        token_embeddings2 = []
        batch_i = 0

        # For each token in the sentence...
        for token_i in range(len(tokenized_text2)):

            # Holds 12 layers of hidden states for each token
            hidden_layers = []

            # For each of the 12 layers...
            for layer_i in range(len(encoded_layers2)):
                # Lookup the vector for `token_i` in `layer_i`
                vec2 = encoded_layers2[layer_i][batch_i][token_i]

                hidden_layers.append(vec2)

            token_embeddings2.append(hidden_layers)

        #concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in
        #                              token_embeddings]  # [number_of_tokens, 3072]

        summed_last_4_layers2 = [torch.sum(torch.stack(layer)[-4:], 0) for layer in
                                token_embeddings2]  # [number_of_tokens, 768]

        #print(len(index))
        print("-------------------------------------------------------------")


        for i in prev:
            text.append(i.item())
        for i in text:
            x = current_output.copy()
            x.append(i)
            raw_text=tokenizer.decode(x, skip_special_tokens=True)
            #raw_text = tokenizer.decode(x)
            marked_text1 = "[CLS] " + raw_text + " [SEP]"
            tokenized_text1 = B_tokenizer.tokenize(marked_text1)
            indexed_tokens1 = B_tokenizer.convert_tokens_to_ids(tokenized_text1)
            segments_ids1 = [0] * len(tokenized_text1)
            tokens_tensor1 = torch.tensor([indexed_tokens1])
            segments_tensors1 = torch.tensor([segments_ids1])
            #outputs = model(tokens_tensor1)
            #last_hidden_states = outputs[0]
            with torch.no_grad():
                encoded_layers1, _ = B_model(tokens_tensor1, segments_tensors1)
            sentence_embedding1 = torch.mean(encoded_layers1[11], 1)
            #print(sentence_embedding1[0])
            sim = cosine_similarity(sentence_embedding1.reshape(1,-1), sentence_embedding2.reshape(1,-1))
            print(raw_text)
            print("sen_sim: ")
            print(sim)
            token_embeddings1 = []

            # For each token in the sentence...
            for token_i in range(len(tokenized_text1)):

                # Holds 12 layers of hidden states for each token
                hidden_layers = []

                # For each of the 12 layers...
                for layer_i in range(len(encoded_layers1)):
                    # Lookup the vector for `token_i` in `layer_i`
                    vec1 = encoded_layers1[layer_i][batch_i][token_i]

                    hidden_layers.append(vec1)

                token_embeddings1.append(hidden_layers)

            #concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in
             #                             token_embeddings]  # [number_of_tokens, 3072]

            summed_last_4_layers1 = [torch.sum(torch.stack(layer)[-4:], 0) for layer in
                                     token_embeddings1]  # [number_of_tokens, 768]
            temp = []
            for i in summed_last_4_layers1:
                a = []
                for j in summed_last_4_layers2:
                    #print("cos: ")
                    #print(cosine_similarity(i.reshape(1,-1), j.reshape(1,-1))[0][0])
                    a.append(cosine_similarity(i.reshape(1,-1), j.reshape(1,-1))[0][0])
                temp.append(max(a))
            #print("temp: ")
            #print(temp)
            print("sim: ")
            sim = sum(temp) / len(temp)
            print(sim)
            x.clear()

    return logits


def sample_sequence(personality, history, tokenizer, B_tokenizer, model, args, B_model, current_output=None):
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
        logits = top_k_top_p_filtering(logits, tokenizer, history, args, B_tokenizer, B_model, top_k=top_k, top_p=top_p,
                                       current_output=current_output)
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
    parser.add_argument("--dataset_path", type=str, default="",
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="gpt", help="Model type (gpt or gpt2)")
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
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
    B_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    B_model = BertModel.from_pretrained('bert-base-uncased')
    B_model.eval()

    logger.info("Sample a personality")
    personalities = get_dataset_personalities(tokenizer, args.dataset_path, args.dataset_cache)
    personality = random.choice(personalities)
    logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))

    history = []

    while True:
        raw_text = input(">>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        history.append(tokenizer.encode(raw_text))
        with torch.no_grad():
            out_ids = sample_sequence(personality, history, tokenizer, B_tokenizer, model, args, B_model)
        history.append(out_ids)
        history = history[-(2 * args.max_history + 1):]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        print(out_text)
'''
####################################
        marked_text2 = "[CLS] " + "hi. how are you?" + " [SEP]"
        tokenized_text2 = B_tokenizer.tokenize(marked_text2)
        indexed_tokens2 = B_tokenizer.convert_tokens_to_ids(tokenized_text2)
        segments_ids2 = [1] * len(tokenized_text2)
        tokens_tensor2 = torch.tensor([indexed_tokens2])
        segments_tensors2 = torch.tensor([segments_ids2])
        with torch.no_grad():
            encoded_layers2, _ = B_model(tokens_tensor2, segments_tensors2)
        sentence_embedding2 = torch.mean(encoded_layers2[11], 1)
        # print(sentence_embedding2[0])

        token_embeddings2 = []
        batch_i = 0

        # For each token in the sentence...
        for token_i in range(len(tokenized_text2)):

            # Holds 12 layers of hidden states for each token
            hidden_layers = []

            # For each of the 12 layers...
            for layer_i in range(len(encoded_layers2)):
                # Lookup the vector for `token_i` in `layer_i`
                vec2 = encoded_layers2[layer_i][batch_i][token_i]

                hidden_layers.append(vec2)

            token_embeddings2.append(hidden_layers)

        # concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in
        #                              token_embeddings]  # [number_of_tokens, 3072]

        summed_last_4_layers2 = [torch.sum(torch.stack(layer)[-4:], 0) for layer in
                                 token_embeddings2]  # [number_of_tokens, 768]

        marked_text1 = "[CLS] " + "i'm good. i am working on my airplane." + " [SEP]"
        tokenized_text1 = B_tokenizer.tokenize(marked_text1)
        indexed_tokens1 = B_tokenizer.convert_tokens_to_ids(tokenized_text1)
        segments_ids1 = [0] * len(tokenized_text1)
        tokens_tensor1 = torch.tensor([indexed_tokens1])
        segments_tensors1 = torch.tensor([segments_ids1])
        with torch.no_grad():
            encoded_layers1, _ = B_model(tokens_tensor1, segments_tensors1)
        sentence_embedding1 = torch.mean(encoded_layers1[11], 1)
        # print(sentence_embedding2[0])

        token_embeddings1 = []
        batch_i = 0

        # For each token in the sentence...
        for token_i in range(len(tokenized_text1)):

            # Holds 12 layers of hidden states for each token
            hidden_layers = []

            # For each of the 12 layers...
            for layer_i in range(len(encoded_layers1)):
                # Lookup the vector for `token_i` in `layer_i`
                vec1 = encoded_layers1[layer_i][batch_i][token_i]

                hidden_layers.append(vec1)

            token_embeddings1.append(hidden_layers)

        # concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in
        #                              token_embeddings]  # [number_of_tokens, 3072]

        summed_last_4_layers1 = [torch.sum(torch.stack(layer)[-4:], 0) for layer in
                                 token_embeddings1]  # [number_of_tokens, 768]
        temp = []
        for i in summed_last_4_layers1:
            a = []
            for j in summed_last_4_layers2:
                print("cos: ")
                print(cosine_similarity(i.reshape(1,-1), j.reshape(1,-1))[0][0])
                a.append(cosine_similarity(i.reshape(1, -1), j.reshape(1, -1))[0][0])
            temp.append(max(a))
        print("temp: ")
        print(temp)
        print("sim: ")
        sim = sum(temp) / len(temp)
        print(sim)
###############################################################################
        '''



if __name__ == "__main__":
    run()