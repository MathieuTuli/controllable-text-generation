"""
@author: Mathieu Tuli, Sarvagya Agrawal
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from pathlib import Path

# BertTokenizer, OpenAIGPTTokenizer, GPT2Tokenizer, AlbertTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import (
    GPT2Config, GPT2LMHeadModel, BertLMHeadModel,
    AutoTokenizer, BertConfig,
    AutoModelForSeq2SeqLM
)

import torch

from .configs import CONFIG_DICTS

TOKENIZERS = set(['bert-base-uncased', 'openai-gpt',
                  'gpt2', 'bert-base-uncased', 't5-small'])
MODELS = set(['gpt2', 'distilgpt2', 'bert-base-uncased', 't5-small'])
SEQ2SEQ_MODELS = set(['t5-small'])


def get_tokenizer(tokenizer_name: str,
                  cache_dir: Path,
                  dataset: str,
                  task: str,
                  lower_case: bool = True) -> PreTrainedTokenizerBase:
    # if tokenizer_name == 'bert-base-uncased':
    #     tokenizer = BertTokenizer.from_pretrained(
    #         tokenizer_name, cache_dir=cache_dir)
    # elif tokenizer_name == 'openai-gpt':
    #     tokenizer = OpenAIGPTTokenizer.from_pretrained(
    #         tokenizer_name, cache_dir=cache_dir)
    # elif tokenizer_name == 'gpt2':
    #     tokenizer = GPT2Tokenizer.from_pretrained(
    #         tokenizer_name, cache_dir=cache_dir)
    # elif tokenizer_name == 'albert-base-v2':
    #     tokenizer = AlbertTokenizer.from_pretrained(
    #         tokenizer_name, cache_dir=cache_dir)
    if tokenizer_name in TOKENIZERS:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                                  cache_dir=cache_dir)
    else:
        raise ValueError(f'Unknown tokenizer. Must be one of {TOKENIZERS}')
    if dataset == 'multiwoz2.1':
        if task == 'clm':
            tokenizer.add_special_tokens(
                {'bos_token': '<|endoftext|>'})
            tokenizer.add_special_tokens(
                {'eos_token': '<|endoftext|>'})
    if tokenizer._pad_token is None:
        tokenizer.add_special_tokens(
            {'pad_token': '[PAD]'})
        # tokenizer.pad_token = 0.
    return tokenizer


def get_model(model_name: str,
              cache_dir: Path,
              pretrained: bool = True,
              weights: str = None,
              task: str = 'clm') -> torch.nn.Module:
    if task == 'nmt' and model_name in SEQ2SEQ_MODELS:
        _model = AutoModelForSeq2SeqLM
        _config = None
    elif model_name == 'gpt2':
        _model = GPT2LMHeadModel
        _config = GPT2Config()
    elif model_name == 'distilgpt2':
        _model = GPT2LMHeadModel
        _config = GPT2Config
    elif model_name == 'bert-base-uncased':
        _model = BertLMHeadModel
        _config = BertConfig
    if pretrained:
        model_name = weights if weights is not None else model_name
        if _config is not None:
            model = _model.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                config=_config.from_pretrained(model_name))
        else:
            model = _model.from_pretrained(
                model_name,
                cache_dir=cache_dir)
    else:
        model = _model(
            config=_config(**CONFIG_DICTS[model_name]))
    # if task == 'nmt' and model.config.decoder_state_token_id is None:
    #     raise ValueError(
    #         "Make sure that `decoder_start_token_id` is correctly defined")
    return model
