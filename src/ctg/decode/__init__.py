"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from argparse import Namespace
from pathlib import Path
from typing import List

from transformers import EncoderDecoderModel

import torch
import tqdm

from ..data.utils import load_data, extend_vocabulary
from ..models import get_model_tokenizer
from ..utils.io import create_dir

DECODE_METHODS = ['greedy', 'beam']


def get_decoder(method: str) -> callable:
    if method == 'greedy':
        def greedy(model: torch.nn.Module, inputs: torch.Tensor,
                   max_length: int, break_tokens: List[str]):
            # outputs = []
            # while decoded_token not in break_tokens:
            #     output = model.generate(
            #         inputs, max_length=max_length)[0]
            #     next_token = torch.argmax(output[0, -1, :]).item()
            # with torch.no_grad():
            if isinstance(model, EncoderDecoderModel):
                outputs = model.generate(
                    inputs, max_length=max_length,
                    decoder_start_token_id=model.config.decoder.pad_token_id)
            else:
                outputs = model.generate(
                    inputs, max_length=max_length,
                    decoder_start_token_id=model.config.pad_token_id)
            return outputs
        return greedy
    elif method == 'beam':
        def beam_search(model: torch.nn.Module,
                        inputs: torch.Tensor,
                        beam_depth: int = -1,
                        beam_width: int = -1):
            if not beam_width > 0 or not beam_depth > 0:
                raise ValueError("Beam depth and width must be > 0.")
            # with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=beam_depth,
                num_beams=beam_width, early_stopping=True,
                num_return_sequences=beam_width)
            return outputs
        return beam_search
    else:
        raise NotImplementedError(
            f"Unkown decoding method {method}")


def main(args: Namespace) -> None:
    tokenizer = get_tokenizer(tokenizer_name=args.decode.tokenizer,
                              cache_dir=args.io.cache_dir,
                              dataset=args.data.name,
                              task=args.data.task)
    # extend_vocabulary(tokenizer, fname=Path(args.data.vocab))
    model = get_model(args.decode.model,
                      weights=args.decode.weights,
                      cache_dir=args.io.cache_dir,
                      tokenizer_len=len(tokenizer),
                      pretrained=True,
                      task=args.data.task)
    model.eval()
    device = torch.device('cuda', args.gpu)
    model.to(device)
    break_tokens = tokenizer.encode(tokenizer._eos_token)
    decoder = get_decoder(method=args.decode.method)
    data_loader, sampler = load_data(
        fname=Path(args.data.src),
        tokenizer=tokenizer,
        model=model,
        max_length=args.decode.max_length,
        max_samples=args.decode.max_samples,
        task=args.data.task,
        batch_size=args.decode.batch_size,
        cache_dir=args.io.cache_dir,
        prefix=args.data.prefix,
        overwrite_cache=args.data.overwrite_cache,
        num_workers=args.data.num_workers,
        split='test',
        distributed=False)
    output_dir = create_dir(Path(args.io.output))
    output_filename = output_dir / 'decoded.txt'
    output_filename.unlink(missing_ok=True)
    for step, batch in enumerate(
            tqdm.tqdm(data_loader,
                      desc='Decoding')):
        inputs = batch['input_ids'].to(
            device, non_blocking=True)
        # print(tokenizer.batch_decode(inputs, skip_special_tokens=True))
        # break
        if args.decode.method == 'greedy':
            outputs = decoder(model, inputs,
                              max_length=args.decode.max_length,
                              break_tokens=break_tokens)
        elif args.decode.method == 'beam':
            outputs = decoder(model, inputs,
                              beam_depth=args.decode.beam_depth,
                              beam_width=args.decode.beam_width)
        text = tokenizer.batch_decode(
            outputs, skip_special_tokens=True)
        for i, line in enumerate(text):
            output_filename.open('a+').write(line + '\n')
