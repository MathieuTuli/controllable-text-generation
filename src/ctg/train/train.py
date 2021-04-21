"""
@author: Mathieu Tuli
@github: MathieuTuli
@email: tuli.mathieu@gmail.com
"""
from argparse import Namespace
from pathlib import Path

from transformers import DataCollatorForSeq2Seq, default_data_collator
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

import torch

from ..data.utils import load_seq2seq_data
from ..models import get_tokenizer_model
from ..utils.logging import logger
from ..utils.io import create_dir


def main(args: Namespace) -> None:
    args.ngpus_per_node = ngpus_per_node = torch.cuda.device_count() if \
        args.gpu is None else 1 if isinstance(args.gpu, int) else len(args.gpu)
    args.distributed = (args.mpd or args.world_size > 1) and not args.cpu
    if args.gpu is None and not args.distributed:
        raise ValueError("Must specify gpu or distributed")
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"NGPUS: {ngpus_per_node}")
    tokenizer, model = get_tokenizer_model(
        tokenizer_name=args.train.tokenizer,
        model_name=args.train.model,
        cache_dir=str(create_dir(args.io.cache_dir)))
    train_dataset = load_seq2seq_data(
        fname=Path(args.data.train_src),
        tokenizer=tokenizer,
        # model=model,
        cache_dir=args.io.cache_dir,
        max_samples=args.data.max_train_samples,
        max_src_length=args.train.max_src_length,
        max_tgt_length=args.train.max_tgt_length,
        pad_to_max_length=args.data.pad_to_max_length,
        batch_size=args.train.batch_size,
        num_workers=args.data.num_workers,
        overwrite_cache=args.data.overwrite_cache,
        ignore_pad_for_loss=args.train.ignore_pad_for_loss,
        split='train',
        distributed=args.distributed)
    val_dataset = load_seq2seq_data(
        fname=Path(args.data.val_src),
        tokenizer=tokenizer,
        # model=model,
        cache_dir=args.io.cache_dir,
        max_samples=args.data.max_val_samples,
        max_src_length=args.train.max_src_length,
        max_tgt_length=args.train.max_tgt_length,
        pad_to_max_length=args.data.pad_to_max_length,
        batch_size=args.train.batch_size,
        num_workers=args.data.num_workers,
        overwrite_cache=args.data.overwrite_cache,
        ignore_pad_for_loss=args.train.ignore_pad_for_loss,
        split='val',
        distributed=args.distributed)
    if args.data.pad_to_max_length:
        collator = default_data_collator
    else:
        collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100 if args.train.ignore_pad_for_loss else
            tokenizer.pad_token_id,)
    trainer_args = Seq2SeqTrainingArguments(
        output_dir=str(create_dir(args.io.output)),
        do_train=True,
        do_eval=True,
        evluation_strategy='epoch',
        per_device_train_batch_size=args.train.batch_size,
        per_device_eval_batch_size=args.train.batch_size if
        args.train.val_batch_size is None else args.train.val_batch_size,
        learning_rate=args.train.optimizer_kwargs['lr'],
        weight_decay=args.train.optimizer_kwargs['weight_decay'],
        lr_scheduler_type=args.train.scheduler,
        warmup_ratio=args.train.scheduler_kwargs['warmup_ratio'],
        warmup_steps=args.train.scheduler_kwargs['warmup_steps'],
        num_train_epochs=args.train.max_epochs,
        save_steps=args.io.save_freq *
        int(len(train_dataset) / args.train.batch_size)
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collator)

    train_result = trainer.train()
