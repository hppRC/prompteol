import logging
import os
import warnings
from dataclasses import dataclass

import datasets as ds
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from mteb import MTEB
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    BatchEncoding,
    HfArgumentParser,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)
from transformers import Trainer as HFTrainer


class Batch:
    anc: BatchEncoding
    pos: BatchEncoding
    neg: BatchEncoding


@dataclass
class Args(TrainingArguments):
    output_dir: str = None
    data_path: str = "data/nli_for_simcse.csv"
    model_name: str = "meta-llama/Llama-2-7b-hf"

    learning_rate: float = 5e-4
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 128
    warmup_ratio: float = 0.1
    temperature: float = 0.05

    max_seq_len: int = 32

    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    logging_steps: int = 10
    bf16: bool = True

    evaluation_strategy: str = "steps"
    eval_steps: int = 10

    save_strategy: str = "steps"
    save_steps: int = 10
    save_total_limit: int = 1

    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = True

    report_to: str = "none"
    ddp_find_unused_parameters: bool = False

    load_best_model_at_end: bool = False  # This is importnant for preventing hangup
    remove_unused_columns: bool = False
    metric_for_best_model: str = "spearman"

    optim: str = "paged_adamw_32bit"


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizer
    max_seq_len: int
    template: str = 'This sentence: "{text}" means in one word: "'

    def process(self, texts: list[str]) -> BatchEncoding:
        texts = [self.template.format(text=t) for t in texts]
        return self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",  # fix tensor's shape for performance with `torch.compile``
            return_tensors="pt",
            max_length=self.max_seq_len,
        )

    def __call__(self, batch: list[dict]) -> BatchEncoding[str, BatchEncoding]:
        return BatchEncoding(
            {
                "anc": self.process([d["sent0"] for d in batch]),
                "pos": self.process([d["sent1"] for d in batch]),
                "neg": self.process([d["hard_neg"] for d in batch]),
            }
        )


class Trainer(HFTrainer):
    args: Args
    data_collator: DataCollator

    def gather_and_replace(self, x: torch.Tensor) -> torch.Tensor:
        # This gathered tensor don't have gradients unfortunatelly.
        # As well as PromptEOL's original code, we inject the original tensor and its gradients to the gathered tensor.
        batch_size: int = x.size(0)
        pid: int = self.accelerator.local_process_index
        gathered = self.accelerator.gather(x)
        gathered[batch_size * pid : batch_size * (pid + 1)] = x
        return gathered

    @torch.compile
    def calc_loss(self, anc: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        sim_mat_1st = F.cosine_similarity(anc.unsqueeze(1), pos.unsqueeze(0), dim=-1)
        sim_mat_2nd = F.cosine_similarity(anc.unsqueeze(1), neg.unsqueeze(0), dim=-1)

        sim_mat = torch.cat([sim_mat_1st, sim_mat_2nd], dim=1)
        sim_mat = sim_mat / self.args.temperature

        labels = torch.arange(sim_mat.size(0)).long().to(sim_mat.device)
        return F.cross_entropy(sim_mat, labels)

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Batch,
        return_outputs: bool = False,
    ) -> torch.Tensor:
        anc: torch.Tensor = model(**inputs.anc).last_hidden_state[:, -1]
        pos: torch.Tensor = model(**inputs.pos).last_hidden_state[:, -1]
        neg: torch.Tensor = model(**inputs.neg).last_hidden_state[:, -1]

        # gather from all processes
        # if you have 2 GPUs, anc, pos, and neg will be of size (batch_size * 2, dim)
        anc = self.gather_and_replace(anc)
        pos = self.gather_and_replace(pos)
        neg = self.gather_and_replace(neg)

        loss = self.calc_loss(anc, pos, neg)

        return (loss, anc) if return_outputs else loss

    @torch.no_grad()
    def evaluate(self, **kwargs):
        if self.accelerator.is_main_process:
            mteb = MTEB(tasks=["STSBenchmark"])
            mteb.print_selected_tasks = lambda: None  # supress logging

            try:
                result = mteb.run(self, eval_splits=["validation"], output_folder=None, verbosity=0)
                metrics = {
                    "eval_spearman": result["STSBenchmark"]["validation"]["cos_sim"]["spearman"]
                }
            except Exception:
                metrics = {"eval_spearman": 0.0}

            self.log(metrics)
            return metrics

    @torch.no_grad()
    def encode(self, sentences: list[str], **kwargs):
        data_loader = DataLoader(
            sentences,
            collate_fn=self.data_collator.process,
            batch_size=self.args.per_device_train_batch_size,
            num_workers=0,
            pin_memory=True,
        )

        model: PreTrainedModel = self.accelerator.unwrap_model(self.model).eval()
        device = self.accelerator.device

        embs = []
        for batch in data_loader:
            emb = model(**batch.to(device)).last_hidden_state[:, -1]
            embs.append(emb.cpu().float())
        embs = torch.cat(embs, dim=0)
        return embs

    def save_best_model(self):
        if self.accelerator.is_main_process:
            self._load_best_model()
            self.save_model(args.output_dir)


def find_all_linear_names(model: nn.Module, num_bits: int) -> list[str]:
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names = name.split(".")
            lora_module_names.add(names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    return sorted(list(lora_module_names))


def main(args: Args):
    use_cache = not args.gradient_checkpointing
    model: PreTrainedModel = AutoModel.from_pretrained(
        args.model_name,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
        use_cache=use_cache,
    )
    model.config.use_cache = use_cache

    target_modules = find_all_linear_names(model, args.num_bits)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )

    # we can compile our model only with `use_reentrant=False`
    if not args.gradient_checkpointing_use_reentrant:
        model = torch.compile(model)
    model = get_peft_model(model, lora_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset: ds.Dataset = ds.load_dataset(
        "csv", data_files=args.data_path, split="train"
    ).shuffle()
    data_collator = DataCollator(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
    )

    trainer = Trainer(
        args=args,
        model=model,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_best_model()


if __name__ == "__main__":
    logging.disable(logging.FATAL)
    warnings.filterwarnings("ignore")

    parser = HfArgumentParser(Args)
    args: Args = parser.parse_args_into_dataclasses()[0]
    if args.output_dir is None:
        model_name = args.model_name.replace("/", "__")
        args.output_dir = f"outputs/{model_name}/lora"

    # monkey patching
    original_checkpoint = torch.utils.checkpoint.checkpoint
    use_reentrant = args.gradient_checkpointing_use_reentrant

    def checkpoint(*args, use_reentrant=False, **kwargs):
        return original_checkpoint(*args, **kwargs, use_reentrant=use_reentrant)

    torch.utils.checkpoint.checkpoint = checkpoint

    # to prevent too many re-compiling
    import torch._dynamo.config

    torch._dynamo.config.cache_size_limit = 4

    main(args)
