# Unofficial Re-Implementeation of PromptEOL

Please refer [the exciting original paper](https://arxiv.org/abs/2307.16645) and [the original implementation](https://github.com/kongds/scaling_sentemb).


## Instllation

```
rye sync

# load .envrc
direnv allow
```


## Usage

We have three implementations.

- `src/qlora.py`: Fine-tuning with QLoRA (w/ 4 or 8 bit quantization)
- `src/lora.py`: Fine-tuning with LoRA (w/o quantization)
- `src/full.py`: Full-parameter fine-tuning

```
# Use 2 GPUs (cuda:0, cuda:1)
accelerate launch --config_file accelerate.json src/qlora.py

# Use 2 GPUs (cuda:2, cuda:3)
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --config_file accelerate.json src/qlora.py

# Use 4 GPUs
accelerate launch --config_file accelerate.json --num_processes 4 src/qlora.py
```

One of the most significant differences from the original implementation is that we evaluate at regular steps during training, leaving only the best performing checkpoints to reduce storage space and effort.

Also, in our code, we omitted evaluation by SentEval for simplicity.