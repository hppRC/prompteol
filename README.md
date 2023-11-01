# prompteol

## Instllation

```
rye sync
```


## Usage

```
# Use 2 GPUs (cuda:0, cuda:1)
accelerate launch --config_file accelerate.json src/train.py

# Use 2 GPUs (cuda:2, cuda:3)
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --config_file accelerate.json src/train.py

# Use 4 GPUs
accelerate launch --config_file accelerate.json --num_processes 4 src/train.py
```