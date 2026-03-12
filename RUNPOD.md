# Training nanochat on RunPod

## Instance Setup

1. Go to console.runpod.io > Deploy a Pod
2. Select **H100 SXM** ($2.69/hr per GPU, $22/hr for 8x)
3. Use the **runpod-torch** template
4. Ensure enough disk space (default may be too small — increase volume size)
5. Launch and open JupyterLab terminal

## Environment Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

# Clone repo and install dependencies
git clone https://github.com/thotchkiss/nanochat.git && cd nanochat
uv sync --extra gpu

# Install Python dev headers (missing on RunPod base image, needed for FP8 compilation)
apt update && apt install -y python3-dev
```

## Training Pipeline

```bash
# 1. Download training data (~170 shards, ~17GB)
uv run python -m nanochat.dataset -n 170

# 2. Train tokenizer
uv run python -m scripts.tok_train

# 3. Pretrain base model (~2 hours on 8xH100)
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --device-batch-size=16 --fp8

# 4. Download identity conversations for SFT
curl -L -o ~/.cache/nanochat/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# 5. Fine-tune for chat (SFT)
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16

# 6. Evaluate chat model
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft
```

## Downloading the Model

Package checkpoints and tokenizer:
```bash
cd ~/.cache/nanochat
tar czf /workspace/nanochat-model.tar.gz chatsft_checkpoints/ base_checkpoints/ tokenizer/
```

Download `nanochat-model.tar.gz` from JupyterLab file browser (navigate to `/workspace/`).

**Terminate the pod immediately after downloading to stop billing.**

## Using the Model Locally

Extract to your local cache directory:
```bash
# Windows (Git Bash)
tar xzf nanochat-model.tar.gz -C ~/.cache/nanochat/
```

Run chat locally:
```bash
python -m scripts.chat_cli -p "Hello!"
python -m scripts.chat_web
```

## Notes

- tmux and screen are not available on the RunPod image — keep the browser tab open during training
- If training is interrupted, nanochat saves checkpoints and can resume
- The hardlink warning during `uv sync` is harmless
- Estimated cost for full pipeline (8xH100): ~$44-50 (~2 hours training + setup time)
- Checkpoints are saved to `~/.cache/nanochat/` on the pod
