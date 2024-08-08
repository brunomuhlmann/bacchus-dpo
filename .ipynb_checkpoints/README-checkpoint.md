# Bacchus DPO

- Create `env.json` file
```bash
touch env.json
```
- Add HF_TOKEN
```json
{
    "HF_TOKEN": "your_hf_token_here"
}
```

# setup
pip install flash-attn --no-build-isolation
pip install deepspeed
pip install trl==0.9.6

# To execute
accelerate launch main.py

