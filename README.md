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
pip install transformers==4.42.4
pip install deepspeed==0.14.4

# To execute
accelerate launch main.py

