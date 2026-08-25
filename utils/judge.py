# ============================================================
# utils/judge.py
# Loads the LLM-as-Judge model with automatic device selection.
#
# Device strategy:
#   - If GPU has >= 3GB free VRAM after other models load → GPU
#   - Otherwise → CPU (slower but always works)
#
# The judge is loaded ONCE at server startup and reused for
# every evaluation — no reloading per request.
# ============================================================

import os
import gc
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

try:
    from utils.config import cfg
except ImportError:
    from config import cfg


JUDGE_NAME      = cfg["judge"]["judge_name"]
JUDGE_MAX_TOKS  = cfg["judge"]["max_new_tokens"]
JUDGE_TEMP      = cfg["judge"]["temperature"]
JUDGE_4BIT      = cfg["judge"].get("load_in_4bit", True)

# Min free VRAM (in GB) required to put the judge on GPU.
# Llama 3.2 3B in 4-bit uses ~2GB; require a small safety buffer.
MIN_GPU_FREE_GB = 2.5


def _detect_device() -> str:
    """
    Decide whether the judge should run on 'cuda' or 'cpu'
    based on current free VRAM.
    """
    if not torch.cuda.is_available():
        print("[Judge] No CUDA device — using CPU.")
        return "cpu"

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb  = free_bytes  / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    print(f"[Judge] GPU free: {free_gb:.2f} GB / {total_gb:.2f} GB total")

    if free_gb >= MIN_GPU_FREE_GB:
        print(f"[Judge] Free VRAM sufficient — using GPU.")
        return "cuda"
    else:
        print(f"[Judge] Free VRAM < {MIN_GPU_FREE_GB} GB — "
              f"running judge on CPU.")
        return "cpu"


def load_judge():
    """
    Load the judge LLM. Auto-detects GPU vs CPU based on
    available VRAM. Returns (model, tokenizer, device).
    """
    device = _detect_device()
    print(f"[Judge] Loading judge model: {JUDGE_NAME} on {device.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(JUDGE_NAME)

    if device == "cuda" and JUDGE_4BIT:
        # GPU path — 4-bit quantisation
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            JUDGE_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    elif device == "cuda":
        # GPU path — fp16, no quantisation
        model = AutoModelForCausalLM.from_pretrained(
            JUDGE_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    else:
        # CPU path — bitsandbytes does NOT work on CPU, use fp32
        model = AutoModelForCausalLM.from_pretrained(
            JUDGE_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    print(f"[Judge] Judge model loaded on {device.upper()}.")
    return model, tokenizer, device


def unload_model(model, tokenizer):
    """Free memory used by a loaded model."""
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("[Judge] Model unloaded, memory freed.")


def run_judge(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = None,
    temperature: float = None,
) -> str:
    """
    Run a single inference call on the judge.
    Works on both CPU and GPU — device is determined by where
    the model was loaded.
    """
    max_new_tokens = max_new_tokens or JUDGE_MAX_TOKS
    temperature    = temperature    or JUDGE_TEMP

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Move inputs to the same device as the model
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0.0),
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ============================================================
# Pluggable judge abstraction (task 21)
# Decouples evaluation from the generator so an INDEPENDENT judge
# (a separate local model, or an OpenAI-compatible API such as
# Azure GPT-5-nano or an HPC-hosted vLLM endpoint) can be dropped in
# via config, instead of the generator grading its own output.
# ============================================================
class Judge:
    """Uniform judge interface: generate(system, user) -> raw text.
    `name` is recorded in the evaluation provenance."""
    def __init__(self, name: str, backend: str):
        self.name = name
        self.backend = backend

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class LocalJudge(Judge):
    """Judge backed by a local HuggingFace model + tokenizer."""
    def __init__(self, model, tokenizer, name: str, backend: str = "local"):
        super().__init__(name, backend)
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return run_judge(self.model, self.tokenizer,
                         system_prompt, user_prompt)


class APIJudge(Judge):
    """Independent judge behind an OpenAI-compatible chat API
    (Azure OpenAI, OpenAI, or a self-hosted vLLM/HPC endpoint).
    Endpoint/model come from config.judge.api; the key from an env var."""
    def __init__(self, model: str, base_url: str, api_key: str,
                 name: str = None, temperature: float = 0.1,
                 max_tokens: int = 512, api_version: str = None):
        super().__init__(name or f"api:{model}", "api")
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_version = api_version     # set -> Azure OpenAI
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            if self.api_version:           # Azure OpenAI endpoint
                from openai import AzureOpenAI
                self._client = AzureOpenAI(
                    azure_endpoint=self.base_url,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
            else:                          # OpenAI / Anthropic-compat / vLLM
                from openai import OpenAI
                self._client = OpenAI(base_url=self.base_url or None,
                                      api_key=self.api_key or "not-needed")
        return self._client

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        client = self._client_lazy()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        try:
            resp = client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=self.temperature, max_tokens=self.max_tokens)
        except Exception:
            # Some newer models reject temperature/max_tokens; retry minimal.
            resp = client.chat.completions.create(
                model=self.model, messages=messages)
        return (resp.choices[0].message.content or "").strip()


def build_judge(cfg, generator_model=None, generator_tokenizer=None) -> Judge:
    """Construct the judge from config.judge.backend:
        api              -> independent OpenAI-compatible endpoint
        local            -> a separate local model (config judge_name)
        reuse_generator  -> the generator itself (self-bias; dev only)
    """
    jcfg    = cfg.get("judge", {})
    backend = jcfg.get("backend", "reuse_generator")

    if backend == "api":
        api = jcfg.get("api", {})
        key = os.getenv(api.get("api_key_env", "JUDGE_API_KEY"), "")
        print(f"[Judge] Backend=api  model={api.get('model')} (independent)")
        return APIJudge(
            model=api.get("model", "gpt-5-nano"),
            base_url=api.get("base_url", ""),
            api_key=key,
            name=api.get("name") or f"api:{api.get('model','')}",
            temperature=jcfg.get("temperature", 0.1),
            max_tokens=jcfg.get("max_new_tokens", 512),
            api_version=api.get("api_version") or None,
        )

    if backend == "local":
        print("[Judge] Backend=local (separate model)")
        model, tokenizer, _ = load_judge()
        return LocalJudge(model, tokenizer,
                          name=jcfg.get("judge_name", "local-judge"))

    # default — reuse the generator (SAME model grades itself)
    print("[Judge] Backend=reuse_generator — WARNING: self-bias, the "
          "generator is grading its own output.")
    gname = cfg.get("model", {}).get("llm_name", "generator")
    return LocalJudge(generator_model, generator_tokenizer,
                      name=f"{gname} (reused as judge)",
                      backend="reuse_generator")


# ============================================================
# TEST — same as before
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("JUDGE LOAD TEST (auto-detect)")
    print("=" * 60)

    model, tokenizer, device = load_judge()

    system = "You are a careful evaluator. Respond with exactly one short sentence."
    user = "In one sentence, what does 'faithful to context' mean in RAG?"

    print(f"\n[Judge] Running smoke test on {device}...")
    output = run_judge(model, tokenizer, system, user, max_new_tokens=80)
    print("\n--- Judge output ---")
    print(output)
    print("--------------------")

    unload_model(model, tokenizer)
    print("\nJudge load test complete.")