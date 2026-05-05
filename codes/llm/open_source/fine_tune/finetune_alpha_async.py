"""
finetune_eval.py  --  LoRA fine-tuning + evaluation on alpha-asynchronous ECA inverse problem.

- Fine-tunes on phase2/train (179 training rules)
- Evaluates on phase2/test using SAME fixed indices as zero_shot_eval.py and few_shot_eval.py
- Resume support: skips training if adapter exists; skips already-evaluated test samples
- 70B models loaded in 8-bit to fit in 2x H100

Usage:
  python finetune_eval.py --model Qwen2.5-7B-Instruct --n_train 5000 --n_test 500 --epochs 3
  python finetune_eval.py --model all --n_train 5000 --n_test 500 --epochs 3

Requirements:
  pip install peft accelerate bitsandbytes
"""

import os, json, time, argparse, re
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR   = os.environ.get("BASE_DIR",   os.path.join(os.getcwd(), "ECA_alpha_async"))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.getcwd(), "models"))
DATA_DIR    = os.path.join(BASE_DIR, "ECA_Data_New")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "finetune")
CKPT_DIR    = os.path.join(BASE_DIR, "checkpoints", "finetune")
INDEX_FILE  = os.path.join(BASE_DIR, "results", "fixed_test_indices.npy")
SEED        = 42

ALPHA_VALUES = [round(a * 0.1, 1) for a in range(1, 11)]
ALPHA_TOL    = 0.05
W, T         = 20, 100

ALL_MODELS = [
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct",
    "Mistral-7B-Instruct-v0.3",
    "Mixtral-8x7B-Instruct-v0.1",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-72B-Instruct",
]

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

SYSTEM_PROMPT = """You are an expert in cellular automata. You are given a space-time orbit of an Elementary Cellular Automaton (ECA) perturbed by alpha-asynchronous noise.

In alpha-asynchronous noise each cell independently updates with probability alpha per timestep.
Alpha is one of: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

The orbit is a grid of 0s and 1s. Each row is one timestep.

Identify:
1. The ECA rule number (integer 0-255)
2. The alpha value

Respond ONLY in this JSON format with no extra text:
{"rule": <integer>, "alpha": <float>}"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def orbit_to_str(orbit):
    return "\n".join("".join(str(int(c)) for c in row) for row in orbit)

def make_user_prompt(orbit):
    return (f"Space-time orbit ({T} rows x {W} cells):\n\n"
            f"{orbit_to_str(orbit)}\n\nWhat is the ECA rule number and alpha value?")

def make_target(rule, alpha):
    return json.dumps({"rule": int(rule), "alpha": round(float(alpha), 1)})

def parse_response(response):
    try:
        s = response.find("{"); e = response.rfind("}") + 1
        if s != -1 and e > s:
            d     = json.loads(response[s:e])
            rule  = int(d.get("rule", -1))
            alpha = float(d.get("alpha", -1))
            if 0 <= rule <= 255:
                alpha = min(ALPHA_VALUES, key=lambda a: abs(a - alpha))
                return rule, alpha
    except Exception:
        pass
    rm = re.search(r'"rule"\s*:\s*(\d+)',     response)
    am = re.search(r'"alpha"\s*:\s*([\d.]+)', response)
    if rm and am:
        rule  = int(rm.group(1))
        alpha = float(am.group(1))
        if 0 <= rule <= 255:
            alpha = min(ALPHA_VALUES, key=lambda a: abs(a - alpha))
            return rule, alpha
    return None, None

# ── Data ──────────────────────────────────────────────────────────────────────

def get_fixed_test_indices(n_samples):
    """Load shared fixed test indices (same as zero_shot and few_shot)."""
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    if os.path.exists(INDEX_FILE):
        idx = np.load(INDEX_FILE)
        return idx[:n_samples]
    rng    = np.random.default_rng(SEED)
    orbits = np.load(os.path.join(DATA_DIR, "phase2", "test", "orbits.npy"))
    n      = min(n_samples, len(orbits))
    idx    = rng.choice(len(orbits), size=n, replace=False)
    np.save(INDEX_FILE, idx)
    return idx

def load_train_data(n_train):
    rng      = np.random.default_rng(SEED + 1)
    orbits   = np.load(os.path.join(DATA_DIR, "phase2", "train", "orbits.npy"))
    rule_ids = np.load(os.path.join(DATA_DIR, "phase2", "train", "rule_ids.npy"))
    alphas   = np.load(os.path.join(DATA_DIR, "phase2", "train", "alphas.npy"))
    n        = min(n_train, len(orbits))
    idx      = rng.choice(len(orbits), size=n, replace=False)
    return orbits[idx], rule_ids[idx], alphas[idx]

def load_test_data(n_test):
    idx      = get_fixed_test_indices(n_test)
    orbits   = np.load(os.path.join(DATA_DIR, "phase2", "test", "orbits.npy"))
    rule_ids = np.load(os.path.join(DATA_DIR, "phase2", "test", "rule_ids.npy"))
    alphas   = np.load(os.path.join(DATA_DIR, "phase2", "test", "alphas.npy"))
    return orbits[idx], rule_ids[idx], alphas[idx]

# ── Dataset ───────────────────────────────────────────────────────────────────

class ECADataset(Dataset):
    def __init__(self, orbits, rule_ids, alphas, tok, max_length=2048):
        self.samples    = []
        self.tok        = tok
        self.max_length = max_length
        print(f"  Building dataset ({len(orbits)} samples) ...")
        for orbit, rule, alpha in zip(orbits, rule_ids, alphas):
            self.samples.append([
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": make_user_prompt(orbit)},
                {"role": "assistant", "content": make_target(rule, alpha)},
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        messages = self.samples[idx]
        try:
            full_text   = self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            prompt_text = self.tok.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True)
        except Exception:
            parts = []
            for m in messages:
                if m["role"] == "system":      parts.append(f"System: {m['content']}")
                elif m["role"] == "user":      parts.append(f"User: {m['content']}")
                elif m["role"] == "assistant": parts.append(f"Assistant: {m['content']}")
            full_text   = "\n\n".join(parts)
            prompt_text = full_text.rsplit("Assistant:", 1)[0] + "Assistant:"

        full_enc   = self.tok(full_text,   truncation=True,
                              max_length=self.max_length, return_tensors="pt")
        prompt_enc = self.tok(prompt_text, truncation=True,
                              max_length=self.max_length, return_tensors="pt")

        input_ids = full_enc["input_ids"][0]
        labels    = input_ids.clone()
        labels[:prompt_enc["input_ids"].shape[1]] = -100  # mask prompt from loss

        return {"input_ids":      input_ids,
                "attention_mask": full_enc["attention_mask"][0],
                "labels":         labels}

class PadCollator:
    def __init__(self, tok):
        self.tok = tok
    def __call__(self, batch):
        max_len = max(b["input_ids"].shape[0] for b in batch)
        ids, masks, labels = [], [], []
        for b in batch:
            pad = max_len - b["input_ids"].shape[0]
            ids.append(torch.cat([b["input_ids"],
                                  torch.full((pad,), self.tok.pad_token_id)]))
            masks.append(torch.cat([b["attention_mask"],
                                    torch.zeros(pad, dtype=torch.long)]))
            labels.append(torch.cat([b["labels"],
                                     torch.full((pad,), -100)]))
        return {"input_ids":      torch.stack(ids),
                "attention_mask": torch.stack(masks),
                "labels":         torch.stack(labels)}

# ── Model loading ─────────────────────────────────────────────────────────────

def is_large(model_name):
    """Check if model is 70B/72B scale by name string."""
    return any(x in model_name for x in ["70B", "72B"])

def load_model_for_training(model_name):
    path = os.path.join(MODELS_DIR, model_name)
    print(f"  Loading {model_name} for training ...")
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    if is_large(model_name):
        print(f"  70B model -- loading in 8-bit for LoRA")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    return tok, model

def load_finetuned_for_eval(model_name, adapter_path):
    path = os.path.join(MODELS_DIR, model_name)
    print(f"  Loading fine-tuned {model_name} for evaluation ...")
    tok = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if is_large(model_name):
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        base = AutoModelForCausalLM.from_pretrained(
            path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return tok, model

# ── Training ──────────────────────────────────────────────────────────────────

def train(model, tok, train_dataset, ckpt_dir, epochs, batch_size, model_name):
    # FIX: model_name (str) is now passed explicitly so is_large() works correctly
    # instead of accidentally receiving the PeftModelForCausalLM object.

    # Resume from last HF trainer checkpoint if available
    last_ckpt = None
    if os.path.isdir(ckpt_dir):
        ckpts = sorted([d for d in os.listdir(ckpt_dir)
                        if d.startswith("checkpoint-")],
                       key=lambda x: int(x.split("-")[1]))
        if ckpts:
            last_ckpt = os.path.join(ckpt_dir, ckpts[-1])
            print(f"  Resuming training from: {last_ckpt}")

    args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 8 // batch_size),
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=not is_large(model_name),   # FIX: use model_name string, not model object
        bf16=is_large(model_name),       # FIX: use model_name string, not model object
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=PadCollator(tok),
    )
    trainer.train(resume_from_checkpoint=last_ckpt)

    adapter_path = os.path.join(ckpt_dir, "lora_adapter")
    model.save_pretrained(adapter_path)
    tok.save_pretrained(adapter_path)
    print(f"  Adapter saved: {adapter_path}")
    return adapter_path

# ── Eval checkpoint ───────────────────────────────────────────────────────────

def load_eval_checkpoint(path):
    if not os.path.exists(path):
        return [], set()
    try:
        with open(path) as f:
            data = json.load(f)
        samples = data.get("samples", [])
        done    = {s["sample_idx"] for s in samples}
        print(f"  Eval resume: {len(done)} samples done.")
        return samples, done
    except Exception:
        return [], set()

def save_eval_checkpoint(path, model_name, n_total, results):
    n         = len(results)
    rule_acc  = sum(r["rule_correct"] for r in results) / n * 100 if n else 0
    alpha_acc = sum(r["alpha_ok"]     for r in results) / n * 100 if n else 0
    with open(path, "w") as f:
        json.dump({
            "model":          model_name,
            "mode":           "finetuned",
            "n_evaluated":    n,
            "n_total":        n_total,
            "rule_accuracy":  rule_acc,
            "alpha_accuracy": alpha_acc,
            "parse_failures": sum(1 for r in results if r["pred_rule"] is None),
            "timestamp":      datetime.now().isoformat(),
            "samples":        results,
        }, f, indent=2)

# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(tok, model, orbit, max_new_tokens=64):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": make_user_prompt(orbit)},
    ]
    try:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = (f"System: {SYSTEM_PROMPT}\n\n"
                f"User: {make_user_prompt(orbit)}\n\nAssistant:")
    inputs = tok(text, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=1.0,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()

# ── Run one model ─────────────────────────────────────────────────────────────

def run_model(model_name, n_train, n_test, epochs, batch_size):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_ckpt = os.path.join(CKPT_DIR, model_name)
    os.makedirs(model_ckpt, exist_ok=True)
    adapter_path = os.path.join(model_ckpt, "lora_adapter")
    result_path  = os.path.join(RESULTS_DIR, f"{model_name}_finetuned.json")

    print(f"\n{'='*65}\n  {model_name}  |  LoRA fine-tune + eval\n{'='*65}")

    # ── Step 1: Train (skip if adapter already saved) ─────────────────────────
    if os.path.exists(adapter_path):
        print(f"  Adapter found at {adapter_path}. Skipping training.")
    else:
        print(f"\n  Loading training data ({n_train} samples) ...")
        tr_orbits, tr_rules, tr_alphas = load_train_data(n_train)
        tok, model = load_model_for_training(model_name)
        train_ds   = ECADataset(tr_orbits, tr_rules, tr_alphas, tok)
        t0 = time.time()
        # FIX: pass model_name explicitly so train() can call is_large(model_name)
        adapter_path = train(model, tok, train_ds, model_ckpt, epochs, batch_size, model_name)
        print(f"  Training time: {(time.time()-t0)/3600:.2f}h")
        del model; torch.cuda.empty_cache()

    # ── Step 2: Evaluate on held-out test set (with resume) ───────────────────
    print(f"\n  Loading test data ({n_test} samples, same as zero/few-shot) ...")
    te_orbits, te_rules, te_alphas = load_test_data(n_test)

    results, done_set = load_eval_checkpoint(result_path)
    remaining = [(i, te_orbits[i], te_rules[i], te_alphas[i])
                 for i in range(len(te_orbits)) if i not in done_set]

    if not remaining:
        print("  All test samples already evaluated.")
        return

    tok, model = load_finetuned_for_eval(model_name, adapter_path)
    t0 = time.time()

    for pos, (i, orbit, true_rule, true_alpha) in enumerate(remaining):
        raw               = run_inference(tok, model, orbit)
        pred_rule, pred_alpha = parse_response(raw)
        r_ok = pred_rule == int(true_rule) if pred_rule is not None else False
        a_ok = abs(pred_alpha - float(true_alpha)) <= ALPHA_TOL if pred_alpha is not None else False

        results.append({
            "sample_idx":   i,
            "true_rule":    int(true_rule),
            "pred_rule":    pred_rule,
            "true_alpha":   float(true_alpha),
            "pred_alpha":   pred_alpha,
            "rule_correct": r_ok,
            "alpha_ok":     a_ok,
            "raw_output":   raw,
        })

        if (pos + 1) % 10 == 0 or pos + 1 == len(remaining):
            save_eval_checkpoint(result_path, model_name, n_test, results)
            done      = len(results)
            eta       = (time.time() - t0) / (pos + 1) * (len(remaining) - pos - 1)
            rule_acc  = sum(r["rule_correct"] for r in results) / done * 100
            alpha_acc = sum(r["alpha_ok"]     for r in results) / done * 100
            print(f"  [{done:>4}/{n_test}]  rule={rule_acc:.1f}%  "
                  f"alpha={alpha_acc:.1f}%  ETA={eta:.0f}s", flush=True)

    n         = len(results)
    rule_acc  = sum(r["rule_correct"] for r in results) / n * 100
    alpha_acc = sum(r["alpha_ok"]     for r in results) / n * 100
    print(f"\n  FINAL  rule={rule_acc:.2f}%  alpha={alpha_acc:.2f}%")
    print(f"  Saved: {result_path}")
    del model; torch.cuda.empty_cache()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="Qwen2.5-7B-Instruct")
    parser.add_argument("--n_train",    type=int, default=5000)
    parser.add_argument("--n_test",     type=int, default=500)
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    args   = parser.parse_args()
    models = ALL_MODELS if args.model == "all" else [args.model]
    for m in models:
        run_model(m, args.n_train, args.n_test, args.epochs, args.batch_size)

if __name__ == "__main__":
    main()
