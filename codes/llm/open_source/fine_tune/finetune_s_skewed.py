"""
finetune_skew.py  --  LoRA fine-tuning + evaluation on s-skewed ECA inverse problem.

- Fine-tunes on phase2/train (179 training rules, all 20 s values)
- Evaluates on phase2/test using same fixed indices as zero/few-shot
- Resume support throughout

Usage:
  python finetune_skew.py --model Qwen2.5-7B-Instruct --n_train 5000 --n_test 500 --epochs 3
  python finetune_skew.py --model all --n_train 5000 --n_test 500 --epochs 3
"""

import os, json, time, argparse, re
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR   = os.environ.get("BASE_DIR",   os.path.join(os.getcwd(), "ECA_s_skewed"))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(os.getcwd(), "models"))
DATA_DIR    = os.path.join(BASE_DIR, "ECA_Data_Skew")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "finetune")
CKPT_DIR    = os.path.join(BASE_DIR, "checkpoints", "finetune")
INDEX_FILE  = os.path.join(BASE_DIR, "results", "fixed_test_indices.npy")
SEED        = 42

S_VALUES = list(range(1, 21))
S_TOL    = 1
W, T     = 20, 200

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

SYSTEM_PROMPT = """You are an expert in cellular automata. You are given a space-time orbit of an Elementary Cellular Automaton (ECA) perturbed by s-skewed noise.

In s-skewed noise, at each timestep exactly s contiguous cells are chosen at random and updated according to the ECA rule. All other cells retain their current state. The parameter s ranges from 1 to 20 (the full grid width), where s=20 means all cells update (fully synchronous).

The orbit is a grid of 0s and 1s. Each row is one timestep. Grid width is 20 cells.

Identify:
1. The ECA rule number (integer 0-255)
2. The value of s (integer from 1 to 20)

Respond ONLY in this JSON format with no extra text:
{"rule": <integer>, "s": <integer>}"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def orbit_to_str(orbit):
    return "\n".join("".join(str(int(c)) for c in row) for row in orbit)

def make_user_prompt(orbit):
    return (f"Space-time orbit ({T} rows x {W} cells):\n\n"
            f"{orbit_to_str(orbit)}\n\nWhat is the ECA rule number and s value?")

def make_target(rule, s):
    return json.dumps({"rule": int(rule), "s": int(s)})

def parse_response(response):
    try:
        st = response.find("{"); e = response.rfind("}") + 1
        if st != -1 and e > st:
            d    = json.loads(response[st:e])
            rule = int(d.get("rule", -1))
            sv   = int(round(float(d.get("s", -1))))
            if 0 <= rule <= 255 and 1 <= sv <= 20:
                return rule, sv
    except Exception:
        pass
    rm = re.search(r'"rule"\s*:\s*(\d+)', response)
    sm = re.search(r'"s"\s*:\s*(\d+)',    response)
    if rm and sm:
        rule = int(rm.group(1)); sv = int(sm.group(1))
        if 0 <= rule <= 255 and 1 <= sv <= 20:
            return rule, sv
    return None, None

# ── Data ──────────────────────────────────────────────────────────────────────

def get_fixed_test_indices(n_samples):
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    if os.path.exists(INDEX_FILE):
        return np.load(INDEX_FILE)[:n_samples]
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
    s_vals   = np.load(os.path.join(DATA_DIR, "phase2", "train", "s_values.npy"))
    n        = min(n_train, len(orbits))
    idx      = rng.choice(len(orbits), size=n, replace=False)
    return orbits[idx], rule_ids[idx], s_vals[idx]

def load_test_data(n_test):
    idx      = get_fixed_test_indices(n_test)
    orbits   = np.load(os.path.join(DATA_DIR, "phase2", "test", "orbits.npy"))
    rule_ids = np.load(os.path.join(DATA_DIR, "phase2", "test", "rule_ids.npy"))
    s_vals   = np.load(os.path.join(DATA_DIR, "phase2", "test", "s_values.npy"))
    return orbits[idx], rule_ids[idx], s_vals[idx]

# ── Dataset ───────────────────────────────────────────────────────────────────

class ECADataset(Dataset):
    def __init__(self, orbits, rule_ids, s_vals, tok, max_length=4096):
        self.samples    = []
        self.tok        = tok
        self.max_length = max_length
        print(f"  Building dataset ({len(orbits)} samples) ...")
        for orbit, rule, sv in zip(orbits, rule_ids, s_vals):
            self.samples.append([
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": make_user_prompt(orbit)},
                {"role": "assistant", "content": make_target(rule, sv)},
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
        input_ids  = full_enc["input_ids"][0]
        labels     = input_ids.clone()
        labels[:prompt_enc["input_ids"].shape[1]] = -100
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
            ids.append(torch.cat([b["input_ids"],    torch.full((pad,), self.tok.pad_token_id)]))
            masks.append(torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
            labels.append(torch.cat([b["labels"],    torch.full((pad,), -100)]))
        return {"input_ids": torch.stack(ids),
                "attention_mask": torch.stack(masks),
                "labels": torch.stack(labels)}

# ── Model ─────────────────────────────────────────────────────────────────────

def is_large(model_name):
    return any(x in model_name for x in ["70B", "72B"])

def load_model_for_training(model_name):
    path = os.path.join(MODELS_DIR, model_name)
    tok  = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    if is_large(model_name):
        model = AutoModelForCausalLM.from_pretrained(
            path, load_in_8bit=True, device_map="auto", trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    return tok, model

def load_finetuned(model_name, adapter_path):
    path = os.path.join(MODELS_DIR, model_name)
    tok  = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if is_large(model_name):
        base = AutoModelForCausalLM.from_pretrained(
            path, load_in_8bit=True, device_map="auto", trust_remote_code=True)
    else:
        base = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return tok, model

# ── Training ──────────────────────────────────────────────────────────────────

def train(model_name, model, tok, train_dataset, ckpt_dir, epochs, batch_size):
    last_ckpt = None
    if os.path.isdir(ckpt_dir):
        ckpts = sorted([d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")],
                       key=lambda x: int(x.split("-")[1]))
        if ckpts:
            last_ckpt = os.path.join(ckpt_dir, ckpts[-1])
            print(f"  Resuming from: {last_ckpt}")

    args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 8 // batch_size),
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=not is_large(model_name),
        bf16=is_large(model_name),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_dataset,
                      data_collator=PadCollator(tok))
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
        print(f"  Eval resume: {len(done)} done.")
        return samples, done
    except Exception:
        return [], set()

def save_eval_checkpoint(path, model_name, n_total, results):
    n        = len(results)
    rule_acc = sum(r["rule_correct"] for r in results) / n * 100 if n else 0
    s_acc    = sum(r["s_ok"]         for r in results) / n * 100 if n else 0
    s_mae    = float(np.mean([r["s_mae"] for r in results if r["s_mae"] is not None])) if n else 0
    with open(path, "w") as f:
        json.dump({
            "model":          model_name,
            "mode":           "finetuned",
            "n_evaluated":    n,
            "n_total":        n_total,
            "rule_accuracy":  rule_acc,
            "s_accuracy_pm1": s_acc,
            "s_mae":          s_mae,
            "parse_failures": sum(1 for r in results if r["pred_rule"] is None),
            "timestamp":      datetime.now().isoformat(),
            "samples":        results,
        }, f, indent=2)

# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(tok, model, orbit, max_new_tokens=32):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": make_user_prompt(orbit)}]
    try:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"System: {SYSTEM_PROMPT}\n\nUser: {make_user_prompt(orbit)}\n\nAssistant:"
    inputs = tok(text, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False, temperature=1.0,
                             pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# ── Run one model ─────────────────────────────────────────────────────────────

def run_model(model_name, n_train, n_test, epochs, batch_size):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_ckpt   = os.path.join(CKPT_DIR, model_name)
    os.makedirs(model_ckpt, exist_ok=True)
    adapter_path = os.path.join(model_ckpt, "lora_adapter")
    result_path  = os.path.join(RESULTS_DIR, f"{model_name}_finetuned.json")

    print(f"\n{'='*65}\n  {model_name}  |  LoRA fine-tune + eval (s-skewed)\n{'='*65}")

    # Step 1: Train
    if os.path.exists(adapter_path):
        print(f"  Adapter found. Skipping training.")
    else:
        print(f"  Loading training data ({n_train} samples) ...")
        tr_orbits, tr_rules, tr_s = load_train_data(n_train)
        tok, model = load_model_for_training(model_name)
        train_ds   = ECADataset(tr_orbits, tr_rules, tr_s, tok)
        t0 = time.time()
        adapter_path = train(model_name, model, tok, train_ds, model_ckpt, epochs, batch_size)
        print(f"  Training time: {(time.time()-t0)/3600:.2f}h")
        del model; torch.cuda.empty_cache()

    # Step 2: Evaluate
    print(f"  Loading test data ({n_test} samples) ...")
    te_orbits, te_rules, te_s = load_test_data(n_test)
    results, done_set = load_eval_checkpoint(result_path)
    remaining = [(i, te_orbits[i], te_rules[i], te_s[i])
                 for i in range(len(te_orbits)) if i not in done_set]

    if not remaining:
        print("  All test samples done.")
        return

    tok, model = load_finetuned(model_name, adapter_path)
    t0 = time.time()

    for pos, (i, orbit, true_rule, true_s) in enumerate(remaining):
        raw             = run_inference(tok, model, orbit)
        pred_rule, pred_s = parse_response(raw)
        r_ok  = pred_rule == int(true_rule) if pred_rule is not None else False
        s_ok  = abs(pred_s - int(true_s)) <= S_TOL if pred_s is not None else False
        s_mae = abs(pred_s - int(true_s)) if pred_s is not None else None

        results.append({
            "sample_idx":   i,
            "true_rule":    int(true_rule),
            "pred_rule":    pred_rule,
            "true_s":       int(true_s),
            "pred_s":       pred_s,
            "rule_correct": r_ok,
            "s_ok":         s_ok,
            "s_mae":        s_mae,
            "raw_output":   raw,
        })

        if (pos + 1) % 10 == 0 or pos + 1 == len(remaining):
            save_eval_checkpoint(result_path, model_name, n_test, results)
            done     = len(results)
            eta      = (time.time() - t0) / (pos + 1) * (len(remaining) - pos - 1)
            rule_acc = sum(r["rule_correct"] for r in results) / done * 100
            s_acc    = sum(r["s_ok"]         for r in results) / done * 100
            print(f"  [{done:>4}/{n_test}]  rule={rule_acc:.1f}%  "
                  f"s_acc(+-1)={s_acc:.1f}%  ETA={eta:.0f}s", flush=True)

    n        = len(results)
    rule_acc = sum(r["rule_correct"] for r in results) / n * 100
    s_acc    = sum(r["s_ok"]         for r in results) / n * 100
    print(f"\n  FINAL  rule={rule_acc:.2f}%  s_acc(+-1)={s_acc:.2f}%")
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
