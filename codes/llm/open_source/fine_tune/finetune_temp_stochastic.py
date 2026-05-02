"""
finetune_temp.py  --  LoRA fine-tuning + evaluation on temporally stochastic ECA.

Key differences from stochastic version:
  - Noise is tau (row-level: entire row uses one rule per timestep)
  - System prompt describes row-level coin flip explicitly
  - Training data generated on-the-fly from train_pairs.npy
  - Same data folder: TSCA_Data/

Usage:
  python finetune_temp.py --model Qwen2.5-7B-Instruct --n_train 2000 --n_test 200 --epochs 3
  python finetune_temp.py --model all --n_train 2000 --n_test 200 --epochs 3
"""

import os, json, time, argparse, re
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR    = "/home/user/project/ECA_TEMP"
MODELS_DIR  = "/home/user/models"
DATA_DIR    = os.path.join(BASE_DIR, "TSCA_Data")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "finetune")
CKPT_DIR    = os.path.join(BASE_DIR, "checkpoints", "finetune")
INDEX_FILE  = os.path.join(BASE_DIR, "results", "fixed_test_indices.npy")
SEED        = 42

TAU_TOL = 0.05
TAU_MIN = 0.1
TAU_MAX = 0.9
W, T, K = 50, 200, 8

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

SYSTEM_PROMPT = """You are an expert in cellular automata. You are given a space-time orbit of an Elementary Cellular Automaton (ECA) under temporally stochastic noise.

In this temporally stochastic model, at each timestep the ENTIRE row is updated by one of two rules. With probability tau the row uses rule G; otherwise the entire row uses rule F. This is a row-level coin flip — all cells in a row use the same rule at each timestep.

The parameter tau is a continuous value between 0.1 and 0.9. The system (F, G, tau) is statistically identical to (G, F, 1-tau), so predict the smaller rule number as rule_f and the larger as rule_g.

The orbit is a grid of 0s and 1s. Each row is one timestep. Grid width is 50 cells.

Identify:
1. Rule F (the first ECA rule, integer 0-255)
2. Rule G (the second ECA rule, integer 0-255, different from F)
3. Tau value (float between 0.1 and 0.9, rounded to 2 decimal places)

Respond ONLY in this JSON format with no extra text:
{"rule_f": <integer>, "rule_g": <integer>, "tau": <float>}"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_table(r):
    return np.array([(r >> i) & 1 for i in range(8)], dtype=np.uint8)

def simulate_tsca(rf, rg, init, tau, rng):
    """Generate one TSCA orbit: each row uses rule F or G (row-level flip)."""
    tf = build_table(rf); tg = build_table(rg)
    orbit = np.zeros((T, W), dtype=np.float32)
    s = init.copy().astype(np.uint8); orbit[0] = s
    for t in range(1, T):
        L = np.roll(s, 1); R = np.roll(s, -1)
        idx = (4 * L + 2 * s + R).astype(np.uint8)
        s   = (tg[idx] if rng.random() < tau else tf[idx]).astype(np.uint8)
        orbit[t] = s.astype(np.float32)
    return orbit

def random_init(rng):
    while True:
        x = rng.integers(0, 2, size=W, dtype=np.uint8)
        if 0 < int(x.sum()) < W:
            return x

def orbit_to_str(orbit):
    return "\n".join("".join(str(int(c)) for c in row) for row in orbit[:50])

def make_user_prompt(orbit):
    return (f"Space-time orbit (showing 50 of {T} rows x {W} cells):\n\n"
            f"{orbit_to_str(orbit)}\n\nWhat are rule F, rule G, and tau?")

def make_target(rf, rg, tau):
    if rf > rg:
        rf, rg = rg, rf
        tau    = 1.0 - tau
    return json.dumps({"rule_f": int(rf), "rule_g": int(rg),
                        "tau": round(float(tau), 2)})

def snap_tau(val):
    val = round(round(val / 0.05) * 0.05, 2)
    return max(TAU_MIN, min(TAU_MAX, val))

def parse_response(response):
    try:
        s = response.find("{"); e = response.rfind("}") + 1
        if s != -1 and e > s:
            d   = json.loads(response[s:e])
            rf  = int(d.get("rule_f", -1))
            rg  = int(d.get("rule_g", -1))
            tau = float(d.get("tau", -1))
            if 0 <= rf <= 255 and 0 <= rg <= 255 and TAU_MIN <= tau <= TAU_MAX:
                return rf, rg, snap_tau(tau)
    except Exception:
        pass
    rfm = re.search(r'"rule_f"\s*:\s*(\d+)', response)
    rgm = re.search(r'"rule_g"\s*:\s*(\d+)', response)
    tm  = re.search(r'"tau"\s*:\s*([\d.]+)', response)
    if rfm and rgm and tm:
        rf = int(rfm.group(1)); rg = int(rgm.group(1)); tau = float(tm.group(1))
        if 0 <= rf <= 255 and 0 <= rg <= 255:
            return rf, rg, snap_tau(tau)
    return None, None, None

def rules_match(pred_f, pred_g, true_f, true_g):
    return ((pred_f == true_f and pred_g == true_g) or
            (pred_f == true_g and pred_g == true_f))

# ── Data ──────────────────────────────────────────────────────────────────────

def get_fixed_test_indices(n_samples):
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    if os.path.exists(INDEX_FILE):
        return np.load(INDEX_FILE)[:n_samples]
    rng    = np.random.default_rng(SEED)
    orbits = np.load(os.path.join(DATA_DIR, "test", "orbits.npy"))
    n      = min(n_samples, len(orbits))
    idx    = rng.choice(len(orbits), size=n, replace=False)
    np.save(INDEX_FILE, idx)
    return idx

def load_test_data(n_test):
    idx      = get_fixed_test_indices(n_test)
    test_dir = os.path.join(DATA_DIR, "test")
    orbits   = np.load(os.path.join(test_dir, "orbits.npy"))
    rule_f   = np.load(os.path.join(test_dir, "rule_f_ids.npy"))
    rule_g   = np.load(os.path.join(test_dir, "rule_g_ids.npy"))
    taus     = np.load(os.path.join(test_dir, "taus.npy"))
    return orbits[idx], rule_f[idx], rule_g[idx], taus[idx]

def generate_train_data(n_train):
    """Generate training samples on-the-fly from train_pairs."""
    rng         = np.random.default_rng(SEED + 1)
    train_pairs = np.load(os.path.join(DATA_DIR, "train_pairs.npy"))
    samples     = []
    print(f"  Generating {n_train} training samples on-the-fly ...")
    for i in range(n_train):
        pi   = int(rng.integers(0, len(train_pairs)))
        rf   = int(train_pairs[pi, 0])
        rg   = int(train_pairs[pi, 1])
        tau  = float(rng.uniform(TAU_MIN, TAU_MAX))
        init = random_init(rng)
        orb  = simulate_tsca(rf, rg, init, tau, rng)
        samples.append((orb, rf, rg, tau))
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{n_train} generated ...", flush=True)
    print(f"  Done generating {n_train} training samples.")
    return samples

# ── Dataset ───────────────────────────────────────────────────────────────────

class ECADataset(Dataset):
    def __init__(self, samples, tok, max_length=4096):
        self.items      = []
        self.tok        = tok
        self.max_length = max_length
        for orbit, rf, rg, tau in samples:
            self.items.append([
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": make_user_prompt(orbit)},
                {"role": "assistant", "content": make_target(rf, rg, tau)},
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        messages = self.items[idx]
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

# FIX 1: Added `model_name` as first parameter so is_large() receives a string,
#         not a model object. The old signature was train(model, tok, ...) which
#         caused is_large(model) to always return False (no "70B"/"72B" in repr).
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
        # FIX 2: was is_large(model) — model object, not string.
        #         Now correctly uses model_name string for both flags.
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
    pair_acc = sum(r["pair_correct"] for r in results) / n * 100 if n else 0
    tau_acc  = sum(r["tau_ok"]       for r in results) / n * 100 if n else 0
    tau_mae  = float(np.mean([r["tau_mae"] for r in results
                               if r["tau_mae"] is not None])) if n else 0
    with open(path, "w") as f:
        json.dump({
            "model":               model_name,
            "mode":                "finetuned",
            "n_evaluated":         n,
            "n_total":             n_total,
            "pair_accuracy_UO":    pair_acc,
            "tau_accuracy_pm005":  tau_acc,
            "tau_mae_sym":         tau_mae,
            "parse_failures":      sum(1 for r in results if r["pred_f"] is None),
            "timestamp":           datetime.now().isoformat(),
            "samples":             results,
        }, f, indent=2)

# ── Inference ─────────────────────────────────────────────────────────────────

# FIX 3: Removed the erroneous `orbit[0]` indexing. The eval loop passes a
#         single orbit array (shape [T, W]) directly; indexing with [0] would
#         silently pass just the first row (one timestep) to make_user_prompt
#         instead of the full orbit, producing garbage prompts at eval time.
def run_inference(tok, model, orbit, max_new_tokens=48):
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

    print(f"\n{'='*65}\n  {model_name}  |  LoRA fine-tune + eval (temporal stochastic)\n{'='*65}")

    # Step 1: Train
    if os.path.exists(adapter_path):
        print(f"  Adapter found. Skipping training.")
    else:
        train_samples = generate_train_data(n_train)
        tok, model    = load_model_for_training(model_name)
        train_ds      = ECADataset(train_samples, tok)
        t0 = time.time()
        # FIX 1 call site: pass model_name as first argument
        adapter_path  = train(model_name, model, tok, train_ds, model_ckpt, epochs, batch_size)
        print(f"  Training time: {(time.time()-t0)/3600:.2f}h")
        del model; torch.cuda.empty_cache()

    # Step 2: Evaluate
    print(f"  Loading test data ({n_test} samples) ...")
    te_orbits, te_f, te_g, te_taus = load_test_data(n_test)
    results, done_set = load_eval_checkpoint(result_path)
    remaining = [(i, te_orbits[i], te_f[i], te_g[i], te_taus[i])
                 for i in range(len(te_orbits)) if i not in done_set]

    if not remaining:
        print("  All test samples done.")
        return

    tok, model = load_finetuned(model_name, adapter_path)
    t0 = time.time()

    for pos, (i, orbit, true_f, true_g, true_tau) in enumerate(remaining):
        # FIX 3 call site: pass orbit directly (shape [T, W]), not orbit[0]
        raw                      = run_inference(tok, model, orbit)
        pred_f, pred_g, pred_tau = parse_response(raw)
        pair_ok = rules_match(pred_f, pred_g, int(true_f), int(true_g)) \
                  if pred_f is not None else False
        if pred_tau is not None:
            tau_err = min(abs(pred_tau - float(true_tau)),
                          abs(pred_tau - (1.0 - float(true_tau))))
            tau_ok  = tau_err <= TAU_TOL
        else:
            tau_err = None; tau_ok = False

        results.append({
            "sample_idx":   i,
            "true_f":       int(true_f),
            "true_g":       int(true_g),
            "pred_f":       pred_f,
            "pred_g":       pred_g,
            "true_tau":     float(true_tau),
            "pred_tau":     pred_tau,
            "pair_correct": pair_ok,
            "tau_ok":       tau_ok,
            "tau_mae":      tau_err,
            "raw_output":   raw,
        })

        if (pos + 1) % 10 == 0 or pos + 1 == len(remaining):
            save_eval_checkpoint(result_path, model_name, n_test, results)
            done     = len(results)
            eta      = (time.time() - t0) / (pos + 1) * (len(remaining) - pos - 1)
            pair_acc = sum(r["pair_correct"] for r in results) / done * 100
            tau_acc  = sum(r["tau_ok"]       for r in results) / done * 100
            print(f"  [{done:>4}/{n_test}]  pair(UO)={pair_acc:.1f}%  "
                  f"tau(+-0.05)={tau_acc:.1f}%  ETA={eta:.0f}s", flush=True)

    n        = len(results)
    pair_acc = sum(r["pair_correct"] for r in results) / n * 100
    tau_acc  = sum(r["tau_ok"]       for r in results) / n * 100
    print(f"\n  FINAL  pair(UO)={pair_acc:.2f}%  tau(+-0.05)={tau_acc:.2f}%")
    print(f"  Saved: {result_path}")
    del model; torch.cuda.empty_cache()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="Qwen2.5-7B-Instruct")
    parser.add_argument("--n_train",    type=int, default=2000)
    parser.add_argument("--n_test",     type=int, default=200)
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    args   = parser.parse_args()
    models = ALL_MODELS if args.model == "all" else [args.model]
    for m in models:
        run_model(m, args.n_train, args.n_test, args.epochs, args.batch_size)

if __name__ == "__main__":
    main()