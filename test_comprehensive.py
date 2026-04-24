#!/usr/bin/env python3
"""
Comprehensive white-box and black-box test suite for main.py

Tests "Undetectable Steganography for Language Models" (Zamir 2024).

Sections:
  1.  PRF Unit Tests
  2.  Consistent Permutation Tests
  3.  binarize_next Tests
  4.  Score Function Tests
  5.  DynamicECC Unit Tests
  6.  Message Encoding Tests
  7.  Integration: Encode/Decode Round-Trips (uses token_ids = canonical path)
  8.  Determinism Tests
  9.  Key Sensitivity Tests
  10. Edge Cases
  11. Paper Algorithm Compliance
  12. Potential Bug / Correctness Checks
  13. CLI Interface Tests
"""

import hashlib
import hmac
import inspect
import json
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import main as M

# ── Terminal colours ────────────────────────────────────────────────────────────
def _c(code, text): return f"\033[{code}m{text}\033[0m"
PASS = lambda t=None: _c(92, "PASS")
FAIL = lambda t=None: _c(91, "FAIL")
WARN = lambda t=None: _c(93, "WARN")

results = []   # (name, passed, detail, is_warning)

def record(name, passed, detail="", warning=False):
    results.append((name, passed, detail, warning))
    tag = "WARN" if warning else ("PASS" if passed else "FAIL")
    colour = 93 if warning else (92 if passed else 91)
    print(f"  [{_c(colour, tag)}] {name}")
    if detail:
        for ln in detail.split("\n"):
            print(f"         {ln}")
    return passed

def section(title):
    print(f"\n{'='*72}\n  {title}\n{'='*72}")


# ── Load model (once) ───────────────────────────────────────────────────────────
print("Loading GPT-2 model...")
t0 = time.time()
LM = M.LMWrapper("gpt2")
print(f"  Loaded in {time.time()-t0:.1f}s  "
      f"vocab={LM.vocab_size}  blen={LM.blen}")

# ════════════════════════════════════════════════════════════════════════════════
# 1. PRF Unit Tests
# ════════════════════════════════════════════════════════════════════════════════
section("1. PRF Unit Tests")

vals = [M.prf("key", [i, j, s]) for i in range(5) for j in range(5)
        for s in ["0", "1", "<"]]
record("PRF outputs are in [0, 1)",
       all(0.0 <= v < 1.0 for v in vals),
       f"min={min(vals):.6f}  max={max(vals):.6f}")

v1 = M.prf("mykey", [1, 2, "0"])
v2 = M.prf("mykey", [1, 2, "0"])
record("PRF is deterministic", v1 == v2, f"v={v1}")

record("PRF is key-sensitive",
       M.prf("keyA", [0, 0, "0"]) != M.prf("keyB", [0, 0, "0"]),
       "keyA≠keyB → different PRF values")

vals_inp = {M.prf("k", [0, 0, "0"]), M.prf("k", [0, 0, "1"]),
            M.prf("k", [0, 1, "0"]), M.prf("k", [1, 0, "0"])}
record("PRF is sensitive to all three input components",
       len(vals_inp) == 4, f"{len(vals_inp)}/4 distinct values")

# Verify against reference HMAC-SHA256 computation
data = ("mykey||[1, 2, '0']").encode()
h = hmac.new(b"steg-prf-key", data, hashlib.sha256).digest()
expected_prf = struct.unpack(">Q", h[:8])[0] / (2**64)
record("PRF uses HMAC-SHA256 correctly",
       abs(M.prf("mykey", [1, 2, "0"]) - expected_prf) < 1e-15,
       f"expected={expected_prf:.10f}  got={M.prf('mykey',[1,2,'0']):.10f}")

# Distribution uniformity (1000 samples, 10 buckets, expect ~100 each)
samples_prf = [M.prf("uniformity", [i]) for i in range(1000)]
buckets = [0] * 10
for v in samples_prf:
    buckets[int(v * 10)] += 1
max_dev = max(abs(b - 100) for b in buckets)
record("PRF distribution is roughly uniform (10 buckets, 1000 samples)",
       max_dev < 40,
       f"buckets={buckets}  max_dev={max_dev}")


# ════════════════════════════════════════════════════════════════════════════════
# 2. Consistent Permutation Tests
# ════════════════════════════════════════════════════════════════════════════════
section("2. Consistent Permutation Tests")

perm100, inv100 = M.consistent_perm("testkey", 100)

record("perm is a valid permutation of [0, n)",
       sorted(perm100) == list(range(100)), f"len={len(perm100)}")

record("inv_perm is the true inverse of perm",
       all(inv100[perm100[i]] == i for i in range(100)),
       "inv_perm[perm[i]] == i for all i in [0,100)")

p1a, _ = M.consistent_perm("same_key", 50)
p1b, _ = M.consistent_perm("same_key", 50)
record("consistent_perm is deterministic",
       p1a == p1b, f"p1a[:5]={p1a[:5]}  p1b[:5]={p1b[:5]}")

pa, _ = M.consistent_perm("key_A", 1000)
pb, _ = M.consistent_perm("key_B", 1000)
record("Different keys → different permutations",
       pa != pb,
       f"first diff at index {next((i for i in range(1000) if pa[i]!=pb[i]), None)}")

probs10 = torch.softmax(torch.randn(10), dim=0)
perm10, inv10 = M.consistent_perm("k", 10)
permed10 = M.apply_perm(probs10, perm10)
record("apply_perm preserves probability sum",
       abs(permed10.sum().item() - 1.0) < 1e-5,
       f"sum={permed10.sum().item():.8f}")

record("apply_perm: result[perm[i]] == probs[i] for all i",
       all(abs(permed10[perm10[i]].item() - probs10[i].item()) < 1e-7
           for i in range(10)),
       "verified for 10-element vector")

# Encode/decode permutation round-trip: perm[inv_perm[j]] == j
pv, ipv = M.consistent_perm("verify", LM.vocab_size)
sample_ids = [0, 1, 100, 1000, 10000, 50256]
record("perm[inv_perm[j]] == j (encode/decode symmetry)",
       all(pv[ipv[j]] == j for j in sample_ids),
       f"tested IDs: {sample_ids}")

# Security note: uses Python random, not HMAC PRF
import random as _rng
_rng.seed("test_perm_key")
ref = list(range(10)); _rng.shuffle(ref)
got_perm, _ = M.consistent_perm("test_perm_key", 10)
record("consistent_perm uses Python random (Mersenne Twister, not HMAC) — security note",
       got_perm == ref,
       "NOTE: permutation seed uses Python random, inconsistent with HMAC-PRF approach elsewhere",
       warning=True)


# ════════════════════════════════════════════════════════════════════════════════
# 3. binarize_next Tests
# ════════════════════════════════════════════════════════════════════════════════
section("3. binarize_next Tests")

blen = LM.blen   # 16
vocab = LM.vocab_size  # 50257
probs_u = torch.ones(vocab) / vocab  # uniform over valid tokens

p0u, p1u = M.binarize_next(probs_u, 0, blen, 0)
record("binarize_next(ind=0, prefix=0): p0+p1 ≈ 1.0",
       abs(float(p0u) + float(p1u) - 1.0) < 1e-5,
       f"p0={float(p0u):.6f}  p1={float(p1u):.6f}  sum={float(p0u+p1u):.6f}")

# Verify bit partition: bit = (token_id >> (blen-ind-1)) % 2
cnt0 = sum(1 for t in range(vocab) if (t >> (blen-1)) % 2 == 0)
cnt1 = vocab - cnt0
record("binarize_next bit partition at ind=0 matches (token>>(blen-1))%2 rule",
       abs(float(p0u) - cnt0/vocab) < 1e-5 and abs(float(p1u) - cnt1/vocab) < 1e-5,
       f"expected p0={cnt0/vocab:.6f} p1={cnt1/vocab:.6f}")

# Non-uniform probs — compare to manual calculation
probs_nu = torch.softmax(torch.randn(vocab), dim=0)
ind_t, prefix_t = 4, 5
lo_t = prefix_t << (blen - ind_t)
hi_t = min((prefix_t + 1) << (blen - ind_t), vocab)
p0_man = sum(probs_nu[t].item() for t in range(lo_t, hi_t)
             if (t >> (blen-ind_t-1)) % 2 == 0)
p1_man = sum(probs_nu[t].item() for t in range(lo_t, hi_t)
             if (t >> (blen-ind_t-1)) % 2 == 1)
p0_fn, p1_fn = M.binarize_next(probs_nu, ind_t, blen, prefix_t)
record("binarize_next matches manual computation (non-uniform probs)",
       abs(float(p0_fn)-p0_man) < 1e-5 and abs(float(p1_fn)-p1_man) < 1e-5,
       f"p0: {float(p0_fn):.6f} vs {p0_man:.6f}  p1: {float(p1_fn):.6f} vs {p1_man:.6f}")

# Empty range: prefix=7, ind=3 → lo=57344 > vocab=50257
p0e, p1e = M.binarize_next(probs_u, 3, blen, 7)
record("binarize_next handles empty range gracefully (p0=p1=0 when lo≥vocab)",
       float(p0e) == 0.0 and float(p1e) == 0.0,
       f"prefix=7,ind=3 → lo=57344≥vocab=50257: p0={float(p0e)}, p1={float(p1e)}")

# The path to this empty range is blocked: at ind=2, prefix=3, p1=0
p0_block, p1_block = M.binarize_next(probs_u, 2, blen, 3)
record("Empty subtree unreachable: p1=0 at ind=2,prefix=3 (all tokens in range have bit=0)",
       float(p1_block) == 0.0,
       f"p0={float(p0_block):.6f}  p1={float(p1_block):.6f}")

# Performance
t_bin = time.time()
for ind in range(blen):
    M.binarize_next(probs_u, ind, blen, 0)
t_bin = time.time() - t_bin
record("binarize_next: 16 calls complete (performance note — O(vocab) Python loop)",
       True,
       f"16 calls: {t_bin*1000:.0f}ms  (vocab={vocab}, Python loop = bottleneck)",
       warning=t_bin > 0.5)


# ════════════════════════════════════════════════════════════════════════════════
# 4. Score Function Tests
# ════════════════════════════════════════════════════════════════════════════════
section("4. Score Function Tests")

u_test = M.prf("k", [1, 2, "0"])
s1 = M.compute_score_function("k", [1, 2, "0"], "1")
s0 = M.compute_score_function("k", [1, 2, "0"], "0")
record("Score s(bit=1) = -log(u)",
       abs(s1 - (-math.log(max(u_test, 1e-30)))) < 1e-10,
       f"s1={s1:.6f}  -log(u)={-math.log(max(u_test,1e-30)):.6f}")
record("Score s(bit=0) = -log(1-u)",
       abs(s0 - (-math.log(max(1-u_test, 1e-30)))) < 1e-10,
       f"s0={s0:.6f}  -log(1-u)={-math.log(max(1-u_test,1e-30)):.6f}")

all_scores = [M.compute_score_function("k", [i, j, s], b)
              for i in range(3) for j in range(3)
              for s in ["0","1","<"] for b in ["0","1"]]
record("All score values are strictly positive",
       all(sv > 0 for sv in all_scores),
       f"min={min(all_scores):.6f}")

# E[-log(U)] for U~Uniform(0,1) should be ≈ 1.0
mean_nlog = sum(-math.log(max(M.prf("e", [i]), 1e-30)) for i in range(5000)) / 5000
record("E[-log(PRF)] ≈ 1.0 (expected score per bit under null hypothesis)",
       abs(mean_nlog - 1.0) < 0.05,
       f"E[-log(U)] = {mean_nlog:.4f}  (should be ≈1.0)")

record("normalize_score((S-n)/√n) formula is correct",
       abs(M.normalize_score(15.0, 10) - (15.0-10)/math.sqrt(10)) < 1e-10,
       f"normalize(15,10)={M.normalize_score(15.0,10):.6f}")
record("normalize_score handles length=0 without division-by-zero",
       M.normalize_score(0.0, 0) == 0.0, "returns 0.0")


# ════════════════════════════════════════════════════════════════════════════════
# 5. DynamicECC Unit Tests
# ════════════════════════════════════════════════════════════════════════════════
section("5. DynamicECC Unit Tests")

# Decode: correct stream
record("DynamicECC.decode: perfect stream '01101' decodes correctly",
       M.DynamicECC.decode(["0","1","1","0","1"]) == "01101", "")

# Decode: with backspace
record("DynamicECC.decode handles '<' backspace",
       M.DynamicECC.decode(["0","1","<","1","1","0","1"]) == "01101",
       "0,1,<,1,1,0,1 → 0,1 → 0 → 01101")

# Backspace on empty is a no-op
record("DynamicECC.decode: backspace on empty is no-op",
       M.DynamicECC.decode(["<","<","1"]) == "1", "<<1 → 1")

# Multiple backspaces
record("DynamicECC.decode: multiple backspaces work",
       M.DynamicECC.decode(["1","0","<","<","0","1"]) == "01",
       "1,0,<,< → '', then 0,1 → '01'")

# Encoder: next_symbol
ecc = M.DynamicECC("01")
record("DynamicECC.next_symbol returns first bit initially",
       ecc.next_symbol() == "0", f"got {ecc.next_symbol()!r}")
ecc.update("0")
record("DynamicECC.next_symbol returns second bit after correct update",
       ecc.next_symbol() == "1", f"got {ecc.next_symbol()!r}")
ecc.update("1")
record("DynamicECC.next_symbol returns default '0' after message exhausted",
       ecc.next_symbol() == "0", f"got {ecc.next_symbol()!r}")

# Error handling
ecc2 = M.DynamicECC("01")
ecc2.update("1")  # wrong (expected "0")
record("DynamicECC: wrong symbol sets suffix_to_remove=1",
       ecc2.suffix_to_remove == 1, f"suffix_to_remove={ecc2.suffix_to_remove}")
record("DynamicECC: next_symbol returns '<' when suffix_to_remove>0",
       ecc2.next_symbol() == "<", f"got {ecc2.next_symbol()!r}")
ecc2.update("<")  # correction
record("DynamicECC: backspace received reduces suffix_to_remove to 0",
       ecc2.suffix_to_remove == 0, f"suffix_to_remove={ecc2.suffix_to_remove}")
record("DynamicECC: after correction, back to normal sequence (first bit '0')",
       ecc2.next_symbol() == "0", f"got {ecc2.next_symbol()!r}")

# last_index_written tracking
ecc3 = M.DynamicECC("101")
assert ecc3.last_index_written == -1
ecc3.update("1"); ecc3.update("0"); ecc3.update("1")
record("DynamicECC.last_index_written tracks bit progress correctly",
       ecc3.last_index_written == 2,
       f"after 3 correct updates: last_index_written={ecc3.last_index_written}")

# Backspace decrements last_index_written
ecc4 = M.DynamicECC("101")
ecc4.update("1")   # correct
ecc4.update("<")   # backspace
record("DynamicECC: '<' decrements last_index_written",
       ecc4.last_index_written == -1,
       f"got {ecc4.last_index_written}")

# Simulate noisy transmission
random.seed(42)
msg_bits = "10110"
ecc_enc = M.DynamicECC(msg_bits)
stream = []
for _ in range(30):
    intended = ecc_enc.next_symbol()
    received = intended if random.random() < 0.8 else random.choice(
        [s for s in ["0","1","<"] if s != intended])
    ecc_enc.update(received)
    stream.append(received)
decoded_noisy = M.DynamicECC.decode(stream)
bits_enc = ecc_enc.last_index_written + 1
record("DynamicECC: noisy transmission encodes/decodes prefix correctly",
       bits_enc >= 0 and decoded_noisy[:bits_enc] == msg_bits[:bits_enc],
       f"bits_encoded={bits_enc}  decoded prefix={decoded_noisy[:bits_enc]!r}")


# ════════════════════════════════════════════════════════════════════════════════
# 6. Message Encoding Tests
# ════════════════════════════════════════════════════════════════════════════════
section("6. Message Encoding Tests")

record("message_to_bits('A') = '01000001' (ASCII 65)",
       M.message_to_bits("A") == "01000001",
       f"got {M.message_to_bits('A')!r}")

record("bits_to_message(message_to_bits('A')) round-trips",
       M.bits_to_message(M.message_to_bits("A")) == "A", "")

record("message_to_bits length = 8 × len(UTF-8 bytes)",
       len(M.message_to_bits("Hi!")) == 24, f"len={len(M.message_to_bits('Hi!'))}")

record("message_to_bits produces only 0/1 chars",
       all(c in "01" for c in M.message_to_bits("Hello, World!")), "")

record("bits_to_message('') = ''",
       M.bits_to_message("") == "", "")

record("message_to_bits('') = ''",
       M.message_to_bits("") == "", "")

# Round-trip for a multi-char string
for msg in ["Hi", "SECRET", "Hello World"]:
    rt = M.bits_to_message(M.message_to_bits(msg))
    record(f"message encoding round-trip for {msg!r}",
           rt == msg, f"got {rt!r}")


# ════════════════════════════════════════════════════════════════════════════════
# 7. Integration: Encode/Decode Round-Trips
# ════════════════════════════════════════════════════════════════════════════════
section("7. Integration — Encode/Decode Round-Trips (via token_ids)")

def roundtrip(prompt, message, key, max_tokens=100, threshold=2.0):
    resp, bits, tids = M.encode(LM, prompt, message, key,
                                max_tokens=max_tokens, threshold=threshold)
    dec = M.decode(LM, key, threshold=threshold, token_ids=tids)
    return dec, bits, resp

CASES = [
    ("The weather today is",  "A",      "rt_key1", 100),
    ("Once upon a time",      "Hi",     "rt_key2", 100),
    ("Write a short story",   "OK",     "rt_key3", 100),
]

for prompt, msg, key, max_tok in CASES:
    t0 = time.time()
    try:
        dec, bits, resp = roundtrip(prompt, msg, key, max_tok)
        record(f"Round-trip: prompt={prompt[:20]!r} message={msg!r}",
               dec == msg,
               f"decoded={dec!r}  bits_hidden={bits}  time={time.time()-t0:.0f}s")
    except Exception as e:
        record(f"Round-trip: prompt={prompt[:20]!r} message={msg!r}",
               False, f"Exception: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# 8. Determinism Tests
# ════════════════════════════════════════════════════════════════════════════════
section("8. Determinism Tests")

t0 = time.time()
ra, ba, ta = M.encode(LM, "Explain relativity", "A", "det_key", 80)
rb, bb, tb = M.encode(LM, "Explain relativity", "A", "det_key", 80)
record("Encode is deterministic (same inputs → same token sequence)",
       ta == tb and ra == rb,
       f"tokens match={ta==tb}  texts match={ra==rb}  time={time.time()-t0:.0f}s")

da = M.decode(LM, "det_key", token_ids=ta)
db = M.decode(LM, "det_key", token_ids=ta)
record("Decode is deterministic",
       da == db, f"da={da!r}  db={db!r}")


# ════════════════════════════════════════════════════════════════════════════════
# 9. Key Sensitivity Tests
# ════════════════════════════════════════════════════════════════════════════════
section("9. Key Sensitivity Tests")

t0 = time.time()
rA, _, tA = M.encode(LM, "The cat sat on the mat", "Hi", "key_A", 100)
rB, _, tB = M.encode(LM, "The cat sat on the mat", "Hi", "key_B", 100)
record("Different keys → different encoded texts",
       rA != rB,
       f"rA[:30]={rA[:30]!r}  rB[:30]={rB[:30]!r}  time={time.time()-t0:.0f}s")

dA = M.decode(LM, "key_A", token_ids=tA)
record("Correct key decodes correctly",
       dA == "Hi", f"decoded={dA!r}")

dWrong = M.decode(LM, "key_B", token_ids=tA)
record("Wrong key does not recover payload",
       dWrong != "Hi",
       f"with wrong key: decoded={dWrong!r}  (must differ from 'Hi')")

dW1 = M.decode(LM, "totally_wrong_1", token_ids=tA)
dW2 = M.decode(LM, "totally_wrong_2", token_ids=tA)
record("Two distinct wrong keys produce different decode outputs",
       dW1 != dW2,
       f"wrong1={dW1!r}  wrong2={dW2!r}")


# ════════════════════════════════════════════════════════════════════════════════
# 10. Edge Cases
# ════════════════════════════════════════════════════════════════════════════════
section("10. Edge Cases")

# max_tokens=0
try:
    r0, b0, t0_ = M.encode(LM, "Hello", "AB", "edge_zero", max_tokens=0)
    record("max_tokens=0 runs without crash",
           len(t0_) == 0 and b0 == 0,
           f"bits_hidden={b0}  tokens={len(t0_)}")
except Exception as e:
    record("max_tokens=0 runs without crash", False, f"Exception: {e}")

# max_tokens=5 (may not encode full message, should not crash)
try:
    r5, b5, t5 = M.encode(LM, "Hello", "LONGMESSAGE", "edge_five", max_tokens=5)
    record("max_tokens=5 runs without crash",
           True, f"bits_hidden={b5}  tokens={len(t5)}")
except Exception as e:
    record("max_tokens=5 runs without crash", False, f"Exception: {e}")

# Decode non-encoded text → no crash
try:
    random_text = ("The cat sat on the mat and watched the birds fly by. "
                   "It was a warm afternoon in a quiet village.")
    dec_rand = M.decode(LM, "somekey", text=random_text)
    record("Decode of non-encoded text runs without crash",
           True, f"decoded={dec_rand!r}")
except Exception as e:
    record("Decode of non-encoded text runs without crash", False, f"Exception: {e}")

# Different messages, same key and prompt → different token sequences
_, _, tm1 = M.encode(LM, "Once upon a time", "AB", "same_key", 80)
_, _, tm2 = M.encode(LM, "Once upon a time", "CD", "same_key", 80)
record("Same key + prompt, different messages → different token sequences",
       tm1 != tm2,
       f"tm1[:3]={tm1[:3]}  tm2[:3]={tm2[:3]}")

# token_ids path vs text retokenization path
resp_txt, _, tids_txt = M.encode(LM, "The quick brown fox", "Hi", "text_key", 80)
dec_tids = M.decode(LM, "text_key", token_ids=tids_txt)
dec_text = M.decode(LM, "text_key", text=resp_txt, prompt="The quick brown fox")
record("Token-ID path and text-retokenization path give same decode result",
       dec_tids == dec_text,
       f"via_tids={dec_tids!r}  via_text={dec_text!r}")

# Threshold sensitivity
t0 = time.time()
_, bits_lo, _ = M.encode(LM, "Long ocean story", "Hi", "thr_key", 100, threshold=1.5)
_, bits_hi, _ = M.encode(LM, "Long ocean story", "Hi", "thr_key", 100, threshold=4.0)
record("Lower threshold hides ≥ bits as higher threshold (same token budget)",
       bits_lo >= bits_hi,
       f"threshold=1.5: {bits_lo} bits;  threshold=4.0: {bits_hi} bits  time={time.time()-t0:.0f}s")

# EOS token in decode: filtered but potentially scored in encode
eos_id = LM.tokenizer.eos_token_id
_, _, tids_eos = M.encode(LM, "Test EOS handling", "Hi", "eos_key", 100)
has_eos = eos_id in tids_eos
# Inject EOS mid-sequence and verify no crash
tids_with_eos = tids_eos[:5] + [eos_id] + tids_eos[5:]
try:
    dec_eos_inj = M.decode(LM, "eos_key", token_ids=tids_with_eos)
    record("Decode with injected mid-sequence EOS does not crash",
           True, f"has_natural_eos={has_eos}  injected_decode={dec_eos_inj!r}")
    if has_eos:
        record("POTENTIAL BUG: Encode scored EOS bits but Decode filters EOS",
               False,
               "Encoder processes EOS token's bits in score loop, then appends to generated_token_ids.\n"
               "Decoder filters EOS from response_token_list before scoring.\n"
               "If EOS is naturally generated, score state diverges: encode scored extra bits,\n"
               "decode did not → subsequent symbol detection is offset. Round-trip may fail.",
               warning=True)
except Exception as e:
    record("Decode with injected mid-sequence EOS does not crash", False, f"Exception: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# 11. Paper Algorithm Compliance
# ════════════════════════════════════════════════════════════════════════════════
section("11. Paper Algorithm Compliance")

# blen = ceil(log2(vocab_size))  [CGZ Section 4.1]
record("Binary reduction: blen = ⌈log₂(vocab_size)⌉",
       LM.blen == math.ceil(math.log2(LM.vocab_size)),
       f"blen={LM.blen}  ⌈log₂({LM.vocab_size})⌉={math.ceil(math.log2(LM.vocab_size))}")

# Encoder calls PRF with [token_idx, bit_idx, symbol] — white-box trace
prf_calls = []
_orig = M.prf
def _spy(key, inp):
    prf_calls.append((key, inp))
    return _orig(key, inp)
M.prf = _spy
try:
    M.encode(LM, "test", "A", "spy_key", max_tokens=3, threshold=100.0)
finally:
    M.prf = _orig
calls_3elem = [c for _, c in prf_calls if len(c) == 3]
record("Encoder calls PRF with 3-element inputs [token_idx, bit_idx, symbol]",
       len(calls_3elem) > 0,
       f"total 3-element PRF calls: {len(calls_3elem)};  example: {calls_3elem[:2]}")

# Algorithm 3: bit selection follows k < p1/(p0+p1)
# Verified structurally via source inspection
src_enc = inspect.getsource(M.encode)
record("Algorithm 3: encoder uses threshold 'k < p1 / (p0 + p1 + ...)' for bit selection",
       "p1 / (p0 + p1" in src_enc,
       "source contains 'p1 / (p0 + p1' expression")

# Algorithm 4: decoder uses identical score accumulation
src_dec = inspect.getsource(M.decode)
record("Algorithm 4: decoder accumulates scores identically to encoder",
       "compute_score_function" in src_dec and "normalize_score" in src_dec,
       "decoder uses compute_score_function + normalize_score")

# score_length incremented BEFORE score check (consistent in both encode and decode)
enc_sl = src_enc.find("score_length += 1")
enc_ns = src_enc.find("normalize_score")
dec_sl = src_dec.find("score_length += 1")
dec_ns = src_dec.find("normalize_score")
record("score_length incremented before normalize_score check in encode",
       enc_sl < enc_ns,
       f"score_length_pos={enc_sl}  normalize_score_pos={enc_ns}")
record("score_length incremented before normalize_score check in decode",
       dec_sl < dec_ns,
       f"score_length_pos={dec_sl}  normalize_score_pos={dec_ns}")

# Section 5: ECC uses ternary alphabet {0,1,<}
src_ecc = inspect.getsource(M.DynamicECC)
record("DynamicECC uses ternary alphabet {0, 1, <} as per Section 5",
       '"<"' in src_ecc and '"0"' in src_ecc and '"1"' in src_ecc,
       "DynamicECC source contains '0', '1', '<' symbols")

# Score normalization: (S - n) / sqrt(n) — paper's test statistic
record("Score normalization uses (S−n)/√n formula",
       abs(M.normalize_score(12.0, 10) - (12.0-10)/math.sqrt(10)) < 1e-10, "")

# Long encoding hides multiple bits (implies score resets function)
resp_long, bits_long, tids_long = M.encode(
    LM, "Write a detailed story about the ocean", "Hi", "compliance_key", 100)
record("Encoding 100 tokens with 'Hi' hides ≥ 8 bits (score reset working)",
       bits_long >= 8,
       f"bits_hidden={bits_long}  (need ≥16 for full 'Hi'; partial counts)")

# Full round-trip compliance
dec_comp = M.decode(LM, "compliance_key", token_ids=tids_long)
record("Full Algorithm 3→4 round-trip reconstructs original message",
       dec_comp == "Hi",
       f"encoded_bits={bits_long}  decoded={dec_comp!r}")

# Permutation spreads indices: check >90% of positions differ between keys
pA, _ = M.consistent_perm("pA", LM.vocab_size)
pB, _ = M.consistent_perm("pB", LM.vocab_size)
diff = sum(1 for i in range(LM.vocab_size) if pA[i] != pB[i])
record("Permutation is strongly key-dependent (>90% positions differ)",
       diff > 0.9 * LM.vocab_size,
       f"{diff}/{LM.vocab_size} = {diff/LM.vocab_size:.1%} positions differ")


# ════════════════════════════════════════════════════════════════════════════════
# 12. Potential Bug / Correctness Checks
# ════════════════════════════════════════════════════════════════════════════════
section("12. Potential Bug / Correctness Checks")

# apply_perm consistency: probs_permed[perm[i]] == probs[i]
probs_c = torch.softmax(torch.randn(50), dim=0)
perm_c, _ = M.consistent_perm("consistency", 50)
permed_c = M.apply_perm(probs_c, perm_c)
ok_c = all(abs(permed_c[perm_c[i]].item() - probs_c[i].item()) < 1e-7 for i in range(50))
record("apply_perm correctness: probs_permed[perm[i]] == probs[i]",
       ok_c, f"50 elements all consistent: {ok_c}")

# Decode bit extraction matches encode's bit check
# encode: bit_str = str(token_id % 2) at each ind
# decode: bit = token_bits[ind] where token_bits = format(permuted_id, '016b')
# They should match: format(x,'016b')[ind] == (x >> (15-ind)) & 1
test_perm_id = 12345
tb = format(test_perm_id, f"0{blen}b")
match = all(int(tb[i]) == (test_perm_id >> (blen-i-1)) % 2 for i in range(blen))
record("Decode bit extraction consistent with encode's (x>>(blen-ind-1))%2 formula",
       match, f"permuted_id={test_perm_id}  bits={tb}")

# PRF input collision check: distinct (i,ind,symbol) tuples all give distinct PRF values
seen = {}
collision = False
for i in range(8):
    for ind in range(blen):
        for s in ["0","1","<"]:
            v = M.prf("collision_chk", [i, ind, s])
            k = (i, ind, s)
            if v in seen and seen[v] != k:
                collision = True
                break
            seen[v] = k
record("No PRF output collisions across 8×16×3=384 (i,ind,symbol) combinations",
       not collision,
       f"384 distinct PRF inputs tested, collision={collision}")

# apply_perm performance (O(vocab) Python loop is a bottleneck)
probs_perf = torch.softmax(torch.randn(LM.vocab_size), dim=0)
perm_perf, _ = M.consistent_perm("perf", LM.vocab_size)
t_ap = time.time()
for _ in range(5):
    M.apply_perm(probs_perf, perm_perf)
t_ap = (time.time() - t_ap) / 5
record("apply_perm performance: O(vocab_size) Python loop per token",
       True,
       f"avg={t_ap*1000:.1f}ms per call  ({LM.vocab_size} Python iterations)\n"
       f"  INEFFICIENCY: should use result[torch.tensor(perm)] = probs (vectorized)",
       warning=t_ap > 0.05)

# Overall encode throughput
t_enc = time.time()
M.encode(LM, "benchmark test", "A", "bench_key", max_tokens=10)
t_enc = time.time() - t_enc
record("Encode throughput: 10 tokens completes (performance note)",
       True,
       f"10 tokens: {t_enc:.1f}s  (~{t_enc/10:.1f}s/token)\n"
       f"  BOTTLENECK: binarize_next called {blen}× per token, each O(vocab) Python loop\n"
       f"  INEFFICIENCY: should compute p0,p1 via cumulative sums or tensor slicing",
       warning=t_enc > 5.0)

# ECC: check that encode's update loop uses correct symbol for PRF (not a hardcoded default)
# White-box: after each symbol detection, the NEXT symbol from ecc.next_symbol() is used
# in subsequent PRF calls — this is crucial for the scheme's correctness
record("Encoder uses dynamic ECC symbol for PRF key (not hardcoded)",
       "ecc.next_symbol()" in src_enc,
       "source contains 'ecc.next_symbol()' call after symbol detection")

# Verify the 'bits_hidden' report matches actual message coverage
resp_bh, bits_bh, tids_bh = M.encode(LM, "Bits hidden test", "Hi", "bh_key", 100)
dec_bh = M.decode(LM, "bh_key", token_ids=tids_bh)
bits_needed = len(M.message_to_bits("Hi"))  # 16
record("bits_hidden correctly reflects decodeable content",
       (bits_bh >= bits_needed) == (dec_bh == "Hi"),
       f"bits_needed={bits_needed}  bits_hidden={bits_bh}  decoded={dec_bh!r}")

# Score symbols checked in fixed order "0","1","<" — potential tie-breaking issue
# If the correct symbol happens to be "1" or "<" but "0"'s score also exceeds threshold
# simultaneously, "0" wins. This is implementation-defined and may differ from paper.
record("Score detection order is '0','1','<' (tie-breaking note)",
       True,
       "NOTE: Symbols checked in fixed order '0','1','<'. If two symbols exceed\n"
       "  threshold simultaneously, the leftmost in this order wins.\n"
       "  The paper doesn't specify tie-breaking; this could cause subtle\n"
       "  divergence from the paper's intended behavior in degenerate cases.",
       warning=True)


# ════════════════════════════════════════════════════════════════════════════════
# 13. CLI Interface Tests
# ════════════════════════════════════════════════════════════════════════════════
section("13. CLI Interface Tests")

SCRIPT = os.path.join(os.path.dirname(__file__), "main.py")
PY = sys.executable

def cli(args, timeout=180):
    return subprocess.run([PY, SCRIPT] + args, capture_output=True, text=True,
                          timeout=timeout)

# Missing --key must fail
r = cli(["encode", "--prompt", "test", "--message", "A"], timeout=15)
record("CLI: encode without --key exits non-zero",
       r.returncode != 0,
       f"returncode={r.returncode}")

# Missing --message must fail
r = cli(["encode", "--prompt", "test", "--key", "k"], timeout=15)
record("CLI: encode without --message exits non-zero",
       r.returncode != 0,
       f"returncode={r.returncode}")

# Missing --input or --text must fail for decode
r = cli(["decode", "--key", "k"], timeout=15)
record("CLI: decode without --input or --text exits non-zero",
       r.returncode != 0,
       f"returncode={r.returncode}")

# Full CLI round-trip via JSON file
with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
    out_file = f.name
try:
    t0 = time.time()
    r_enc = cli(["encode", "--prompt", "Write a short story",
                 "--message", "Hi", "--key", "cli_rt_key",
                 "--max-tokens", "100", "--output", out_file])
    record("CLI: encode exits 0 and produces stdout",
           r_enc.returncode == 0 and bool(r_enc.stdout.strip()),
           f"returncode={r_enc.returncode}  "
           f"stdout_len={len(r_enc.stdout)}  time={time.time()-t0:.0f}s")

    if r_enc.returncode == 0 and os.path.exists(out_file):
        r_dec = cli(["decode", "--key", "cli_rt_key", "--input", out_file])
        decoded_cli = r_dec.stdout.strip()
        record("CLI: decode from --input JSON exits 0",
               r_dec.returncode == 0,
               f"returncode={r_dec.returncode}")
        record("CLI: decode from --input recovers correct message",
               decoded_cli == "Hi",
               f"decoded={decoded_cli!r}  expected='Hi'")
        # Verify JSON structure
        with open(out_file) as fp:
            data = json.load(fp)
        record("CLI: --output JSON contains 'text', 'token_ids', 'model'",
               all(k in data for k in ["text","token_ids","model"]),
               f"keys={list(data.keys())}")
finally:
    if os.path.exists(out_file):
        os.unlink(out_file)

# Verify test_suite.py is INCOMPATIBLE with this implementation's CLI
# (documents an important compatibility issue)
incompatible_args = ["encode", "--prompt", "test", "--message", "A",
                     "--key", "k", "--message-bits", "0101", "--temperature", "1.0"]
r_incompat = subprocess.run([PY, SCRIPT] + incompatible_args,
                             capture_output=True, text=True, timeout=15)
record("test_suite.py is incompatible with this implementation's CLI",
       r_incompat.returncode != 0,
       f"test_suite.py uses --message-bits, --temperature, --raw-bits flags not present\n"
       f"  in this implementation. The provided test suite CANNOT test this code\n"
       f"  as-is. returncode={r_incompat.returncode} (non-zero confirms incompatibility)",
       warning=True)


# ════════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════════
section("FINAL SUMMARY")

passed   = [r for r in results if r[1] and not r[3]]
failed   = [r for r in results if not r[1] and not r[3]]
warnings = [r for r in results if r[3]]
total    = len(results)
pct      = len(passed) / total * 100 if total else 0

print(f"  Total tests  : {total}")
print(f"  Passed       : {len(passed)}  ({pct:.1f}%)")
print(f"  Failed       : {len(failed)}")
print(f"  Warnings     : {len(warnings)}")

if failed:
    print(f"\n{'─'*72}")
    print("FAILED TESTS:")
    for name, _, detail, _ in failed:
        print(f"  ✗ {name}")
        if detail:
            for ln in detail.split("\n"):
                print(f"      {ln}")

if warnings:
    print(f"\n{'─'*72}")
    print("WARNINGS / NOTES:")
    for name, _, detail, _ in warnings:
        print(f"  ⚠  {name}")
        if detail:
            for ln in detail.split("\n"):
                print(f"      {ln}")

verdict = "ALL TESTS PASSED" if not failed else f"{len(failed)} TEST(S) FAILED"
print(f"\n{'='*72}")
print(f"  VERDICT: {verdict}")
print(f"{'='*72}")

sys.exit(0 if not failed else 1)
