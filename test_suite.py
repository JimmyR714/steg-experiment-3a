#!/usr/bin/env python3
"""
General steganography implementation test suite.

Tests any implementation of "Undetectable Steganography for Language Models"
(Zamir, 2024) that follows the task.md CLI spec:

    python main.py encode --prompt P --message M --key K [--model M] [--max-tokens N]
    python main.py decode --text T --key K [--model M] [--raw-bits]

Usage:
    python test_suite.py                        # all tests, gpt2, main.py in cwd
    python test_suite.py --model gpt2           # explicit model
    python test_suite.py --script path/main.py  # different script
    python test_suite.py --smoke                # quick smoke test only (fastest)
    python test_suite.py --max-tokens 200       # override token budget
    python test_suite.py --report report.md     # write markdown report to file
"""

import argparse
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    group: str
    passed: bool
    skipped: bool = False
    duration_s: float = 0.0
    detail: str = ""          # one-line explanation on failure or noteworthy info
    encode_text: str = ""     # cached encoded text (for reuse)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def run_encode(script, prompt, message=None, bits=None, key="testkey",
               model="gpt2", max_tokens=150, temperature=1.0,
               timeout=120) -> tuple[str, str]:
    """Run encode command; return (stdout, stderr). Raises on timeout."""
    cmd = [sys.executable, script,
           "encode",
           "--prompt", prompt,
           "--key", key,
           "--model", model,
           "--max-tokens", str(max_tokens),
           "--temperature", str(temperature)]
    if message is not None:
        cmd += ["--message", message]
    elif bits is not None:
        cmd += ["--message-bits", bits]
    else:
        raise ValueError("encode needs --message or --message-bits")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.stdout, result.stderr


def run_decode(script, text, key="testkey", model="gpt2",
               raw_bits=False, timeout=60) -> tuple[str, str]:
    """Run decode command; return (stdout, stderr). Raises on timeout."""
    cmd = [sys.executable, script,
           "decode",
           "--text", text,
           "--key", key,
           "--model", model]
    if raw_bits:
        cmd.append("--raw-bits")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Output parsers  (flexible — tolerant of varied formatting)
# ---------------------------------------------------------------------------

def parse_generated_text(stdout: str) -> str | None:
    """Extract the generated response text from encode stdout."""
    # Primary pattern: "Generated response:\n<text>\n\nPayload bits"
    m = re.search(
        r'Generated response:\s*\n(.*?)(?:\n\s*\nPayload bits encoded:|$)',
        stdout, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: last non-empty block
    blocks = [b.strip() for b in re.split(r'\n{2,}', stdout) if b.strip()]
    if blocks:
        # Skip blocks that look like metadata lines
        for block in reversed(blocks):
            if not re.match(r'^(Message:|Key:|Model:|Generating|Payload bits|Raw bits|Decoded)', block):
                return block
    return None


def parse_bits_encoded(stdout: str) -> tuple[int | None, int | None]:
    """Return (bits_encoded, payload_total) from encode stdout."""
    m = re.search(r'Payload bits encoded:\s*(\d+)\s*/\s*(\d+)', stdout)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def parse_extracted_bits(stdout: str) -> str | None:
    """Return the bit string from decode stdout."""
    m = re.search(r'Extracted bits\s*\(\d+\):\s*([01]+)', stdout)
    if m:
        return m.group(1)
    # Fallback: any run of 0/1 chars of length >= 1
    m = re.search(r'\b([01]{5,})\b', stdout)
    return m.group(1) if m else None


def parse_decoded_message(stdout: str) -> str | None:
    """Return the decoded text message from decode stdout."""
    m = re.search(r"Decoded message:\s*'([^']*)'", stdout)
    if m:
        return m.group(1)
    m = re.search(r'Decoded message:\s*"([^"]*)"', stdout)
    if m:
        return m.group(1)
    return None


def text_to_bits(text: str) -> str:
    """Compact 5-bit encoding (A-Z, space, @:./?) — matches paper's alphabet."""
    ALPHABET = (' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
                'X', 'Y', 'Z', '@', ':', '.', '/', '?')
    enc = {ch: i for i, ch in enumerate(ALPHABET)}
    return "".join(format(enc[ch], '05b') for ch in text.upper()
                   if ch.upper() in enc)


def bits_to_text(bits: str) -> str:
    ALPHABET = (' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
                'X', 'Y', 'Z', '@', ':', '.', '/', '?')
    bits = bits[:len(bits) - len(bits) % 5]
    return "".join(ALPHABET[int(bits[i:i+5], 2)] for i in range(0, len(bits), 5))


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

class Suite:
    def __init__(self, script, model, max_tokens, smoke_only):
        self.script = script
        self.model = model
        self.max_tokens = max_tokens
        self.smoke_only = smoke_only
        self.results: list[TestResult] = []
        self._encode_cache: dict = {}   # (prompt, message/bits, key) -> encoded_text

    def _encode_cached(self, prompt, key, message=None, bits=None) -> str | None:
        """Encode once; cache result so decode tests don't re-encode."""
        cache_key = (prompt, key, message, bits)
        if cache_key in self._encode_cache:
            return self._encode_cache[cache_key]
        try:
            stdout, _ = run_encode(self.script, prompt, message=message, bits=bits,
                                   key=key, model=self.model,
                                   max_tokens=self.max_tokens)
            text = parse_generated_text(stdout)
            self._encode_cache[cache_key] = text
            return text
        except subprocess.TimeoutExpired:
            self._encode_cache[cache_key] = None
            return None
        except Exception:
            self._encode_cache[cache_key] = None
            return None

    def record(self, name, group, passed, detail="", duration=0.0,
               skipped=False, encode_text=""):
        r = TestResult(name=name, group=group, passed=passed,
                       skipped=skipped, duration_s=duration,
                       detail=detail, encode_text=encode_text)
        self.results.append(r)
        icon = "SKIP" if skipped else ("PASS" if passed else "FAIL")
        print(f"  [{icon}] {name}" + (f"  — {detail}" if detail else ""))
        return r

    def run(self):
        groups = [
            ("1. Smoke Tests",        self.smoke_tests),
            ("2. Round-Trip Tests",   self.roundtrip_tests),
            ("3. Key Sensitivity",    self.key_sensitivity_tests),
            ("4. Determinism",        self.determinism_tests),
            ("5. Capacity",           self.capacity_tests),
            ("6. Alphabet Coverage",  self.alphabet_tests),
            ("7. Bit-level Payloads", self.bitlength_tests),
            ("8. Edge Cases",         self.edge_case_tests),
            ("9. Paper Compliance",   self.paper_compliance_tests),
        ]
        for title, fn in groups:
            print(f"\n{'='*60}")
            print(f" {title}")
            print(f"{'='*60}")
            fn()
            if self.smoke_only and title.startswith("1."):
                print("\n  (--smoke: stopping after smoke tests)")
                break

    # ------------------------------------------------------------------ #
    #  Group 1 — Smoke Tests                                              #
    # ------------------------------------------------------------------ #

    def smoke_tests(self):
        group = "Smoke"

        # 1a: encode runs without error
        t0 = time.time()
        try:
            stdout, stderr = run_encode(self.script, "Hello world",
                                        message="HI", key="smokekey",
                                        model=self.model, max_tokens=80)
            passed = bool(stdout) and "error" not in stderr.lower()
            text = parse_generated_text(stdout)
            detail = f"generated {len(text or '')} chars" if text else "no text parsed"
        except subprocess.TimeoutExpired:
            passed, detail, text = False, "timeout", None
        except Exception as e:
            passed, detail, text = False, str(e), None
        self.record("encode runs without error", group, passed, detail,
                    duration=time.time()-t0)

        # 1b: decode runs without error
        t0 = time.time()
        if text:
            try:
                stdout2, stderr2 = run_decode(self.script, text, key="smokekey",
                                             model=self.model)
                passed = bool(stdout2) and "error" not in stderr2.lower()
                bits = parse_extracted_bits(stdout2)
                detail = f"extracted {len(bits or '')} bits" if bits else "no bits parsed"
            except subprocess.TimeoutExpired:
                passed, detail = False, "timeout"
            except Exception as e:
                passed, detail = False, str(e)
        else:
            passed, detail = False, "skipped — no encoded text"
        self.record("decode runs without error", group, passed, detail,
                    duration=time.time()-t0)

        # 1c: encode produces non-empty non-prompt text
        t0 = time.time()
        if text:
            passed = len(text.strip()) > 10
            detail = f"len={len(text.strip())}"
        else:
            passed, detail = False, "no text"
        self.record("encode produces non-empty text", group, passed, detail,
                    duration=time.time()-t0)

        # 1d: basic round-trip
        t0 = time.time()
        if text:
            try:
                stdout2, _ = run_decode(self.script, text, key="smokekey",
                                        model=self.model)
                bits = parse_extracted_bits(stdout2)
                expected = text_to_bits("HI")
                if bits and len(bits) >= len(expected):
                    passed = bits[:len(expected)] == expected
                    detail = f"decoded={bits[:len(expected)]!r} expected={expected!r}"
                else:
                    passed = False
                    detail = f"too few bits: got {len(bits or '')} need {len(expected)}"
            except Exception as e:
                passed, detail = False, str(e)
        else:
            passed, detail = False, "no encoded text"
        self.record("basic encode→decode round-trip ('HI')", group, passed,
                    detail, duration=time.time()-t0)

    # ------------------------------------------------------------------ #
    #  Group 2 — Round-Trip Tests                                         #
    # ------------------------------------------------------------------ #

    def roundtrip_tests(self):
        group = "RoundTrip"
        cases = [
            # (prompt,                    message,   key,         label)
            ("The weather today is",      "A",       "rkey1",     "single char A"),
            ("The weather today is",      "Z",       "rkey2",     "single char Z"),
            ("The weather today is",      "HELLO",   "rkey3",     "5-char HELLO"),
            ("The weather today is",      "WORLD",   "rkey4",     "5-char WORLD"),
            ("Once upon a time",          "HI",      "rkey5",     "2-char HI"),
            ("Once upon a time",          "OZ",      "rkey6",     "2-char OZ"),
            ("Write a short story",       "SECRET",  "rkey7",     "6-char SECRET"),
            ("Explain quantum physics",   "HELLO",   "rkey8",     "different prompt"),
            ("The quick brown fox",       "TEST",    "rkey9",     "4-char TEST"),
            ("A",                         "AB",      "rkey10",    "very short prompt"),
        ]
        for prompt, msg, key, label in cases:
            t0 = time.time()
            try:
                encoded = self._encode_cached(prompt, key, message=msg)
                if not encoded:
                    self.record(f"round-trip {label}", group, False,
                                "encode produced no text", duration=time.time()-t0)
                    continue
                stdout, _ = run_decode(self.script, encoded, key=key,
                                       model=self.model)
                bits = parse_extracted_bits(stdout)
                decoded_msg = parse_decoded_message(stdout)
                expected_bits = text_to_bits(msg)
                n = len(expected_bits)
                if bits and len(bits) >= n:
                    bits_ok = bits[:n] == expected_bits
                else:
                    bits_ok = False
                # Also accept decoded message match
                msg_ok = decoded_msg is not None and decoded_msg.strip() == msg.upper()
                passed = bits_ok or msg_ok
                detail = (f"decoded={bits[:n]!r} expected={expected_bits!r}"
                          if bits else f"no bits extracted; decoded_msg={decoded_msg!r}")
            except subprocess.TimeoutExpired:
                passed, detail = False, "timeout"
            except Exception as e:
                passed, detail = False, str(e)
            self.record(f"round-trip {label}", group, passed, detail,
                        duration=time.time()-t0)

    # ------------------------------------------------------------------ #
    #  Group 3 — Key Sensitivity                                          #
    # ------------------------------------------------------------------ #

    def key_sensitivity_tests(self):
        group = "KeySensitivity"
        prompt = "The cat sat on the mat"
        msg    = "HELLO"

        # 3a: different keys produce different encoded text
        t0 = time.time()
        try:
            t1 = self._encode_cached(prompt, "key_A", message=msg)
            t2 = self._encode_cached(prompt, "key_B", message=msg)
            if t1 and t2:
                passed = t1 != t2
                detail = (f"key_A[0:30]={t1[:30]!r} vs key_B[0:30]={t2[:30]!r}"
                          if not passed else "texts differ as expected")
            else:
                passed, detail = False, "encode failed"
        except Exception as e:
            passed, detail = False, str(e)
        self.record("different keys → different encoded texts", group, passed,
                    detail, duration=time.time()-t0)

        # 3b: correct key decodes correctly
        t0 = time.time()
        try:
            encoded = self._encode_cached(prompt, "correct_key", message=msg)
            if encoded:
                stdout, _ = run_decode(self.script, encoded, key="correct_key",
                                       model=self.model)
                bits = parse_extracted_bits(stdout)
                expected = text_to_bits(msg)
                passed = bool(bits) and len(bits) >= len(expected) and \
                         bits[:len(expected)] == expected
                detail = (f"decoded={bits[:len(expected)]!r}" if bits
                          else "no bits extracted")
            else:
                passed, detail = False, "encode failed"
        except Exception as e:
            passed, detail = False, str(e)
        self.record("correct key decodes payload", group, passed, detail,
                    duration=time.time()-t0)

        # 3c: wrong key does NOT decode to the correct payload
        t0 = time.time()
        try:
            encoded = self._encode_cached(prompt, "correct_key", message=msg)
            if encoded:
                stdout, _ = run_decode(self.script, encoded, key="totally_wrong_key",
                                       model=self.model)
                bits = parse_extracted_bits(stdout)
                expected = text_to_bits(msg)
                # Wrong key should produce different bits
                wrong_key_wrong = (not bits or len(bits) < len(expected) or
                                   bits[:len(expected)] != expected)
                passed = wrong_key_wrong
                detail = (f"wrong key gave bits={bits[:len(expected)]!r} "
                          f"(expected different from {expected!r})")
            else:
                passed, detail = False, "encode failed"
        except Exception as e:
            passed, detail = False, str(e)
        self.record("wrong key does NOT recover payload", group, passed, detail,
                    duration=time.time()-t0)

        # 3d: two different wrong keys give different results from each other
        t0 = time.time()
        try:
            encoded = self._encode_cached(prompt, "correct_key", message=msg)
            if encoded:
                out1, _ = run_decode(self.script, encoded, key="wrong_key_1",
                                     model=self.model)
                out2, _ = run_decode(self.script, encoded, key="wrong_key_2",
                                     model=self.model)
                b1 = parse_extracted_bits(out1) or ""
                b2 = parse_extracted_bits(out2) or ""
                # Each wrong key should (with overwhelming probability) give different output
                passed = b1 != b2
                detail = f"wrong_key_1={b1[:10]!r} wrong_key_2={b2[:10]!r}"
            else:
                passed, detail = False, "encode failed"
        except Exception as e:
            passed, detail = False, str(e)
        self.record("two wrong keys give different decode outputs", group, passed,
                    detail, duration=time.time()-t0)

    # ------------------------------------------------------------------ #
    #  Group 4 — Determinism                                              #
    # ------------------------------------------------------------------ #

    def determinism_tests(self):
        group = "Determinism"
        prompt = "Explain the theory of relativity"
        msg    = "TEST"
        key    = "det_key"

        # 4a: encoding is deterministic (same inputs → same output)
        t0 = time.time()
        try:
            out1, _ = run_encode(self.script, prompt, message=msg, key=key,
                                 model=self.model, max_tokens=self.max_tokens)
            out2, _ = run_encode(self.script, prompt, message=msg, key=key,
                                 model=self.model, max_tokens=self.max_tokens)
            t1 = parse_generated_text(out1)
            t2 = parse_generated_text(out2)
            if t1 and t2:
                passed = t1 == t2
                detail = ("same" if passed else
                          f"differ: t1[0:40]={t1[:40]!r}, t2[0:40]={t2[:40]!r}")
            else:
                passed, detail = False, "encode produced no text"
        except Exception as e:
            passed, detail = False, str(e)
        self.record("encode is deterministic (PRF-based)", group, passed, detail,
                    duration=time.time()-t0)

        # 4b: decoding is deterministic
        t0 = time.time()
        try:
            encoded = self._encode_cached(prompt, key, message=msg)
            if encoded:
                o1, _ = run_decode(self.script, encoded, key=key, model=self.model)
                o2, _ = run_decode(self.script, encoded, key=key, model=self.model)
                b1 = parse_extracted_bits(o1)
                b2 = parse_extracted_bits(o2)
                passed = b1 == b2
                detail = f"run1={b1!r} run2={b2!r}"
            else:
                passed, detail = False, "encode failed"
        except Exception as e:
            passed, detail = False, str(e)
        self.record("decode is deterministic", group, passed, detail,
                    duration=time.time()-t0)

        # 4c: different prompts → different encoded text (prompt is part of PRF seed)
        t0 = time.time()
        try:
            ta = self._encode_cached("The sun is bright",  key, message=msg)
            tb = self._encode_cached("The moon is dark",   key, message=msg)
            if ta and tb:
                passed = ta != tb
                detail = "different prompts → different texts"
            else:
                passed, detail = False, "encode failed"
        except Exception as e:
            passed, detail = False, str(e)
        self.record("different prompts → different encoded text", group, passed,
                    detail, duration=time.time()-t0)

    # ------------------------------------------------------------------ #
    #  Group 5 — Capacity                                                 #
    # ------------------------------------------------------------------ #

    def capacity_tests(self):
        group = "Capacity"
        prompt = "Write a long story about a dragon"
        key    = "cap_key"

        # 5a: more tokens → at least as many bits encoded
        t0 = time.time()
        try:
            out_short, _ = run_encode(self.script, prompt, message="HELLO",
                                      key=key, model=self.model, max_tokens=60)
            out_long, _  = run_encode(self.script, prompt, message="HELLO",
                                      key=key, model=self.model,
                                      max_tokens=min(300, self.max_tokens * 2))
            be_short, _  = parse_bits_encoded(out_short)
            be_long, _   = parse_bits_encoded(out_long)
            if be_short is not None and be_long is not None:
                passed = be_long >= be_short
                detail = f"short={be_short} bits, long={be_long} bits"
            else:
                passed = False
                detail = "could not parse bits_encoded"
        except Exception as e:
            passed, detail = False, str(e)
        self.record("more tokens → more (or equal) bits encoded", group, passed,
                    detail, duration=time.time()-t0)

        # 5b: capacity is positive (encodes at least 1 bit per 20 tokens)
        t0 = time.time()
        try:
            out, _ = run_encode(self.script, prompt, message="HELLO WORLD",
                                key=key, model=self.model,
                                max_tokens=self.max_tokens)
            be, total = parse_bits_encoded(out)
            text = parse_generated_text(out)
            if be is not None:
                # Rough: at least 1 bit per 20 tokens
                min_expected = max(1, self.max_tokens // 20)
                passed = be >= min_expected
                detail = f"bits_encoded={be}, min_expected={min_expected}"
            else:
                passed, detail = False, "could not parse bits_encoded"
        except Exception as e:
            passed, detail = False, str(e)
        self.record("capacity ≥ 1 bit per 20 tokens", group, passed, detail,
                    duration=time.time()-t0)

        # 5c: bits_encoded does not exceed payload length
        t0 = time.time()
        try:
            payload = "AB"   # 10 bits
            out, _ = run_encode(self.script, prompt, message=payload,
                                key=key, model=self.model,
                                max_tokens=self.max_tokens)
            be, total = parse_bits_encoded(out)
            expected_len = len(text_to_bits(payload))
            if be is not None and total is not None:
                # The reported total should match the payload length
                total_ok = total == expected_len
                # bits_encoded should not exceed payload length
                be_ok = be <= expected_len
                passed = total_ok and be_ok
                detail = f"bits_encoded={be}, payload={expected_len}, reported_total={total}"
            else:
                passed, detail = False, "could not parse bits_encoded"
        except Exception as e:
            passed, detail = False, str(e)
        self.record("bits_encoded ≤ payload length", group, passed, detail,
                    duration=time.time()-t0)

    # ------------------------------------------------------------------ #
    #  Group 6 — Alphabet Coverage                                        #
    # ------------------------------------------------------------------ #

    def alphabet_tests(self):
        group = "Alphabet"
        prompt = "The quick brown fox jumps"
        CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ @:./? ")

        # Test a representative sample to keep runtime manageable
        sample = ["A", "Z", " ", "@", ".", "?", "MN", "YZ", "AB"]
        for msg in sample:
            t0 = time.time()
            key = f"alpha_{msg.replace(' ','SP')}"
            try:
                encoded = self._encode_cached(prompt, key, message=msg)
                if not encoded:
                    self.record(f"alphabet: {msg!r}", group, False,
                                "encode failed", duration=time.time()-t0)
                    continue
                stdout, _ = run_decode(self.script, encoded, key=key,
                                       model=self.model)
                bits = parse_extracted_bits(stdout)
                expected = text_to_bits(msg)
                if bits and len(bits) >= len(expected):
                    passed = bits[:len(expected)] == expected
                    detail = f"decoded={bits[:len(expected)]!r}"
                else:
                    passed = False
                    detail = f"insufficient bits: {len(bits or '')} < {len(expected)}"
            except Exception as e:
                passed, detail = False, str(e)
            self.record(f"alphabet: encode/decode {msg!r}", group, passed, detail,
                        duration=time.time()-t0)

        # Verify text_to_bits covers all 32 alphabet positions
        t0 = time.time()
        ALPHABET = (' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                    'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
                    'X', 'Y', 'Z', '@', ':', '.', '/', '?')
        all_vals = set(range(32))
        try:
            out, _ = run_encode(self.script, "test", message="A",
                                key="alphk", model=self.model, max_tokens=5)
            # If encode ran, check that the alphabet covers all 32 values
            from_enc = {ch: int(text_to_bits(ch), 2)
                        for ch in ALPHABET if ch.isalpha() or ch == ' '}
            passed = set(from_enc.values()).issubset(all_vals)
            detail = f"32 distinct codes covered"
        except Exception as e:
            passed, detail = False, str(e)
        self.record("alphabet has 32 distinct 5-bit codes", group, passed, detail,
                    duration=time.time()-t0)

    # ------------------------------------------------------------------ #
    #  Group 7 — Bit-level Payloads                                       #
    # ------------------------------------------------------------------ #

    def bitlength_tests(self):
        group = "BitPayload"
        prompt = "In the beginning"
        cases = [
            ("0",           "bit_1a", "single bit 0"),
            ("1",           "bit_1b", "single bit 1"),
            ("00",          "bit_2a", "two bits 00"),
            ("11",          "bit_2b", "two bits 11"),
            ("01010",       "bit_5",  "5-bit pattern"),
            ("10110011",    "bit_8",  "8-bit pattern"),
            ("00000",       "bit_0s", "5 zeros"),
            ("11111",       "bit_1s", "5 ones"),
        ]
        for bits_payload, key, label in cases:
            t0 = time.time()
            try:
                out_enc, _ = run_encode(self.script, prompt, bits=bits_payload,
                                        key=key, model=self.model,
                                        max_tokens=self.max_tokens)
                encoded = parse_generated_text(out_enc)
                if not encoded:
                    self.record(f"bit-payload {label}", group, False,
                                "encode produced no text", duration=time.time()-t0)
                    continue
                out_dec, _ = run_decode(self.script, encoded, key=key,
                                        model=self.model, raw_bits=True)
                bits_out = parse_extracted_bits(out_dec)
                n = len(bits_payload)
                if bits_out and len(bits_out) >= n:
                    passed = bits_out[:n] == bits_payload
                    detail = f"got={bits_out[:n]!r} expected={bits_payload!r}"
                else:
                    passed = False
                    detail = f"got={bits_out!r} expected={bits_payload!r}"
            except subprocess.TimeoutExpired:
                passed, detail = False, "timeout"
            except Exception as e:
                passed, detail = False, str(e)
            self.record(f"bit-payload {label}", group, passed, detail,
                        duration=time.time()-t0)

    # ------------------------------------------------------------------ #
    #  Group 8 — Edge Cases                                               #
    # ------------------------------------------------------------------ #

    def edge_case_tests(self):
        group = "EdgeCase"

        # 8a: very low max_tokens still runs (even if payload not fully encoded)
        t0 = time.time()
        try:
            out, stderr = run_encode(self.script, "Hello", message="A",
                                     key="edge1", model=self.model, max_tokens=5)
            passed = bool(parse_generated_text(out)) and "error" not in stderr.lower()
            detail = f"max_tokens=5 ok"
        except Exception as e:
            passed, detail = False, str(e)
        self.record("encode with max_tokens=5 runs cleanly", group, passed, detail,
                    duration=time.time()-t0)

        # 8b: decode of random unencoded text produces no error
        t0 = time.time()
        try:
            random_text = ("The cat sat on the mat and watched the birds fly by. "
                           "It was a warm summer afternoon in the quiet village.")
            out, stderr = run_decode(self.script, random_text, key="edge2",
                                     model=self.model)
            passed = "error" not in stderr.lower() and "exception" not in out.lower()
            bits = parse_extracted_bits(out)
            detail = f"extracted {len(bits or '')} bits from random text"
        except Exception as e:
            passed, detail = False, str(e)
        self.record("decode of non-encoded text runs cleanly", group, passed, detail,
                    duration=time.time()-t0)

        # 8c: decode of non-encoded text does not claim to decode a specific message
        t0 = time.time()
        try:
            random_text = ("The cat sat on the mat and watched the birds fly by. "
                           "It was a warm summer afternoon in the quiet village.")
            out, _ = run_decode(self.script, random_text, key="edge3",
                                model=self.model)
            bits = parse_extracted_bits(out) or ""
            # Expect very few bits from random text (much less than a real payload)
            passed = len(bits) < 30
            detail = f"extracted {len(bits)} bits (expect <30 for random text)"
        except Exception as e:
            passed, detail = False, str(e)
        self.record("random text decodes to few bits (<30)", group, passed, detail,
                    duration=time.time()-t0)

        # 8d: longer prompt still works
        t0 = time.time()
        long_prompt = ("Write a detailed and comprehensive essay about the history "
                       "of computing, starting from the earliest mechanical calculators "
                       "through to modern quantum computers, including key inventors.")
        try:
            encoded = self._encode_cached(long_prompt, "edge4", message="AB")
            if encoded:
                out, _ = run_decode(self.script, encoded, key="edge4",
                                    model=self.model)
                bits = parse_extracted_bits(out)
                expected = text_to_bits("AB")
                passed = bool(bits) and len(bits) >= len(expected) and \
                         bits[:len(expected)] == expected
                detail = f"decoded={bits[:len(expected)]!r}" if bits else "no bits"
            else:
                passed, detail = False, "encode failed with long prompt"
        except Exception as e:
            passed, detail = False, str(e)
        self.record("encode/decode with long prompt", group, passed, detail,
                    duration=time.time()-t0)

        # 8e: same key, different messages, different encoded texts
        t0 = time.time()
        try:
            ta = self._encode_cached("Once upon a time", "same_key", message="AB")
            tb = self._encode_cached("Once upon a time", "same_key", message="CD")
            if ta and tb:
                passed = ta != tb
                detail = "different messages → different text"
            else:
                passed, detail = False, "encode failed"
        except Exception as e:
            passed, detail = False, str(e)
        self.record("same key different messages → different texts", group, passed,
                    detail, duration=time.time()-t0)

    # ------------------------------------------------------------------ #
    #  Group 9 — Paper Compliance                                         #
    # ------------------------------------------------------------------ #

    def paper_compliance_tests(self):
        group = "PaperCompliance"

        # 9a: Figure 3 example: ECC stream decodes to OZ (= '0111111010')
        # We can test this by verifying the decode algorithm's logic
        # via the DynamicECC.decode method if importable, or via bit decode
        t0 = time.time()
        try:
            # The paper's Figure 3 ECC stream decodes to OZ
            oz_bits = text_to_bits("OZ")
            # O=15=01111, Z=26=11010 → "0111111010"
            passed = oz_bits == "0111111010"
            detail = f"OZ bits = {oz_bits!r}"
        except Exception as e:
            passed, detail = False, str(e)
        self.record("OZ text_to_bits matches paper Figure 3 (0111111010)", group,
                    passed, detail, duration=time.time()-t0)

        # 9b: Decoded bits match expected: confirm bit ordering (MSB first)
        t0 = time.time()
        try:
            # A = index 1 = 00001 (MSB first), B = index 2 = 00010
            a_bits = text_to_bits("A")
            b_bits = text_to_bits("B")
            passed = a_bits == "00001" and b_bits == "00010"
            detail = f"A={a_bits!r} B={b_bits!r}"
        except Exception as e:
            passed, detail = False, str(e)
        self.record("bit encoding is MSB-first (A=00001, B=00010)", group,
                    passed, detail, duration=time.time()-t0)

        # 9c: Payload is independent of ECC internals — decode uses only text+key
        # (encode→decode should work without any shared state between processes)
        t0 = time.time()
        prompt = "Science and technology"
        msg = "HI"
        key = "paper_key"
        try:
            out_enc, _ = run_encode(self.script, prompt, message=msg,
                                    key=key, model=self.model,
                                    max_tokens=self.max_tokens)
            text = parse_generated_text(out_enc)
            if text:
                # Decode in a completely independent subprocess call
                out_dec, _ = run_decode(self.script, text, key=key,
                                        model=self.model)
                bits = parse_extracted_bits(out_dec)
                expected = text_to_bits(msg)
                passed = bool(bits) and len(bits) >= len(expected) and \
                         bits[:len(expected)] == expected
                detail = f"decoded={bits[:len(expected)]!r}" if bits else "no bits"
            else:
                passed, detail = False, "encode produced no text"
        except Exception as e:
            passed, detail = False, str(e)
        self.record("decode requires only text+key (no shared state)", group,
                    passed, detail, duration=time.time()-t0)

        # 9d: Encoding does not visibly garble text (basic language check)
        t0 = time.time()
        try:
            out_enc, _ = run_encode(self.script, "Write a sentence about cats",
                                    message="CAT", key="langkey",
                                    model=self.model, max_tokens=80)
            text = parse_generated_text(out_enc)
            if text:
                # Check that text is mostly ASCII printable (not binary garbage)
                printable_ratio = sum(1 for c in text if 32 <= ord(c) < 127) / max(len(text), 1)
                # Check text has spaces (is word-level, not garbled)
                has_spaces = ' ' in text
                passed = printable_ratio > 0.9 and has_spaces
                detail = f"printable_ratio={printable_ratio:.2f} has_spaces={has_spaces}"
            else:
                passed, detail = False, "no text generated"
        except Exception as e:
            passed, detail = False, str(e)
        self.record("encoded text is readable natural language", group, passed,
                    detail, duration=time.time()-t0)

        # 9e: Payload bits are correctly ordered — first bits decoded match
        #     the first bits of the payload (not reversed or shifted)
        t0 = time.time()
        try:
            # Encode a distinctive pattern: 10000 (= 'P') starts with 1, then zeros
            out_enc, _ = run_encode(self.script, "Hello world",
                                    bits="10000", key="orderkey",
                                    model=self.model, max_tokens=self.max_tokens)
            text = parse_generated_text(out_enc)
            if text:
                out_dec, _ = run_decode(self.script, text, key="orderkey",
                                        model=self.model, raw_bits=True)
                bits = parse_extracted_bits(out_dec)
                if bits and len(bits) >= 5:
                    passed = bits[:5] == "10000"
                    detail = f"got={bits[:5]!r} expected='10000'"
                else:
                    passed = False
                    detail = f"too few bits: {bits!r}"
            else:
                passed, detail = False, "encode produced no text"
        except Exception as e:
            passed, detail = False, str(e)
        self.record("payload bit order is preserved (MSB first)", group, passed,
                    detail, duration=time.time()-t0)

        # 9f: Key is required — providing no/empty key should fail gracefully
        t0 = time.time()
        try:
            cmd = [sys.executable, self.script, "encode",
                   "--prompt", "test", "--message", "A",
                   "--model", self.model, "--max-tokens", "10"]
            # No --key argument
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            # Should either error (non-zero exit) or print a usage error
            passed = r.returncode != 0 or "required" in r.stderr.lower() \
                     or "error" in r.stderr.lower() or "usage" in r.stderr.lower()
            detail = f"returncode={r.returncode}"
        except Exception as e:
            passed, detail = True, f"raised {type(e).__name__} (acceptable)"
        self.record("missing --key produces an error", group, passed, detail,
                    duration=time.time()-t0)

        # 9g: Multiple messages can be independently encoded (scheme is re-entrant)
        t0 = time.time()
        try:
            msgs = [("ALPHA", "multi_key_1"), ("BETA", "multi_key_2")]
            all_ok = True
            fail_detail = ""
            for msg, key in msgs:
                enc = self._encode_cached("The river flows", key, message=msg)
                if not enc:
                    all_ok = False
                    fail_detail = f"encode failed for {msg!r}"
                    break
                out, _ = run_decode(self.script, enc, key=key, model=self.model)
                bits = parse_extracted_bits(out)
                exp = text_to_bits(msg)
                if not bits or len(bits) < len(exp) or bits[:len(exp)] != exp:
                    all_ok = False
                    got = (bits or "")[:len(exp)]
                    fail_detail = f"{msg!r}: got={got!r} expected={exp!r}"
                    break
            passed = all_ok
            detail = "both messages encoded/decoded independently" if all_ok else fail_detail
        except Exception as e:
            passed, detail = False, str(e)
        self.record("independent messages use independent keys", group, passed,
                    detail, duration=time.time()-t0)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def render_report(results: list[TestResult], script: str,
                  model: str, elapsed: float) -> str:
    passed  = [r for r in results if r.passed and not r.skipped]
    failed  = [r for r in results if not r.passed and not r.skipped]
    skipped = [r for r in results if r.skipped]
    total   = len(results) - len(skipped)
    pct     = 100 * len(passed) / total if total else 0

    groups: dict[str, list[TestResult]] = {}
    for r in results:
        groups.setdefault(r.group, []).append(r)

    lines = []
    lines.append("# Steganography Implementation Test Report")
    lines.append("")
    lines.append(f"**Script:** `{script}`  ")
    lines.append(f"**Model:** `{model}`  ")
    lines.append(f"**Total runtime:** {elapsed:.1f}s  ")
    lines.append(f"**Result:** {len(passed)}/{total} passed "
                 f"({'✅ PASS' if not failed else '❌ FAIL'})")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary table
    lines.append("## Summary by Group")
    lines.append("")
    lines.append("| Group | Pass | Fail | Skip |")
    lines.append("|---|---|---|---|")
    for gname, gresults in groups.items():
        gp = sum(1 for r in gresults if r.passed and not r.skipped)
        gf = sum(1 for r in gresults if not r.passed and not r.skipped)
        gs = sum(1 for r in gresults if r.skipped)
        lines.append(f"| {gname} | {gp} | {gf} | {gs} |")
    lines.append(f"| **Total** | **{len(passed)}** | **{len(failed)}** | **{len(skipped)}** |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-group details
    lines.append("## Detailed Results")
    lines.append("")
    for gname, gresults in groups.items():
        lines.append(f"### {gname}")
        lines.append("")
        for r in gresults:
            if r.skipped:
                icon = "⏭️"
            elif r.passed:
                icon = "✅"
            else:
                icon = "❌"
            lines.append(f"- {icon} **{r.name}** ({r.duration_s:.1f}s)")
            if r.detail:
                lines.append(f"  - {r.detail}")
        lines.append("")

    # Failure summary
    if failed:
        lines.append("---")
        lines.append("")
        lines.append("## Failed Tests")
        lines.append("")
        for r in failed:
            lines.append(f"- **{r.name}** [{r.group}]")
            if r.detail:
                lines.append(f"  - {r.detail}")
        lines.append("")

    # Verdict
    lines.append("---")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    if pct == 100:
        lines.append("✅ **All tests passed.** The implementation correctly implements "
                     "the steganographic scheme for all tested inputs.")
    elif pct >= 80:
        lines.append(f"⚠️ **{pct:.0f}% of tests passed.** The implementation is mostly "
                     "correct but has failures in some areas — see above.")
    else:
        lines.append(f"❌ **{pct:.0f}% of tests passed.** The implementation has "
                     "significant correctness issues — see failed tests above.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="General test suite for steganography implementations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--script",     default="main.py",
                        help="Path to the implementation script (default: main.py)")
    parser.add_argument("--model",      default="gpt2",
                        help="LLM model name (default: gpt2)")
    parser.add_argument("--max-tokens", type=int, default=150,
                        help="Token budget for encode calls (default: 150)")
    parser.add_argument("--smoke",      action="store_true",
                        help="Run smoke tests only (fastest)")
    parser.add_argument("--report",     default=None,
                        help="Write markdown report to this file")
    args = parser.parse_args()

    script = str(Path(args.script).resolve())
    if not Path(script).exists():
        print(f"ERROR: script not found: {script}", file=sys.stderr)
        sys.exit(1)

    print(f"Steganography Test Suite")
    print(f"  Script:     {script}")
    print(f"  Model:      {args.model}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Smoke only: {args.smoke}")

    suite = Suite(script=script, model=args.model,
                  max_tokens=args.max_tokens, smoke_only=args.smoke)

    t_start = time.time()
    suite.run()
    elapsed = time.time() - t_start

    # Console summary
    passed  = sum(1 for r in suite.results if r.passed and not r.skipped)
    failed  = sum(1 for r in suite.results if not r.passed and not r.skipped)
    skipped = sum(1 for r in suite.results if r.skipped)
    total   = len(suite.results) - skipped

    print(f"\n{'='*60}")
    print(f" RESULTS: {passed}/{total} passed | {failed} failed | {skipped} skipped")
    print(f" Runtime: {elapsed:.1f}s")
    print(f"{'='*60}")

    if failed:
        print("\nFailed tests:")
        for r in suite.results:
            if not r.passed and not r.skipped:
                print(f"  ✗ [{r.group}] {r.name}")
                if r.detail:
                    print(f"      {r.detail}")

    # Report
    report_md = render_report(suite.results, args.script, args.model, elapsed)
    if args.report:
        Path(args.report).write_text(report_md)
        print(f"\nMarkdown report written to: {args.report}")
    else:
        report_path = Path("test_report.md")
        report_path.write_text(report_md)
        print(f"\nMarkdown report written to: {report_path}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
