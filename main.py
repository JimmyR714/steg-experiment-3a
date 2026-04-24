#!/usr/bin/env python3
"""
Undetectable Steganography for Language Models

Implementation of the scheme from:
  "Undetectable Steganography for Language Models" by Or Zamir (2024)
  Published in Transactions on Machine Learning Research (10/2024)

Implements Algorithms 3 & 4 (single-query undetectable steganography)
with the binary reduction from CGZ Section 4.1.

Usage:
  python main.py encode --prompt "Write a poem about nature" --message "SECRET" --key mykey
  python main.py decode --text "<encoded text>" --key mykey
"""

import argparse
import hashlib
import hmac
import json
import math
import struct
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Pseudorandom Function (PRF)
# ---------------------------------------------------------------------------

def prf(key: str, prf_input: list) -> float:
    """
    HMAC-SHA256 based PRF. Maps (key, input) -> float in [0, 1).
    Cryptographically secure unlike the reference impl's random.seed approach.
    """
    data = (str(key) + "||" + str(prf_input)).encode()
    h = hmac.new(b"steg-prf-key", data, hashlib.sha256).digest()
    val = struct.unpack(">Q", h[:8])[0]
    return val / (2**64)


# ---------------------------------------------------------------------------
# Consistent permutation of token IDs (spreads indices uniformly)
# ---------------------------------------------------------------------------

def consistent_perm(key: str, n: int) -> tuple[list[int], list[int]]:
    """Generate a deterministic permutation and its inverse from a key."""
    import random as _rng
    _rng.seed(str(key))
    perm = list(range(n))
    _rng.shuffle(perm)
    inv_perm = [0] * n
    for i in range(n):
        inv_perm[perm[i]] = i
    return perm, inv_perm


def apply_perm(probs: torch.Tensor, perm: list[int]) -> torch.Tensor:
    """Apply permutation to a probability vector."""
    result = probs.clone().detach()
    for i in range(len(probs)):
        result[perm[i]] = probs[i]
    return result


# ---------------------------------------------------------------------------
# Binary reduction (CGZ Section 4.1)
# Decomposes token selection into ceil(log2(vocab_size)) binary decisions.
# ---------------------------------------------------------------------------

def binarize_setup(tokenizer) -> tuple[int, dict, dict]:
    """Setup for binary reduction: compute bit length needed."""
    blen = math.ceil(math.log2(len(tokenizer)))
    token_to_id = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in token_to_id.items()}
    return blen, token_to_id, id_to_token


def binarize_next(probs: torch.Tensor, ind: int, blen: int, prefix: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For a given bit position `ind` and prefix of already-decided bits,
    compute probabilities of the next bit being 0 or 1.

    Args:
        probs: probability distribution over permuted token IDs
        ind: current bit index (0 to blen-1)
        blen: total number of bits per token
        prefix: the bits decided so far (as an integer)
    """
    p0 = torch.tensor([0.0])
    p1 = torch.tensor([0.0])
    lo = prefix << (blen - ind)
    hi = min((prefix + 1) << (blen - ind), len(probs))
    for token_id in range(lo, hi):
        if (token_id >> (blen - ind - 1)) % 2 == 0:
            p0 += probs[token_id]
        else:
            p1 += probs[token_id]
    return p0, p1


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------

def compute_score_function(key: str, prf_input: list, bit: str) -> float:
    """
    Compute score s(x, k) for one binary bit.
    s(x, k) = ln(1/k) if x=1, ln(1/(1-k)) if x=0.
    """
    u = prf(key, prf_input)
    v = u if bit == "1" else (1.0 - u)
    return -math.log(max(v, 1e-30))


def normalize_score(score: float, length: int) -> float:
    """Normalize cumulative score: (score - length) / sqrt(length)."""
    if length <= 0:
        return 0.0
    return (score - length) / math.sqrt(length)


# ---------------------------------------------------------------------------
# Dynamic Error Correcting Code with noiseless feedback (Section 5)
# Alphabet: message = {0, 1}, code = {0, 1, <} where < is backspace.
# ---------------------------------------------------------------------------

class DynamicECC:
    """
    Dynamic ECC with feedback from Section 5 / Theorem 5.3.
    Encodes binary message bits into a ternary code {0, 1, <}.
    """

    def __init__(self, input_bits: str):
        """
        Args:
            input_bits: the payload as a binary string, e.g. "01101"
        """
        self.input = input_bits
        self.last_index_written = -1
        self.suffix_to_remove = 0
        self.stream = []
        self.default_symbol = "0"

    @staticmethod
    def decode(stream: list[str]) -> str:
        """Decode a stream of code symbols {0, 1, <} into binary message string."""
        message = []
        for symbol in stream:
            if symbol == "<":
                if message:
                    message.pop()
            else:
                message.append(symbol)
        return "".join(message)

    def update(self, symbol: str):
        """
        Feed a received code symbol. This is called by both encoder (who knows
        the true next symbol) and the emulated retriever within the encoder.
        """
        self.stream.append(symbol)

        if self.suffix_to_remove:
            if symbol == "<":
                self.suffix_to_remove -= 1
            else:
                self.suffix_to_remove += 1
            return

        next_sym = (
            self.input[self.last_index_written + 1]
            if (self.last_index_written + 1 < len(self.input))
            else self.default_symbol
        )

        if symbol == next_sym:
            self.last_index_written += 1
        elif symbol == "<":
            if self.last_index_written > -1:
                self.last_index_written -= 1
        else:
            self.suffix_to_remove = 1

    def next_symbol(self) -> str:
        """Get the next code symbol to transmit."""
        if self.suffix_to_remove:
            return "<"
        next_sym = (
            self.input[self.last_index_written + 1]
            if (self.last_index_written + 1 < len(self.input))
            else self.default_symbol
        )
        return next_sym


# ---------------------------------------------------------------------------
# Message encoding: string <-> bit string
# ---------------------------------------------------------------------------

def message_to_bits(message: str) -> str:
    """Convert a UTF-8 string to a binary string."""
    bits = []
    for byte in message.encode("utf-8"):
        bits.append(format(byte, "08b"))
    return "".join(bits)


def bits_to_message(bits: str) -> str:
    """Convert a binary string back to a UTF-8 string."""
    # Trim to multiple of 8
    n = (len(bits) // 8) * 8
    if n == 0:
        return ""
    data = bytearray()
    for i in range(0, n, 8):
        byte_val = int(bits[i : i + 8], 2)
        data.append(byte_val)
    return data.decode("utf-8", errors="replace").rstrip("\x00")


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class LMWrapper:
    """Wraps a HuggingFace causal LM for token generation."""

    def __init__(self, model_name: str = "gpt2", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print(f"Loading model '{model_name}' on {device}...", file=sys.stderr)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = len(self.tokenizer)
        self.blen, self.token_to_id, self.id_to_token = binarize_setup(self.tokenizer)

    def tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=2048)

    def detokenize(self, token_ids: torch.Tensor) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def get_probs(self, input_ids: torch.Tensor, past_kv=None):
        """Get next-token probabilities. Returns (probs, past_kv)."""
        with torch.no_grad():
            if past_kv is not None:
                output = self.model(
                    input_ids[:, -1:],
                    past_key_values=past_kv,
                    attention_mask=torch.ones(1, input_ids.shape[1], device=self.device),
                )
            else:
                output = self.model(input_ids)
        logits = output.logits[:, -1, : self.vocab_size]
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu()[0, :]
        return probs, output.past_key_values


# ---------------------------------------------------------------------------
# Encode: Generate response with hidden payload (Algorithm 3)
# ---------------------------------------------------------------------------

def encode(
    lm: LMWrapper,
    prompt: str,
    payload: str,
    key: str,
    max_tokens: int = 200,
    threshold: float = 2.0,
) -> tuple[str, int]:
    """
    Encode a secret payload into an LLM-generated response.

    Implements Algorithm 3 from the paper (single-query steganography).

    Args:
        lm: language model wrapper
        prompt: the prompt for text generation
        payload: secret message to hide
        key: secret key string
        max_tokens: max tokens to generate
        threshold: score threshold t (paper uses 2.0)

    Returns:
        (generated_text, num_bits_hidden, generated_token_ids)
    """
    payload_bits = message_to_bits(payload)
    perm, inv_perm = consistent_perm(key, lm.vocab_size)

    ecc = DynamicECC(payload_bits)
    symbol = ecc.next_symbol()
    scores = {"0": 0.0, "1": 0.0, "<": 0.0}
    score_length = 0

    prompt_text_len = len(prompt)
    input_ids = lm.tokenize(prompt).to(lm.device)
    attn = torch.ones_like(input_ids)
    past = None
    generated_token_ids = []

    for i in range(max_tokens):
        probs, past = lm.get_probs(input_ids, past)
        probs_permed = apply_perm(probs, perm)

        # Build token via binary decisions
        token_id = 0
        for ind in range(lm.blen):
            p0, p1 = binarize_next(probs_permed, ind, lm.blen, token_id)
            token_id = token_id << 1

            # Use PRF to determine the binary bit (embed the payload)
            k = prf(key, [i, ind, symbol])
            if k < p1 / (p0 + p1 + 1e-30):
                token_id += 1

            # Update scores for all possible code symbols
            score_length += 1
            bit_str = str(token_id % 2)
            for s in ["0", "1", "<"]:
                scores[s] += compute_score_function(key, [i, ind, s], bit_str)
                if normalize_score(scores[s], score_length) > threshold:
                    ecc.update(s)
                    symbol = ecc.next_symbol()
                    scores = {"0": 0.0, "1": 0.0, "<": 0.0}
                    score_length = 0
                    break

        # Map back through inverse permutation to get real token ID
        real_token_id = inv_perm[token_id]
        generated_token_ids.append(real_token_id)
        token_tensor = torch.tensor([[real_token_id]], device=lm.device)
        input_ids = torch.cat([input_ids, token_tensor], dim=-1)
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

        # Stop on EOS
        if real_token_id == lm.tokenizer.eos_token_id:
            break

    generated_text = lm.detokenize(input_ids.detach().cpu()[0])
    response_text = generated_text[prompt_text_len:]
    bits_hidden = ecc.last_index_written + 1
    return response_text, bits_hidden, generated_token_ids


# ---------------------------------------------------------------------------
# Decode: Extract payload from text (Algorithm 4)
# ---------------------------------------------------------------------------

def decode(
    lm: LMWrapper,
    key: str,
    threshold: float = 2.0,
    token_ids: list[int] = None,
    text: str = None,
    prompt: str = None,
) -> str:
    """
    Decode a hidden payload from text.

    Implements Algorithm 4 from the paper (single-query retriever).
    The retriever does NOT need the model or prompt — it works purely from
    the token values and PRF values.

    Args:
        lm: language model wrapper (only used for tokenizer + binarization info)
        key: secret key string
        threshold: score threshold t (must match encoding)
        token_ids: exact token IDs (preferred, avoids retokenization issues)
        text: response text (will be tokenized; may lose fidelity with BPE)
        prompt: if provided with text, tokenize prompt+text and skip prompt

    Returns:
        decoded secret message
    """
    perm, inv_perm = consistent_perm(key, lm.vocab_size)

    if token_ids is not None:
        response_token_list = list(token_ids)
    elif text is not None:
        if prompt:
            prompt_tokens = lm.tokenize(prompt)[0]
            full_tokens = lm.tokenize(prompt + text)[0]
            response_token_list = full_tokens[len(prompt_tokens):].tolist()
        else:
            response_token_list = lm.tokenize(text)[0].tolist()
    else:
        raise ValueError("Either token_ids or text must be provided")

    # Filter out EOS tokens
    eos_id = lm.tokenizer.eos_token_id
    response_token_list = [t for t in response_token_list if t != eos_id]

    stream = []
    scores = {"0": 0.0, "1": 0.0, "<": 0.0}
    score_length = 0

    for i in range(len(response_token_list)):
        permuted_id = perm[response_token_list[i]]
        token_bits = format(permuted_id, f"0{lm.blen}b")

        for ind in range(lm.blen):
            score_length += 1
            bit = token_bits[ind]
            for s in ["0", "1", "<"]:
                scores[s] += compute_score_function(key, [i, ind, s], bit)
                if normalize_score(scores[s], score_length) > threshold:
                    stream.append(s)
                    scores = {"0": 0.0, "1": 0.0, "<": 0.0}
                    score_length = 0
                    break

    decoded_bits = DynamicECC.decode(stream)
    return bits_to_message(decoded_bits)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Undetectable Steganography for Language Models (Zamir 2024)"
    )
    parser.add_argument("mode", choices=["encode", "decode"], help="Encode or decode")
    parser.add_argument("--prompt", help="Prompt for text generation (encode mode)")
    parser.add_argument("--message", help="Secret message to hide (encode mode)")
    parser.add_argument("--text", help="Text to decode from (decode mode, less reliable)")
    parser.add_argument("--output", "-o", help="Output file for encoded data (JSON with text + token IDs)")
    parser.add_argument("--input", "-i", help="Input file with encoded data (JSON from encode)")
    parser.add_argument("--key", required=True, help="Secret key")
    parser.add_argument("--model", default="gpt2", help="HuggingFace model (default: gpt2)")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens (encode)")
    parser.add_argument("--threshold", type=float, default=2.0, help="Score threshold t")
    args = parser.parse_args()
    lm = LMWrapper(model_name=args.model)

    if args.mode == "encode":
        if not args.prompt:
            parser.error("--prompt is required for encode mode")
        if not args.message:
            parser.error("--message is required for encode mode")

        response, bits_hidden, token_ids = encode(
            lm=lm,
            prompt=args.prompt,
            payload=args.message,
            key=args.key,
            max_tokens=args.max_tokens,
            threshold=args.threshold,
        )
        print(response)
        print(f"\n[Hidden {bits_hidden} bits of payload]", file=sys.stderr)

        if args.output:
            data = {"text": response, "token_ids": token_ids, "model": args.model}
            with open(args.output, "w") as f:
                json.dump(data, f)
            print(f"Saved encoded data to {args.output}", file=sys.stderr)

    elif args.mode == "decode":
        if args.input:
            with open(args.input) as f:
                data = json.load(f)
            result = decode(
                lm=lm,
                key=args.key,
                threshold=args.threshold,
                token_ids=data["token_ids"],
            )
        elif args.text:
            result = decode(
                lm=lm,
                key=args.key,
                threshold=args.threshold,
                text=args.text,
                prompt=args.prompt,
            )
        else:
            parser.error("--input or --text is required for decode mode")

        print(result)


if __name__ == "__main__":
    main()
