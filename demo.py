#!/usr/bin/env python3
"""
External demo driver. Futhark programs are pure-functional kernels, so need an
external driver to do I/O related things. String processing is also cleaner
externally (possible in Futhark but not a good fit).

This driver downloads and prepares the dataset, tokenizes and serializes
training data, calls the Futhark 'demo' entry point, and formats/prints the
returned loss and token IDs.

Usage:
  python3 demo.py                       # default: microgpt_c
  python3 demo.py --backend microgpt_c
  python3 demo.py --backend microgpt_multicore
  python3 demo.py --backend microgpt_opencl
  python3 demo.py --backend microgpt_cuda

Build all backends with:
  make
"""

import argparse
import math
import os
import random
import re
import subprocess
import sys
import urllib.request

NAMES_URL = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
BLOCK_SIZE = 16   # must match the constant compiled into the binary

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--backend", default="microgpt_c")
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--samples", type=int, default=20)
args = parser.parse_args()

binary = f"./{args.backend}"
if not os.path.exists(binary):
    sys.exit(
        f"Binary not found: {binary}\n"
        "Build the backends with make"
    )

# Dataset (mirrors microgpt_orig.py lines 15-27)
if not os.path.exists("input.txt"):
    print("Downloading names.txt ...")
    urllib.request.urlretrieve(NAMES_URL, "input.txt")

docs = [line.strip() for line in open("input.txt") if line.strip()]
random.seed(42)
random.shuffle(docs)

uchars = sorted(set("".join(docs)))
BOS = len(uchars)
vocab_size = BOS + 1
print(f"num docs: {len(docs)}, vocab size: {vocab_size}")
print(f"expected initial loss: ~{math.log(vocab_size):.4f}")

# Tokenize: raw character IDs, padded to BLOCK_SIZE (Futhark prepends BOS itself)
rows, lens = [], []
for doc in docs:
    toks = [uchars.index(ch) for ch in doc]
    n = min(len(toks), BLOCK_SIZE)
    rows.append(toks[:n] + [0] * (BLOCK_SIZE - n))
    lens.append(n)

# Write Futhark text-format input
data = "\n".join(
    [
        f"{vocab_size}i64",
        "1337i32",
        "["
        + ", ".join(
            "[" + ", ".join(f"{t}i64" for t in row) + "]" for row in rows
        )
        + "]",
        "[" + ", ".join(f"{n}i64" for n in lens) + "]",
        f"{args.steps}i64",
        "0.5f32",
        f"{args.samples}i64",
        "42i32",
    ]
)

print(f"training for {args.steps} steps ...", flush=True)
result = subprocess.run(
    [binary, "--entry-point", "demo"],
    input=data.encode(),
    capture_output=True,
)
if result.returncode != 0:
    sys.exit(result.stderr.decode())

# Parse output: first value is loss (f32), rest are token IDs (i64)
output = result.stdout.decode()
loss = float(re.search(r"([-\d.e+]+)f32", output).group(1))
tokens = [int(m) for m in re.findall(r"(-?\d+)i64", output)]
print(f"final loss: {loss:.4f}")

print("\n--- inference (new, hallucinated names) ---")
for i in range(args.samples):
    row = tokens[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE]
    chars = [uchars[t] for t in row if t >= 0]
    print(f"sample {i+1:2d}: {''.join(chars)}")
