# ACL paper draft

Interpretability paper: the LatentMAS $K=40$ KV relay is a prefix, not an
instance-specific message. Naive eviction is the **delete test** (and a KV
side-effect), not the claimed method.

## Build

```bash
cd paper/acl
python3 make_figures.py
pdflatex -interaction=nonstopmode main
bibtex main
pdflatex -interaction=nonstopmode main
pdflatex -interaction=nonstopmode main
```

`main.tex` already uses the official ACL preamble (`\usepackage[review]{acl}`).
Style files `acl.sty` and `acl_natbib.bst` are vendored from
[acl-org/acl-style-files](https://github.com/acl-org/acl-style-files).

For camera-ready, switch to `\usepackage[final]{acl}` and fill in authors.

## Figures

| File | What |
|---|---|
| `figs/swap_budget` | Real vs shuffled vs no cache (MedQA) |
| `figs/acc_evict` | $K=40$ full vs eviction accuracy |
| `figs/latency_agents` | Per-agent time, GSM8K / GPQA |
| `figs/memory_batch` | Peak GB vs Judger batch size |

## Numbers

ARR suite: Qwen3-14B, temp 0.6 / top-p 0.95, three seeds, reports under
`artifacts/exp_latency_mem/arr_*`.
