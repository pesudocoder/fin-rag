# Notebooks

## `train_distilbert.ipynb` — Phase 3 DistilBERT fine-tuning (Google Colab)

The local machine runs `torch 2.13.0+cpu` with no GPU, so fine-tuning happens on
Colab's free T4. This notebook is self-contained: it does not import from `src/`,
because Colab has no copy of the repo.

### 1. Open in Colab

Upload `train_distilbert.ipynb` at [colab.research.google.com](https://colab.research.google.com)
(**File → Upload notebook**), or open it from GitHub once the repo is pushed.

### 2. Enable the GPU

**Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save.**

Do this *before* running anything. Cell 1 calls `nvidia-smi` and aborts if no GPU
is attached, so you cannot accidentally start a CPU run that would take hours.

### 3. Run cells 1–2

Cell 2 pins `transformers==5.15.0`, `datasets==5.0.1`, `accelerate>=1.1.0` and
`scikit-learn==1.9.0` to match the local environment. Colab ships an older
`transformers` by default, so this step is not optional — the notebook uses 5.x
argument names that do not exist in 4.x.

If Colab prompts to restart the runtime after install, restart and re-run from
cell 1.

### 4. Upload the data (cell 5)

When the file picker appears, upload **three** files from the local repo:

| File | Local path |
|---|---|
| `train.csv` | `data/processed/train.csv` |
| `val.csv` | `data/processed/val.csv` |
| `baseline_metrics.json` | `results/baseline_metrics.json` |

`baseline_metrics.json` supplies the Phase 2 numbers for the comparison table in
cell 14, so they are read from the recorded file rather than retyped.

**Do not upload `test.csv`.** The cell raises if it is present. The test split is
read by exactly one script in this project, `src/model/evaluate_test.py`, and only
for final evaluation. Uploading it here would put it one careless cell away from
becoming a second validation set.

Uploaded files land in `/content/` and are lost when the runtime disconnects — if
that happens, re-run from cell 5.

### 5. Run the remaining cells

Cells 11 and 12 train the two variants. On a T4 with `fp16=True`, expect roughly
3–6 minutes each (2,947 training rows, 4 epochs, batch size 16). Both are trained
in one session so they share an identical environment and seed.

### 6. Download the outputs (cell 15)

Four files download:

| File | Where it goes locally |
|---|---|
| `distilbert_standard.zip` | unzip into `models/distilbert_standard/` |
| `distilbert_weighted.zip` | unzip into `models/distilbert_weighted/` |
| `distilbert_history.json` | `results/distilbert_history.json` |
| `val_comparison.json` | `results/val_comparison.json` |

Each model folder is roughly 250 MB. `models/` is gitignored — **do not commit
the checkpoints.** The JSON files in `results/` are tracked and are the evidence
trail.

#### The zips extract one level nested — flatten them

`!zip -r distilbert_standard.zip distilbert_standard` stores the folder itself
inside the archive, so extracting it inside `models/` produces:

```
models/distilbert_standard/distilbert_standard/config.json   <- wrong
```

`evaluate_test.py` looks for `config.json` directly under
`models/distilbert_standard/` and will report the checkpoint as unloadable
otherwise. Flatten after extracting:

```powershell
# PowerShell, from the repo root
Move-Item models\distilbert_standard\distilbert_standard\* models\distilbert_standard\
Remove-Item models\distilbert_standard\distilbert_standard
```

Verify before running the evaluation — each folder should contain
`config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json` and `vocab.txt` at its top level:

```powershell
Get-ChildItem models\distilbert_standard, models\distilbert_weighted
```

If the browser blocks the multi-file download, allow pop-ups for
`colab.research.google.com`, or mount Drive and copy the files there instead.

### 7. Back on the local machine

```bash
# after unzipping both model folders into models/
python -m src.model.distilbert --checkpoint models/distilbert_weighted   # val sanity check
python -m src.model.evaluate_test                                        # final test evaluation
```

`evaluate_test.py` skips any checkpoint it cannot find and reports which were
missing, so it runs even before training is done.

### Reproducibility caveat

All seeds are set from `RANDOM_SEED = 42`, but **GPU training is not fully
deterministic.** cuDNN kernel selection and non-deterministic CUDA reduction order
mean two runs with identical seeds can differ in the third decimal place. Full
determinism would require `torch.use_deterministic_algorithms(True)` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, at a meaningful speed cost. Treat small
differences between runs as noise, not as signal about a hyperparameter change.

### Keeping constants in sync

Cell 6 duplicates the hyperparameters from `src/config.py` (`MAX_LENGTH`,
`LEARNING_RATE`, `NUM_EPOCHS`, `BATCH_SIZE`, `RANDOM_SEED`, `MODEL_CHECKPOINT`).
This duplication is deliberate — Colab cannot import `src.config` — but it can
drift. If you change either file, change both.
