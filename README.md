# EmoTrack

Clinical-semantic modeling for text-based depression severity estimation.
EmoTrack combines 23 clinical features with frozen utterance embeddings to predict
PHQ-8 symptom scores, using the preceding session when available.

> **[EmoTrack: Clinical-Semantic Modeling for Text-Based Depression Severity Estimation](https://arxiv.org/abs/2605.22286)**
>
> Zhaomin Wu, Jiayi Li, and Bingsheng He

## Overview

This repository includes the model, conversation generation pipeline, selected
training configurations, and evaluation code. The main reproduction workflow is:

**Released conversations → clinical features and embeddings → EmoTrack training → test metrics.**

| Command name | Paper dataset | Training config |
| --- | --- | --- |
| `counselarc` | CounselArc (Qwen-generated) | [emotrack.yaml](configs/train/emotrack.yaml) |
| `luna` | LongCounsel-8-Luna | [emotrack_luna.yaml](configs/train/emotrack_luna.yaml) |
| `mini` | LongCounsel-8-GPT54-Mini | [emotrack_mini.yaml](configs/train/emotrack_mini.yaml) |
| `daic` | DAIC-WOZ | [emotrack_daic.yaml](configs/train/emotrack_daic.yaml) |
| `edaic` | DAIC-WOZ → E-DAIC frozen transfer | Reuses the DAIC-WOZ checkpoint |
| `mini-fn` | Optional false-negative penalty experiment | [emotrack_mini_fn.yaml](configs/train/emotrack_mini_fn.yaml) |

## Installation

Use Linux, Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and an NVIDIA GPU with a compatible CUDA driver. The default feature extractor is
`Qwen/Qwen3.5-35B-A3B`; the embedding model is `Qwen/Qwen3-Embedding-8B`.
An 80 GB GPU accommodates the default local setup. Model weights and CUDA
packages require additional disk space beyond this repository.

```bash
git clone https://github.com/hiddensev/EmoTrack.git
cd EmoTrack
uv sync --locked
```

The lockfile fixes the Python dependencies. The main experiments use the released
data directly; the PsychEval submodule is only needed for optional data generation.

## Quick start

1. Request access to the [released datasets](https://huggingface.co/datasets/hiddensev/LongCounsel-8)
   and accept their terms. Once approved, log in:

   ```bash
   uv run hf auth login
   ```

2. Run one CounselArc seed:

   ```bash
   bash main.sh run-main counselarc --seeds 0
   ```

This downloads the core conversations and frozen split, builds embeddings,
extracts clinical features, trains EmoTrack, and evaluates the selected checkpoint.
Models download automatically. The feature extraction step starts and stops a
local vLLM server as needed.

The full paper protocol uses five runs:

```bash
bash main.sh run-main counselarc
bash main.sh run-main luna
bash main.sh run-main mini
```

Each dataset has separate features, embeddings, checkpoints, and metrics. Downloaded
data are pinned to a Hugging Face revision and checked against SHA-256 hashes.
Existing downloads are checked for their split checksum and required core files.
Feature caches are reused; missing cache entries stop training rather than removing samples. Running the training stage
again starts a new training run and replaces that run's checkpoint.

## Run individual stages

```bash
# Inspect all resolved settings without downloading or running anything.
uv run emotrack-reproduce counselarc --seeds 0 --dry-run

# Download only.
uv run emotrack-reproduce mini --stage download

# Prepare the training data and frozen embeddings.
uv run emotrack-reproduce mini --stage prepare

# Extract clinical features, train, or evaluate independently.
uv run emotrack-reproduce mini --stage features
uv run emotrack-reproduce mini --stage train
uv run emotrack-reproduce mini --stage evaluate

# Recompute test predictions instead of reusing saved detector outputs.
uv run emotrack-reproduce mini --stage evaluate --overwrite
```

The default seeds are `0 1 2 3 4`. The original run labelled `seed0` used training
seed 42 for CounselArc, Mini, and DAIC; Luna used 0. Later runs use their run index.
The configs retain the original deterministic-training settings: enabled for Luna,
Mini, and DAIC, and disabled for CounselArc. Exact numerical agreement also depends
on the feature-extraction and GPU environment.

The optimizer uses 5% linear warmup followed by **linear decay**, matching the
original training code. DAIC uses at most 80 epochs with patience 12; the three
longitudinal datasets use at most 50 epochs with patience 8. Checkpoints are selected
by validation loss. `mini-fn` uses the validation-selected penalty weight 2.5;
all main configurations use zero penalty.

### Outputs

| Path | Contents |
| --- | --- |
| `result/<dataset>/` | Downloaded conversations, ground truth, and frozen split |
| `out/datasets/` | Training JSONL files |
| `out/embed_cache/<dataset>/` | Frozen utterance embeddings |
| `out/eval/conversation/<dataset>_features/aida/seedN/feature_cache/` | Clinical features |
| `out/emotrack*/seedN/best/` | Selected checkpoint and model configuration |
| `out/eval/conversation/<dataset>/emotrack/seedN/metrics_test.json` | Longitudinal test metrics |
| `out/eval/{daic_woz,edaic}/<dataset>/emotrack/seedN/metrics.json` | Real-interview test metrics |
| `result/psyche_eval/emotrack*/seedN/metrics.jsonl` | Training and validation history |
| `out/reproduce/<dataset>/seedN/` | Resolved run configurations |

To collect evaluation metrics:

```bash
uv run python scripts/emotrack.py summarize-metrics --eval-root out/eval --output-root result/metrics
```

## External datasets: request access and supply local files

Automatic dataset downloads are limited to CounselArc and the two LongCounsel-8
variants. Obtain other datasets yourself under each provider's terms. The data
preparation commands only read local files; they do not submit access requests,
use stored provider credentials, or download clinical interviews.

For **DAIC-WOZ and E-DAIC**, use the corresponding application buttons on the
[official access page](https://dcapswoz.ict.usc.edu/). After approval, download the
files from the provider and extract the transcripts into this layout relative to
the repository root:

```text
data/
├── daic_woz/
│   ├── labels/
│   │   ├── train_split_Depression_AVEC2017.csv
│   │   ├── dev_split_Depression_AVEC2017.csv
│   │   ├── test_split_Depression_AVEC2017.csv
│   │   └── full_test_split.csv
│   └── transcripts/
│       ├── 300_TRANSCRIPT.csv
│       └── ...
└── edaic/
    ├── labels/
    │   └── detailed_lables.csv
    └── transcripts/
        ├── 600_Transcript.csv
        └── ...
```

Keep the provider's filename spelling and capitalization, including
`detailed_lables.csv`. DAIC-WOZ transcript CSVs are tab-separated; E-DAIC transcripts
are comma-separated. Supply all subjects in the official splits; the E-DAIC
adapter selects the 86 fully autonomous participants from the provided labels.

```bash
# Convert local inputs, then build features, train, and evaluate.
bash main.sh run-main daic

# Run after the corresponding DAIC seeds have finished training.
bash main.sh run-main edaic
```

The E-DAIC run uses the DAIC-trained checkpoint and treats all 86 subjects as an
external test set. Its local input paths are specified in
[edaic_emotrack.yaml](configs/evaluate/edaic_emotrack.yaml); the checkpoint path is
derived from the DAIC training config. To run only local data preparation and
embedding extraction, add `--stage prepare`.

Previously prepared DAIC data are also supported: `data/daic_woz/labels.csv` plus
`data/daic_woz/{train,dev,test}/<participant_id>.txt`, following `LABEL_FIELD_ORDER`
in [daic_woz.py](src/EmoTrack/data/daic_woz.py). Raw inputs are preserved. Conversion
intermediates are temporary, and prepared bundles are written under `result/`.

## Baselines and optional generation

The existing baseline entry point runs AIDA, LMIQ, EnsemBERT, and the two author-code
baselines. For example, run AIDA and LMIQ on the released CounselArc split:

```bash
for seed in 0 1 2 3 4; do
  uv run python scripts/emotrack.py eval \
    --dataset conversation \
    --input-root result/longcounsel-8-qwen-3.5-35b-a3b \
    --experiment-tag counselarc_baselines \
    --methods aida lmiq --seed "$seed" --gpus 0
done
```

See `uv run python scripts/emotrack.py eval --help` and
[configs/evaluate/default.yaml](configs/evaluate/default.yaml) for other baseline
settings. The commands above reproduce the selected EmoTrack models; baseline
variants and ablations must be run with their corresponding settings.

Optional generation also requires two locally supplied public data files. Obtain
them from the original sources under their published terms:

| Source | File provided by source | Local path |
| --- | --- | --- |
| [PSYCHE-D](https://zenodo.org/records/5085146) | `anon_processed_df_parquet` | `data/psyche/raw/data_quarterly.parquet` |
| [NHANES](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DPQ_L.htm) | `DPQ_L.xpt` | `data/DPQ_L.xpt` |

These public files have their own terms; the clinical dataset application does not
apply to them. Once the files are present, generate new CounselArc-style conversations:

```bash
git submodule update --init --recursive
uv run emotrack-psyche-dataset
uv run python scripts/emotrack.py generate --config configs/cohort_longcounsel8.yaml
```

Generation is optional for the main reproduction workflow. Use the released
conversations and frozen splits for comparisons with the paper.

## Configuration and troubleshooting

Training YAMLs are the source of truth for the dataset JSONL, split, feature cache,
embedding cache, and checkpoint paths. The reproduction entry point uses these same
paths for preparation and evaluation. Keep the split manifest in the input dataset
root, as in the supplied presets.

- **GPU selection:** `AIDA_GPUS=1 bash main.sh run-main counselarc --stage features`
  selects the feature-extraction GPU. Use `CUDA_VISIBLE_DEVICES` for embedding and
  training stages. Multiple GPUs for one vLLM worker can be specified as `AIDA_GPUS=0,1`;
  set `tensor_parallel_size` accordingly in `configs/evaluate/default.yaml`.
- **An existing model server:** set `AIDA_BASE_URL=http://localhost:8100/v1`.
  It must serve the configured Qwen model. Using a different extractor changes the experiment.
- **Hugging Face access denied:** check approval for the dataset and `uv run hf auth whoami`.
- **Missing external data:** follow the provider links above and place authorized files
  in the documented directories. Preparation never downloads them.
- **Incomplete prepared data:** rerun `--stage prepare` after restoring the local source
  files. Keep the released synthetic split; the command never generates a replacement split.
- **Missing feature cache:** run `--stage prepare` for embeddings and `--stage features`
  for clinical features. Training reports missing entries before fitting a model.
- **Out of GPU memory:** run stages sequentially, stop other model servers, or configure
  tensor parallelism in `configs/evaluate/default.yaml`.

Optional environment variables are listed in [.env.example](.env.example).
`bash main.sh` loads `.env` if present; `uv run` commands can use `uv run --env-file .env`.

## Tests

```bash
uv run python -m unittest discover -s tests -v
bash -n main.sh
```

The tests cover dataset isolation, frozen-transfer configuration, seed mapping,
archive safety, incomplete-data detection, local-only external preparation,
and the optional false-negative loss. They run on CPU without
private data or model downloads.

## Citation

```bibtex
@misc{wu2026emotrack,
  title={EmoTrack: Clinical-Semantic Modeling for Text-Based Depression Severity Estimation},
  author={Zhaomin Wu and Jiayi Li and Bingsheng He},
  year={2026},
  eprint={2605.22286},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2605.22286}
}
```

## License

The EmoTrack code is licensed under [Apache-2.0](LICENSE). Data, model weights,
and third-party assets retain their own terms. In particular, the included
PsychEval profiles retain [CC BY-NC 4.0](profiles/PsychEval-LICENSE.txt); the
PsychEval submodule, author-code baselines, Hugging Face datasets, and clinical
interview datasets are covered by their respective licenses and access terms.
