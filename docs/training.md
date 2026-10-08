# Training example

Use Python 3.12 and an NVIDIA GPU. The tested environment uses PyTorch 2.8.0, Transformers 5.16.1, and an H200.
The default training example uses **Qwen3.5-0.8B-Base**. It checks the implementation and does not reproduce the full Gemma experiment.

```bash
git clone https://github.com/chenz53/Jev-alpha.git
cd Jev-alpha
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[qwen]'
bash configs/build_data.sh
bash configs/train.sh
```

ms-swift installs from its original repository at commit `c2bcc23a1f1c428dc8be0ae509c31a532922453f` (`4.6.0.dev0`).
There is no framework fork or submodule. The Qwen extra supplies its loader dependencies; the project does not include a full environment lock.
The example uses BoolQ training and calibration data, plus 18 fixed synthetic test questions.
Preparation removes exact and near state matches with held-out data. For a short integration run, use `bash configs/pilot.sh`.

```bash
# Train with the optional reference penalty.
JEV_KL_TOP_K=64 JEV_KL_WEIGHT=0.1 bash configs/train.sh \
  --loss_type jev_options_kl --lora_dropout 0 --output_dir runs/boolq-kl
# Replace the checkpoint placeholder with the path printed by training.
CHECKPOINT=runs/boolq/<run>/checkpoint-<step> bash configs/evaluate.sh
python -m jev_alpha.predict --model Qwen/Qwen3.5-0.8B-Base \
  --adapter runs/boolq/<run>/checkpoint-<step> \
  --input requests.jsonl --output probabilities.jsonl
```

Save input records as JSON Lines: one JSON object per line. Serving loads the adapter's fitted temperature when present; otherwise it uses one.
Keep lazy tokenization, zero data workers, no packing, and full sequence logits for these loss hooks.
Inputs beyond the context limit cause an error. Commands refuse to replace existing data and result files.
The pin's full training flags are available through `python -c 'import sys; from swift.pipelines import sft_main; sys.argv=["sft", "--help"]; sft_main()'`.

## Code and checks

Core modules in `jev_alpha/` handle rendering, templates, losses, prediction, calibration, and evaluation.
`jev_alpha/datasets/` holds the example data workflow. `configs/` contains run scripts; `tests/` covers the core and framework hooks.
Data, checkpoints, reports, caches, and the optional local ms-swift checkout stay outside Git.

```bash
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
```

Plugin tests load the Qwen tokenizer through ms-swift and can download metadata on the first run.
