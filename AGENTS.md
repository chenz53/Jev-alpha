# AGENTS.md

## Purpose

This repository (`jev-alpha`) builds a System One-like decision model on top of ms-swift.

The model reads one state and many typed questions.
It returns one probability distribution for each question.
It does not write answer text.

Question types:

- `noul`: a yes/no question. The answer is p(yes).
- `choice`: pick one option. The answer is a distribution over the options.
- `score`: pick one ordered level. The answer is a distribution over the levels.

## Core principles

1. **Keep ms-swift intact.**
   - Do not edit the ms-swift source code.
   - Do not copy ms-swift files into this repository.
   - Use the extension points of ms-swift: `--external_plugins`, dataset registration, model and template registration, and `--loss_type`.
   - Install ms-swift from its original repository at the commit pinned in `pyproject.toml`.
   - The user approves commit `c2bcc23a1f1c428dc8be0ae509c31a532922453f` as a development-version exception.
   - An optional local `ms-swift/` checkout must stay ignored. Do not commit a submodule.
   - Do not change the pin without asking the user.
   - If a task needs a change inside ms-swift, stop. Tell the user why. Wait for a decision.

2. **Build on top of the framework.**
   - Put all new code in `jev_alpha/`.
   - Make one small, coherent layer. Each file has one job.
   - Use the names and data formats of ms-swift when possible.

3. **Keep it simple.**
   - Write the shortest code that works.
   - Do not add a feature before it is needed.
   - Prefer plain functions to classes.
   - Do not add a new dependency without asking.

4. **Make it readable.**
   - Use clear names. One function does one thing.
   - Put a short comment above each step that is not obvious. Say why, not what.
   - Keep each file under 200 lines. Split a file that grows.

## Repository layout

```
jev-alpha/
  AGENTS.md
  README.md        # setup and how to run
  DATA.md          # source and license of each dataset
  pyproject.toml   # package jev_alpha; pins upstream ms-swift
  ms-swift/        # optional local checkout; ignored, never edited
  jev_alpha/       # our code: a thin layer on top of ms-swift
    plugin.py      # entry point for --external_plugins
    data.py        # record format, render function, preprocessors
    loss.py        # custom loss
    calibrate.py   # temperature fit
    eval.py        # accuracy, Brier, ECE, option-order test
  configs/         # one shell script for each run
  scripts/         # data build and other one-off tools
  tests/
  data/            # raw and built data (git-ignored)
  runs/            # checkpoints and logs (git-ignored)
  reports/         # local evaluation reports (git-ignored)
  docs/assets/     # requested comparison plot and its source values
```

Setup:

```
git clone https://github.com/chenz53/Jev-alpha.git
cd Jev-alpha
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[qwen]'
```

## Work plan

Do the phases in order. Finish and test one phase before you start the next.

### Phase 1: baseline (no change to ms-swift)

1. Render each record as a chat message. The query holds the state, the question, and the options. Each option has a letter label.
2. Make the response one label token.
3. Register the dataset with `register_dataset` and a preprocessor.
4. Register a loss in `loss_map` and select it with `--loss_type`. The loss reads the logits at the answer position. It keeps only the option label tokens. It applies cross-entropy over them.
5. Shuffle the options in every epoch.
6. After training, fit one temperature on a held-out set. Save it with the checkpoint.

### Phase 2: pointer head and shared state

Ask the user before you start this phase.

- First check if a plugin can do the work.
- ms-swift documents custom loss for pre-training and SFT. It does not document it for `seq_cls`, DPO, or PPO. Check the installed version.
- A block-causal mask and a pointer head can need deeper hooks. If so, stop and ask. Do not patch ms-swift.

### Phase 3: reinforcement learning

Start only after Phase 1 has a baseline ECE.

- Use the GRPO tools of ms-swift with an external reward function in a plugin.
- Take the reward from a cost matrix, for example act, escalate, wrong.
- If the reward is a proper scoring rule (log loss, Brier), use a direct loss. It is simpler and has less variance.

## Data rules

- Use one record format: `state` and `questions`. Each question has `type`, `instructions`, and `criteria`.
- Use one function to render a record. Training and serving must call this function.
- Hold out whole datasets for the test. Do not use only a random split.
- Add hard cases: none of the above, missing evidence, and ambiguous labels.
- Remove test items from the training data. Check exact and near matches.
- Write the license of each dataset in `DATA.md`.

## Evaluation rules

- Report accuracy, Brier score, and ECE (ten equal-width bins).
- Report ECE before and after the temperature fit.
- Run the option-order test: shuffle the options and compare the probabilities.
- Run the isolation test in Phase 2: put a secret in a sibling question and check that it has no effect.
- Do not report only a mean. Report each held-out dataset.

## Commands

- Check the flags with `swift sft --help` before you write a command. Flag names change between versions.
- These flags are confirmed in the ms-swift docs: `--external_plugins`, `--loss_type`, `--dataset`, `--model`.
- Write each run as a script in `configs/`. Do not run long jobs by hand.

## Definition of done

- The tests pass.
- The evaluation report exists.
- The upstream ms-swift pin is unchanged. An optional local checkout shows no change.
- A new reader can run the work from the README.

## Stop and ask

Stop and ask the user when:

- A task needs a change to the ms-swift source.
- A task needs a new dependency.
- A task deletes data or checkpoints.
- Two options exist and their results differ in a major way.

## Response style

Every response must be at least 80% compliant with ASD-STE100 (Simplified Technical English).

- Write short sentences. Procedures: 20 words or fewer. Descriptions: 25 words or fewer.
- Put one instruction in each sentence.
- Use the active voice. Use the simple present tense for facts. Use the imperative for steps.
- Use "must" for a rule. Use "do not" for a ban. Do not use "should", "may", or "could" for a rule.
- Use one word for one meaning. Do not use synonyms. Always say "state", "question", and "option".
- Use simple verbs: use, make, put, keep, check, stop, ask. Do not use "utilize" or "leverage".
- Keep the articles: a, an, the.
- Do not use idioms, jokes, or figures of speech.
- Use a list for steps. Put one step in each item.
- Define a technical term the first time you use it.
- If you are not sure, say so. Do not guess.
