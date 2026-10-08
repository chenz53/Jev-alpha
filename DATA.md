# Data sources

The repository distributes code, 18 small synthetic test cases in source code, and aggregate values for the requested comparison plot.
It does not distribute downloaded datasets, model weights, authored experiment data, or raw reports.

## Runnable example

[google/boolq](https://huggingface.co/datasets/google/boolq) supplies yes/no questions and passages.
Its source card states CC-BY-SA-3.0. Preserve source attribution and check the source terms before redistribution.
The downloader records the resolved revision and source name in `data/boolq/source.json`.
Training uses its train split. Temperature calibration uses its validation split.
The fixed synthetic questions in `jev_alpha/datasets/hard_cases.py` test all three question types.
They include missing evidence, conflicting evidence, and none-of-the-above options.
These 18 questions are implementation checks, not evidence of general task performance.

Preparation keeps whole test datasets outside training and calibration.
It removes training states with exact or near matches to held-out states.
Near matches use normalized five-word groups with Jaccard similarity of at least 0.9.
Jaccard similarity is the shared group count divided by the total distinct group count.
The check does not establish the absence of every semantic duplicate or pretraining overlap.

## Local experiment represented in the plot

The completed local experiment uses 8,400 training questions: 6,720 public questions and 1,680 directly authored questions.
It uses training splits from all public sources below and separate questions for development, calibration, and final evaluation.
The user authorizes this same-source evaluation as an exception to whole-dataset holdout.
The final set contains 100 questions per source. BoolQ controls use unused cleaned training records.
The other sources use held-out validation or test records. These are not unseen-dataset transfer results.
The public When2Call subset has 292 clarification targets and 268 cannot-answer targets.
Authored questions include direct-answer and tool-call targets. The imbalance limits the action-task result.
The new authored text and reviewed action labels do not have independent human review.

| Source | Source terms or attribution |
|---|---|
| [BoolQ](https://huggingface.co/datasets/google/boolq) | CC-BY-SA-3.0 |
| [MMLU](https://huggingface.co/datasets/cais/mmlu) | MIT; Hendrycks et al. |
| [SciQ](https://huggingface.co/datasets/allenai/sciq) | CC-BY-NC-3.0; Welbl et al. |
| [GLUE QNLI](https://huggingface.co/datasets/nyu-mll/glue) | GLUE refers to original source terms; derived from [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) |
| [PAWS](https://huggingface.co/datasets/google-research-datasets/paws) | [Custom Google terms](https://github.com/google-research-datasets/paws/blob/master/LICENSE); acknowledge Google and Zhang et al. |
| [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag) | MIT per Kev's manifest; ActivityNet subset and source terms; Zellers et al. |
| [CLINC](https://huggingface.co/datasets/clinc/clinc_oos) | CC-BY-3.0; Larson et al. |
| [SST-5](https://huggingface.co/datasets/SetFit/sst5) | SetFit card does not state a license; check [Stanford source terms](https://nlp.stanford.edu/sentiment/) |
| [When2Call](https://huggingface.co/datasets/nvidia/When2Call) | CC-BY-4.0; Ross et al.; source BFCL v2 Live is Apache-2.0 |
| [DBpedia-14](https://huggingface.co/datasets/fancyzhx/dbpedia_14) | CC-BY-SA and GFDL, per the source card |
| [Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full) | Yelp dataset agreement linked from the source card |
| [WikiText](https://huggingface.co/datasets/Salesforce/wikitext) | CC-BY-SA-3.0 and GFDL; independent text evaluation only |

Source terms are not a new license grant. Raw data and local manifests remain excluded from Git.
The comparison plot contains only aggregate scores. Its [source-value file](docs/assets/jev-comparison.json) links each Jev task to a pinned Kev report.
The Jev values are third-party measurements by Kev. They use different questions and do not establish a paired ranking.
