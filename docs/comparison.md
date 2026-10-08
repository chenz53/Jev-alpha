# Comparison with published Jev results

Our values average three unmerged adapter seeds. They are not the published merged checkpoint results.
The questions and prompts differ from Jev's published evaluations. This is not a paired ranking.

| Accuracy | Jev-alpha | Published Jev |
|---|---:|---:|
| BoolQ | 92.00% | 92.50% |
| QNLI | 94.33% | 92.50% |
| PAWS | 94.00% | 78.75% |
| MMLU | 83.00% | 90.00% |
| SciQ | 100.00% | 98.75% |
| HellaSwag | 90.33% | 89.33% |
| CLINC | 94.33% | 97.33% |
| DBpedia | 100.00% | 95.00% |
| SST-5 | 56.00% | 63.75% |
| Yelp | 62.00% | 66.25% |
| When2Call | 66.33% | 76.00% |

Our test uses 100 questions per dataset. Jev sources use Kev commit `fe64b1274ea7f80d4095866df90666abb03e9cf6`:

- [Transfer-v4](https://github.com/jaredpalmer/kev/blob/fe64b1274ea7f80d4095866df90666abb03e9cf6/runs/jev-transfer-v4/report.json): MMLU, SciQ, QNLI, PAWS; 80 questions each.
- [Decision-v4](https://github.com/jaredpalmer/kev/blob/fe64b1274ea7f80d4095866df90666abb03e9cf6/runs/jev-decision-v4/report.json): BoolQ, DBpedia, SST-5, Yelp; 80 questions each.
- [Breadth-v1](https://github.com/jaredpalmer/kev/blob/fe64b1274ea7f80d4095866df90666abb03e9cf6/runs/breadth-v1-jev/report.json): CLINC and HellaSwag; 150 questions each.
- [Devtools-v1](https://github.com/jaredpalmer/kev/blob/fe64b1274ea7f80d4095866df90666abb03e9cf6/runs/jev-devtools-v1/report.json): When2Call; 150 questions.

[Exact values and source hashes](assets/jev-comparison.json) accompany the [radar plot](assets/jev-comparison.svg).
