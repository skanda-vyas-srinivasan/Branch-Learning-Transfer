# Branch Learning Transfer

Clean runner repo for the branch-learning transfer experiment. It generates MILP instances, evaluates a fixed set-cover-trained Learn2Branch-style policy, computes Maudet-Danoy-style distances to the set-cover reference set, and saves the resulting plots/tables.

For the motivation, figures, and discussion, read the accompanying write-up: [An Exploratory Look at Maudet-Danoy Distances and Learned Branching Transfer](https://skandavyas.com/entries/learned-branching-transfer.html)
## Setup

```bash
conda env create -f environment.yml
conda activate ecole
```

Clone the upstream Learn2Branch generator:

```bash
mkdir -p external
git clone https://github.com/ds4dm/learn2branch.git external/learn2branch
```

## Run

Generate instances:

```bash
python scripts/generate_instances.py --config configs/setcover_final_100.yaml
```

Evaluate SCIP vs the learned branching policy:

```bash
python scripts/run_experiment.py --config configs/setcover_final_100.yaml
```

Make plots and summary CSVs:

```bash
python scripts/make_plots.py --config configs/setcover_final_100.yaml
python scripts/summarize_results.py --config configs/setcover_final_100.yaml
```

## Outputs

Generated data and results are ignored by Git. After running the pipeline, outputs are written to:

```text
data/instances/
results/final_100/
```

Important result files:

```text
results/final_100/raw_results.csv
results/final_100/summary_by_class.csv
results/final_100/correlation_summary.csv
results/final_100/scatter_distance_rnc_log.png
results/final_100/box_rnc_by_class.png
```

## Included Model

The repo already includes the set-cover-trained policy:

```text
models/baseline_setcover/train_params.pkl
models/baseline_setcover/train_log.txt
```

Retraining is not required for this runner.

## Layout

```text
configs/                    Experiment config
models/baseline_setcover/   Fixed trained policy
scripts/                    Generation, evaluation, plotting, summaries
src/branching_eval/         Learned branching evaluation code
src/milp_distance/          Maudet-Danoy-style distance code
data/                       Generated instances, ignored by Git
results/                    Generated outputs, ignored by Git
```
