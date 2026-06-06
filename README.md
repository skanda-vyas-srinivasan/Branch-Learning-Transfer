# Branch Learning Transfer

This repository is a clean, reproducible runner for testing whether a Maudet-style MILP structural distance is related to transfer behavior of a learned branching policy.

The included model is a Learn2Branch-style GNN trained on set-cover instances with 500 rows, 1000 columns, and density 0.05. The experiment compares vanilla SCIP branching with the fixed learned policy on unseen set-cover, combinatorial auction, facility-location, and independent-set instances.

## What Is Included

```text
configs/                  Experiment settings
models/baseline_setcover/  Trained set-cover policy weights and log
scripts/                  Instance generation, evaluation, plotting, summaries
src/branching_eval/        Learn2Branch policy loader and SCIP evaluation
src/milp_distance/         Maudet-style greedy distance implementation
results/.gitkeep           Output directory placeholder; generated results are ignored
data/.gitkeep              Data placeholder; generated instances are ignored
```

Generated MILP instances and results are intentionally not committed. Run the scripts below to recreate them.

## Environment

```bash
conda env create -f environment.yml
conda activate ecole
```

Expected core versions are Ecole 0.8.x, PySCIPOpt, PyTorch, Torch Geometric, NumPy, Pandas, SciPy, and Matplotlib.

## Generate Instances

The generator uses the public Learn2Branch instance generator. Clone it once:

```bash
mkdir -p external
git clone https://github.com/ds4dm/learn2branch.git external/learn2branch
```

Then generate the experiment instances:

```bash
python scripts/generate_instances.py --config configs/setcover_final_100.yaml
```

This creates:

```text
data/instances/reference_setcover
data/instances/eval_setcover
data/instances/eval_cauctions
data/instances/eval_facilities
data/instances/eval_indset
```

## Run Evaluation

```bash
python scripts/run_experiment.py --config configs/setcover_final_100.yaml
```

Results are written to `results/final_100/raw_results.csv`

## Make Plots And Tables

```bash
python scripts/make_plots.py --config configs/setcover_final_100.yaml
python scripts/summarize_results.py --config configs/setcover_final_100.yaml
```

The plotting script writes CSV summaries and PNG figures into the configured results directory. These outputs are ignored by Git so the repository stays clean.

## Distance Implementation Note

`src/milp_distance/distance.py` implements a Maudet-style greedy structural distance. It follows the normalized representation and greedy matching construction described by Maudet and Danoy, but it is not an exact optimal-transport solver. The distance from a test instance to the training distribution is computed as the mean pairwise greedy distance to a reference set of set-cover instances.
