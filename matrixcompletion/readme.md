This repository contains code to reproduce our critical learning periods for matrix completion in deep linear networks.

We used Python 3.7.

# Training

To train a depth 3 network, run the following:

```python main.py --depth 3 -c configs/variable-depth.yaml```

Singular values are plotted during training, and by default saved in a folder specified by the `output_dir` in the
configuration file, as well as other relevant log files.

In our paper we varied the depth, varied the number of training examples, initialization scale, and rank of the matrices. All
our experiments can be
found in the `scripts` folder.

# Experiments

To run all the experiments in the main paper, run the following:

```bash scripts/run-all-exps.sh```

To run the supplemental experiments, run the scripts in the `scripts/supplemental` folder.

The generation of the correspondence between the gradient descent solution and solution obtained by stepping through the
differntial
equation can be obtained py running the following:

```python analytical_singular_vectors_and_sim.py```

# Plotting

To plot the aggregate plots in the paper which combine output from multiple runs, run the scripts beginning with `plot_*`

## Acknowledgements
https://github.com/roosephu/deep_matrix_factorization