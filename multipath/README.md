# CriticalPeriodLinearMultiPath

## To reproduce our results, please run the following commands:

### Phase portraits (Fig 1 and Fig 7):

```
python phase-large-init.py
python phase-small-init.py
```

### Fig 2:

```
python analytical_and_sgd_final.py -c configs/multipath-depth4-analytical.yaml
```

### Fig 3:

```
python analytical_and_sgd_gateSing.py -c configs/multipath-gateSing-analytical.yaml
```

### Nonlinear Experiments for rebuttal (Fig 11 and Fig 12):

```
python nonlin_sgd_rebuttal_iclr.py -c configs/iclr-rebuttal/multipath-nonlin-tanh-depth3-analytical.yaml
python nonlin_sgd_rebuttal_iclr.py -c configs/iclr-rebuttal/multipath-nonlin-relu-depth3-analytical.yaml
```

## We ran this repository using:

- python 3.10.8
- torch==1.13.1
- numpy==1.24.0
- matplotlib==3.6.3
- scipy==1.10.0

## Acknowledgments:

This repository was based off: https://github.com/AllenInstitute/Multipathway_NeurIPS2022