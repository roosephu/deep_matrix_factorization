# Implicit Regularization in Deep Matrix Factorization

Code for [
Implicit Regularization in Deep Matrix Factorization](https://arxiv.org/abs/1905.13655). 

## Installation

Please ues Python 3.7 for running this code. 

```bash
pip install -r requirements.txt
```

## Dataset Generation

Here is the example for generating the inputs for matrix completion with n = 100, rank = 5 and 2k samples. 

```bash
mkdir -p datasets/mat-cmpl
python gen_gt.py --config configs/mat-cmpl/gen_gt.toml
python gen_obs.py --config configs/mat-cmpl/gen_obs.toml --set n_train_samples 2000
```

## Experiments

If you just want to run one experiment, use the following command as an example. 

```bash
python main.py --print_config --log_dir /tmp/exp1 \
    --config configs/mat-cmpl/run.toml \
    --config configs/mat-cmpl/2000.toml \
    --config configs/opt/grouprmsprop.toml \
    --set depth 2 
```

For nuclear norm minimization: 

```bash
python main.py --print_config --log_dir /tmp/exp2 \
    --config configs/mat-cmpl/run.toml \
    --config configs/mat-cmpl/2000.toml \
    --config configs/opt/cvx.toml
```

For dynamics of gradient descent (Figure 3):

```bash
python main.py --log_dir /tmp --print_config \
    --config configs/ml-100k.toml \
    --config configs/opt/SGD.toml \
    --config configs/dynamics.toml \
    --set depth 2
```


The results will be saved at `/tmp/ID`, where `ID` is a different number for each run and startsfrom 0.  

To run multiple experiments sequentially, you can use `./scripts/run.rb` (please make sure Ruby is installed and `gem install colorize --user`). The code will log into `~/logs` by default. 

```bash
./scripts/run.rb --n_jobs 3 --name mat-cmpl \
    --template 'python main.py --print_config --log_dir LOGDIR --config configs/mat-cmpl/run.toml --config configs/mat-cmpl/SAMPLES.toml --config configs/opt/grouprmsprop.toml --set depth DEPTH --set lr LR --set init_scale SCALE' \
    --replace LR=0.001,0.0003 \
    --replace DEPTH=2,3,4 \
    --replace SCALE=1.e-3,1.e-4,1.e-5,1.e-6 \
    --replace SAMPLES=2000,5000
```

For multiple experiments on nuclear norm minimization: 

```bash
./scripts/run.rb --n_jobs 1 --name mat-cmpl-cvx \
    --template 'python main.py --print_config --log_dir LOGDIR --config configs/mat-cmpl/run.toml --config configs/mat-cmpl/SAMPLES.toml --config configs/opt/cvx.toml' \
    --replace SAMPLES=2000,5000
```

# Plotting

We use the Jupyter notebook `plot.ipynb` to generate our figures. 

Please modify 4-th cell to load all results. The directories are the corresponding `--log_dir` option, e.g., `/tmp/exp1` in the first example. 
 