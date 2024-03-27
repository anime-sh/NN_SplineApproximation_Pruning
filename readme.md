# Understanding Deep Neural Networks Through Spline Approximation
This code repo performs experiments on NN pruning by looking at them as a composition of maxaffine spline operators (MASOs)

## How to run
- For FC Net exp - use `mlp.py`
    - Specify config file in the code, look at `config.json` and `config_lth.json` as an example
    - Output plots are saved in `plt/mlp/synthetic/*`
    - The program generates 3 class 2D synth dataset and then plots the subdivision lines of vairous models as they train.
        - Ground truth and model predictions are also plotted
- For ConvNet exp - use `cnn.py`
    - Specify config file in the code, see `config_cnn.json`
    - Output plots are saved in `plt/cnn/mnist/*`
    - The program trains a bunch of models independetly on mnist while pruning them with different strategies, subdivision lines are also plotted
        - accuracies across epochs is saved to `cnn.log`
- In both programs models are first pruned (according to defined frequency) then trained then plotted (according to defined frequency) in an epoch