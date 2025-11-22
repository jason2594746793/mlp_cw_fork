import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mlp.data_providers import EMNISTDataProvider
from mlp.layers import AffineLayer, CustomActivationLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import GlorotUniformInit, ConstantInit
from mlp.learning_rules import AdamLearningRule
from mlp.optimisers import Optimiser

def run_experiment(num_epochs=5, stats_interval=1, seed=111020):
    rng = np.random.RandomState(seed)
    batch_size = 100
    
    train_data = EMNISTDataProvider('train', batch_size=batch_size, rng=rng)
    valid_data = EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)
    
    learning_rate = 0.001
    input_dim, output_dim, hidden_dim = 784, 47, 128

    weights_init = GlorotUniformInit(rng=rng)
    biases_init = ConstantInit(0.)

    # Create model with four hidden layer
    model = MultipleLayerModel([
        AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
        CustomActivationLayer(), 
        AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
        CustomActivationLayer(),
        AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
        CustomActivationLayer(),
        AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init),
        CustomActivationLayer(),
        AffineLayer(hidden_dim, output_dim, weights_init, biases_init) # output layer
    ])

    error = CrossEntropySoftmaxError()
    learning_rule = AdamLearningRule(learning_rate=learning_rate)
    
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, data_monitors, notebook=False)

    print("Running Custom Activation experiment...")
    stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)
    
    # Plot Accuracy
    fig_acc = plt.figure(figsize=(8, 4))
    ax_acc = fig_acc.add_subplot(111)
    ax_acc.plot(np.arange(1, stats.shape[0]) * stats_interval, stats[1:, keys['acc(train)']], label='Train Acc')
    ax_acc.plot(np.arange(1, stats.shape[0]) * stats_interval, stats[1:, keys['acc(valid)']], label='Valid Acc')
    ax_acc.legend()
    ax_acc.set_title('Accuracy vs Epoch (Custom Activation)')
    
    # Plot Error
    fig_err = plt.figure(figsize=(8, 4))
    ax_err = fig_err.add_subplot(111)
    ax_err.plot(np.arange(1, stats.shape[0]) * stats_interval, stats[1:, keys['error(train)']], label='Train Error')
    ax_err.plot(np.arange(1, stats.shape[0]) * stats_interval, stats[1:, keys['error(valid)']], label='Valid Error')
    ax_err.legend()
    ax_err.set_title('Error vs Epoch (Custom Activation)')
    
    # Gradient Flow
    grad_plot, grad_ax = optimiser.plot_grad_flow()
    
    if not os.path.exists('../report/figures'):
        os.makedirs('../report/figures')
        
    fig_acc.savefig('../report/figures/custom_act_acc.png')
    fig_err.savefig('../report/figures/custom_act_error.png')
    grad_plot.savefig('../report/figures/custom_act_grad_flow.png')
    
    print("Figures saved to ../report/figures/")

if __name__ == '__main__':
    os.environ['MLP_DATA_DIR'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    run_experiment(num_epochs=5) # Experiment uses 5 epochs as per instructions
