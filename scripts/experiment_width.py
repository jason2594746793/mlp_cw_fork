import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mlp.data_providers import EMNISTDataProvider
from mlp.layers import AffineLayer, ReluLayer, SoftmaxLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import GlorotUniformInit, ConstantInit
from mlp.learning_rules import AdamLearningRule
from mlp.optimisers import Optimiser

def train_model_and_get_stats(hidden_dim, num_epochs=100, stats_interval=1, seed=111020):
    rng = np.random.RandomState(seed)
    batch_size = 100
    
    # Data providers
    train_data = EMNISTDataProvider('train', batch_size=batch_size, rng=rng)
    valid_data = EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)
    
    # Hyperparameters
    learning_rate = 0.001
    input_dim, output_dim = 784, 47

    weights_init = GlorotUniformInit(rng=rng)
    biases_init = ConstantInit(0.)

    # Create model with TWO hidden layers
    model = MultipleLayerModel([
        AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
        ReluLayer(),
        AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
        ReluLayer(),
        AffineLayer(hidden_dim, output_dim, weights_init, biases_init) 
    ])

    error = CrossEntropySoftmaxError()
    learning_rule = AdamLearningRule(learning_rate=learning_rate)
    
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, data_monitors, notebook=False)

    print(f"Training width={hidden_dim}...")
    stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)
    return stats, keys

def plot_and_save_results(results, widths):
    # Plot Accuracy
    fig_acc = plt.figure(figsize=(10, 6))
    ax_acc = fig_acc.add_subplot(111)
    
    # Plot Error
    fig_err = plt.figure(figsize=(10, 6))
    ax_err = fig_err.add_subplot(111)

    colors = ['r', 'g', 'b']
    
    stats_interval = 1 # Assuming 1 for now

    print("\nResults Table (Width | Val Acc | Train Error | Val Error):")
    print(f"{'Width':<10} | {'Val Acc':<10} | {'Train Error':<12} | {'Val Error':<10}")
    print("-" * 50)

    for i, width in enumerate(widths):
        stats, keys = results[width]
        epochs = np.arange(1, stats.shape[0]) * stats_interval
        
        # Accuracy
        ax_acc.plot(epochs, stats[1:, keys['acc(train)']], linestyle='--', color=colors[i], label=f'Train (w={width})')
        ax_acc.plot(epochs, stats[1:, keys['acc(valid)']], linestyle='-', color=colors[i], label=f'Valid (w={width})')
        
        # Error
        ax_err.plot(epochs, stats[1:, keys['error(train)']], linestyle='--', color=colors[i], label=f'Train (w={width})')
        ax_err.plot(epochs, stats[1:, keys['error(valid)']], linestyle='-', color=colors[i], label=f'Valid (w={width})')
        
        # Final stats
        final_val_acc = stats[-1, keys['acc(valid)']] * 100
        final_train_err = stats[-1, keys['error(train)']]
        final_val_err = stats[-1, keys['error(valid)']]
        
        print(f"{width:<10} | {final_val_acc:<10.2f} | {final_train_err:<12.3f} | {final_val_err:<10.3f}")

    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.legend()
    ax_acc.set_title('Accuracy vs Epoch for different widths')
    
    ax_err.set_xlabel('Epoch')
    ax_err.set_ylabel('Error')
    ax_err.legend()
    ax_err.set_title('Error vs Epoch for different widths')

    # Ensure directory exists
    if not os.path.exists('../report/figures'):
        os.makedirs('../report/figures')

    fig_acc.savefig('../report/figures/width_acccurves.png')
    fig_err.savefig('../report/figures/width_errorcurves.png')
    print("Figures saved to ../report/figures/")

if __name__ == '__main__':
    # Ensure data dir is set
    os.environ['MLP_DATA_DIR'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    
    widths = [32, 64, 128]
    results = {}
    
    for width in widths:
        results[width] = train_model_and_get_stats(width, num_epochs=100)
        
    plot_and_save_results(results, widths)
