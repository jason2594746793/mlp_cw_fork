import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mlp.data_providers import EMNISTDataProvider
from mlp.layers import AffineLayer, ReluLayer, SoftmaxLayer, DropoutLayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import GlorotUniformInit, ConstantInit
from mlp.learning_rules import AdamLearningRule
from mlp.optimisers import Optimiser
from mlp.penalties import L1L2MixPenalty

def train_model(l1, l2, num_epochs=100, stats_interval=1, seed=111020):
    rng = np.random.RandomState(seed)
    batch_size = 100
    
    train_data = EMNISTDataProvider('train', batch_size=batch_size, rng=rng)
    valid_data = EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)
    
    learning_rate = 0.001
    input_dim, output_dim, hidden_dim = 784, 47, 128

    weights_init = GlorotUniformInit(rng=rng)
    biases_init = ConstantInit(0.)

    weights_penalty = L1L2MixPenalty(l1, l2)

    layers = []
    layers.append(AffineLayer(input_dim, hidden_dim, weights_init, biases_init, weights_penalty=weights_penalty))
    layers.append(ReluLayer())
    layers.append(AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, weights_penalty=weights_penalty))
    layers.append(ReluLayer())
    layers.append(AffineLayer(hidden_dim, output_dim, weights_init, biases_init, weights_penalty=weights_penalty))
    
    model = MultipleLayerModel(layers)

    error = CrossEntropySoftmaxError()
    learning_rule = AdamLearningRule(learning_rate=learning_rate)
    
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, data_monitors, notebook=False)

    print(f"Training L1={l1}, L2={l2}...")
    stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)
    
    final_val_acc = stats[-1, keys['acc(valid)']] * 100
    final_train_err = stats[-1, keys['error(train)']]
    final_val_err = stats[-1, keys['error(valid)']]
    
    return final_val_acc, final_train_err, final_val_err

if __name__ == '__main__':
    os.environ['MLP_DATA_DIR'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

    configs = [
        (1e-3, 1e-3),
        (1e-3, 5e-4),
        (5e-4, 1e-3),
        (5e-4, 5e-4)
    ]
    
    results = []
    
    print("Results Table (L1 | L2 | Val Acc | Train Error | Val Error):")
    print(f"{'L1':<10} | {'L2':<10} | {'Val Acc':<10} | {'Train Error':<12} | {'Val Error':<10}")
    print("-" * 65)
    
    for l1, l2 in configs:
        acc, train_err, val_err = train_model(l1, l2, num_epochs=100)
        print(f"{l1:<10} | {l2:<10} | {acc:<10.2f} | {train_err:<12.3f} | {val_err:<10.3f}")
        results.append({'l1': l1, 'l2': l2, 'acc': acc, 'train_err': train_err, 'val_err': val_err})
        
    # Plot Figure 5
    # Validation Accuracy and Generalisation Gap for each of your 4 experiments
    
    labels = [f"L1={r['l1']}\nL2={r['l2']}" for r in results]
    accs = [r['acc'] for r in results]
    gaps = [r['val_err'] - r['train_err'] for r in results]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    rects1 = ax1.bar(x - width/2, accs, width, label='Val Acc', color='b', alpha=0.7)
    ax1.set_ylabel('Accuracy (%)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(0, 100)
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, gaps, width, label='Gen Gap', color='r', alpha=0.7)
    ax2.set_ylabel('Generalisation Gap', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title('Combined L1 & L2 Regularization Results')
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    if not os.path.exists('../report/figures'):
        os.makedirs('../report/figures')
    fig.savefig('../report/figures/combined_reg_plot.png')
    print("Figure saved to ../report/figures/combined_reg_plot.png")
