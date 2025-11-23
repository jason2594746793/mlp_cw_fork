# Task 2.2: Regularization Hyperparameter Search (Q7, Q8) - Testing Dropout, L1/L2, Label Smoothing
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
from mlp.penalties import L1Penalty, L2Penalty

def train_model(config, num_epochs=100, stats_interval=1, seed=111020):
    rng = np.random.RandomState(seed)
    batch_size = 100
    
    train_data = EMNISTDataProvider('train', batch_size=batch_size, rng=rng)
    valid_data = EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)
    
    learning_rate = 0.001
    input_dim, output_dim, hidden_dim = 784, 47, 128

    weights_init = GlorotUniformInit(rng=rng)
    biases_init = ConstantInit(0.)

    # Config extraction
    dropout_prob = config.get('dropout_prob', None)
    l1_coeff = config.get('l1_coeff', None)
    l2_coeff = config.get('l2_coeff', None)
    ls_epsilon = config.get('ls_epsilon', 0.0)
    
    # Weights penalty
    weights_penalty = None
    if l1_coeff:
        weights_penalty = L1Penalty(l1_coeff)
    elif l2_coeff:
        weights_penalty = L2Penalty(l2_coeff)

    layers = []
    # Layer 1
    layers.append(AffineLayer(input_dim, hidden_dim, weights_init, biases_init, weights_penalty=weights_penalty))
    layers.append(ReluLayer())
    if dropout_prob:
        layers.append(DropoutLayer(rng=rng, incl_prob=dropout_prob))
        
    # Layer 2
    layers.append(AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init, weights_penalty=weights_penalty))
    layers.append(ReluLayer())
    if dropout_prob:
        layers.append(DropoutLayer(rng=rng, incl_prob=dropout_prob))
        
    # Output
    layers.append(AffineLayer(hidden_dim, output_dim, weights_init, biases_init, weights_penalty=weights_penalty))
    
    model = MultipleLayerModel(layers)

    error = CrossEntropySoftmaxError(label_smoothing=ls_epsilon)
    learning_rule = AdamLearningRule(learning_rate=learning_rate)
    
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, data_monitors, notebook=False)

    print(f"Training config={config}...")
    stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)
    
    final_val_acc = stats[-1, keys['acc(valid)']] * 100
    final_train_err = stats[-1, keys['error(train)']]
    final_val_err = stats[-1, keys['error(valid)']]
    
    return final_val_acc, final_train_err, final_val_err

def plot_results(existing_data, new_results):
    # Prepare data for plotting
    # Structure: {'Dropout': {prob: (acc, train_err, val_err)}, 'L1': ..., 'L2': ...}
    
    data = existing_data.copy()
    
    # Merge new results
    # config key in new_results corresponds to what we ran
    # We need to map it back
    pass # Logic inside main

if __name__ == '__main__':
    os.environ['MLP_DATA_DIR'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

    # Existing data from Table 3
    # Format: (Val Acc %, Train Err, Val Err)
    existing_dropout = {
        0.6: (80.7, 0.549, 0.593),
        0.85: (85.1, 0.329, 0.434),
        0.97: (85.4, 0.244, 0.457)
    }
    existing_l1 = {
        5e-4: (79.5, 0.642, 0.658),
        5e-3: (2.41, 3.850, 3.850),
        5e-2: (2.20, 3.850, 3.850)
    }
    existing_l2 = {
        5e-4: (85.1, 0.306, 0.460),
        5e-3: (81.3, 0.586, 0.607),
        5e-2: (39.2, 2.258, 2.256)
    }
    
    # Experiments to run
    configs = [
        {'name': 'Dropout 0.7', 'config': {'dropout_prob': 0.7}, 'type': 'Dropout', 'val': 0.7},
        {'name': 'L1 1e-3', 'config': {'l1_coeff': 1e-3}, 'type': 'L1', 'val': 1e-3},
        {'name': 'L2 1e-3', 'config': {'l2_coeff': 1e-3}, 'type': 'L2', 'val': 1e-3},
        {'name': 'Label Smoothing 0.1', 'config': {'ls_epsilon': 0.1}, 'type': 'LS', 'val': 0.1}
    ]
    
    results = {}
    
    print("Running missing experiments...")
    for exp in configs:
        acc, train_err, val_err = train_model(exp['config'], num_epochs=100)
        print(f"Result for {exp['name']}: Acc={acc:.2f}, TrainErr={train_err:.3f}, ValErr={val_err:.3f}")
        results[exp['name']] = (acc, train_err, val_err)
        
        # Update existing dicts
        if exp['type'] == 'Dropout':
            existing_dropout[exp['val']] = (acc, train_err, val_err)
        elif exp['type'] == 'L1':
            existing_l1[exp['val']] = (acc, train_err, val_err)
        elif exp['type'] == 'L2':
            existing_l2[exp['val']] = (acc, train_err, val_err)
            
    # Plotting Figure 4
    # Fig 4a: Accuracy and error by inclusion probability (Dropout)
    # Fig 4b: Accuracy and error by weight penalty (L1/L2)
    
    # Sort keys
    dropout_probs = sorted(existing_dropout.keys())
    l1_coeffs = sorted(existing_l1.keys())
    l2_coeffs = sorted(existing_l2.keys())
    
    # Plot 4a
    fig_dropout = plt.figure(figsize=(10, 6))
    ax_dp = fig_dropout.add_subplot(111)
    
    accs = [existing_dropout[p][0] for p in dropout_probs]
    # errs = [existing_dropout[p][2] for p in dropout_probs] # Which error? "Accuracy and error". Probably Val Error.
    # Wait, Figure 4 caption: "Validation Accuracy and Generalisation Gap"
    # But subfigure captions say: "Accuracy and error by inclusion probability."
    # I will plot Val Acc and Generalisation Gap (Val Err - Train Err).
    
    gen_gaps = [existing_dropout[p][2] - existing_dropout[p][1] for p in dropout_probs]
    
    ax_dp.plot(dropout_probs, accs, 'b-o', label='Val Accuracy (%)')
    ax_dp.set_xlabel('Inclusion Probability')
    ax_dp.set_ylabel('Accuracy (%)', color='b')
    ax_dp.tick_params('y', colors='b')
    
    ax_dp2 = ax_dp.twinx()
    ax_dp2.plot(dropout_probs, gen_gaps, 'r--o', label='Generalisation Gap')
    ax_dp2.set_ylabel('Generalisation Gap (Val Err - Train Err)', color='r')
    ax_dp2.tick_params('y', colors='r')
    
    plt.title('Dropout: Accuracy and Generalisation Gap')
    fig_dropout.tight_layout()
    
    if not os.path.exists('../report/figures'):
        os.makedirs('../report/figures')
    fig_dropout.savefig('../report/figures/dropout_plot.png')
    
    # Plot 4b: L1 and L2
    fig_wd = plt.figure(figsize=(10, 6))
    ax_wd = fig_wd.add_subplot(111)
    
    # L1
    l1_accs = [existing_l1[c][0] for c in l1_coeffs]
    l1_gaps = [existing_l1[c][2] - existing_l1[c][1] for c in l1_coeffs]
    
    # L2
    l2_accs = [existing_l2[c][0] for c in l2_coeffs]
    l2_gaps = [existing_l2[c][2] - existing_l2[c][1] for c in l2_coeffs]
    
    ax_wd.semilogx(l1_coeffs, l1_accs, 'b-o', label='L1 Val Acc')
    ax_wd.semilogx(l2_coeffs, l2_accs, 'g-o', label='L2 Val Acc')
    ax_wd.set_xlabel('Regularisation Coefficient (log scale)')
    ax_wd.set_ylabel('Accuracy (%)')
    
    ax_wd2 = ax_wd.twinx()
    ax_wd2.semilogx(l1_coeffs, l1_gaps, 'b--x', label='L1 Gen Gap')
    ax_wd2.semilogx(l2_coeffs, l2_gaps, 'g--x', label='L2 Gen Gap')
    ax_wd2.set_ylabel('Generalisation Gap')
    
    lines, labels = ax_wd.get_legend_handles_labels()
    lines2, labels2 = ax_wd2.get_legend_handles_labels()
    ax_wd.legend(lines + lines2, labels + labels2, loc='best')
    
    plt.title('Weight Decay: Accuracy and Generalisation Gap')
    fig_wd.tight_layout()
    fig_wd.savefig('../report/figures/wd_plot.png')
    
    print("Figures saved.")
