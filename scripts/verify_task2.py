# Task 2.1: Verification Script for Dropout and Penalties (Q6 Correctness Check)
import os
import sys
import numpy as np

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set data dir
os.environ['MLP_DATA_DIR'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

from mlp.test_methods import test_dropout_layer, test_L1_Penalty, test_L2_Penalty

def run_tests():
    print("Running Dropout and Penalty Tests based on notebooks/DropoutandPenalty_tests.ipynb...")
    
    # Test Dropout
    try:
        fprop_test, fprop_output, fprop_correct, \
        bprop_test, bprop_output, bprop_correct = test_dropout_layer()

        if fprop_test == 1.0:
            print("✅ Dropout Layer Fprop Functionality Test Passed")
        else:
            print("❌ Dropout Layer Fprop Functionality Test Failed")
            # print(f"Correct: {fprop_correct}")
            # print(f"Output: {fprop_output}")
            print(f"Max Diff: {np.max(np.abs(fprop_output-fprop_correct))}")
            print(f"Output Shape: {fprop_output.shape}")
            print(f"Correct Shape: {fprop_correct.shape}")

        if bprop_test == 1.0:
            print("✅ Dropout Layer Bprop Test Passed")
        else:
            print("❌ Dropout Layer Bprop Test Failed")
            print(f"Max Diff: {np.max(np.abs(bprop_output-bprop_correct))}")
            print(f"Output (Mask) Mean: {np.mean(bprop_output)}")
            print(f"Correct (Mask) Mean: {np.mean(bprop_correct)}")
            
            # Check first sample match
            diff_0 = np.max(np.abs(bprop_output[0] - bprop_correct[0]))
            print(f"Max Diff at Batch[0]: {diff_0}")
            
            # Check if correct mask is shared
            # Compare Batch[0] and Batch[1]
            diff_share = np.max(np.abs(bprop_correct[0] - bprop_correct[1]))
            print(f"Correct Mask Diff between Batch[0] and Batch[1]: {diff_share}")


            
    except Exception as e:
        print(f"❌ Dropout Layer Test Error: {e}")

    # Test L1 Penalty
    try:
        call_test, call_output, call_correct, \
        grad_test, grad_output, grad_correct = test_L1_Penalty()

        if call_test == 1.0:
            print("✅ L1 Penalty Call Functionality Test Passed")
        else:
            print("❌ L1 Penalty Call Functionality Test Failed")
            print(f"Diff: {np.abs(call_output-call_correct)}")

        if grad_test == 1.0:
            print("✅ L1 Penalty Grad Function Test Passed")
        else:
            print("❌ L1 Penalty Grad Function Test Failed")
            # print(f"Correct: {grad_correct}")
            # print(f"Output: {grad_output}")
            print(f"Max Diff: {np.max(np.abs(grad_output-grad_correct))}")

    except Exception as e:
        print(f"❌ L1 Penalty Test Error: {e}")

    # Test L2 Penalty
    try:
        call_test, call_output, call_correct, \
        grad_test, grad_output, grad_correct = test_L2_Penalty()

        if call_test == 1.0:
            print("✅ L2 Penalty Call Functionality Test Passed")
        else:
            print("❌ L2 Penalty Call Functionality Test Failed")
            print(f"Diff: {np.abs(call_output-call_correct)}")

        if grad_test == 1.0:
            print("✅ L2 Penalty Grad Function Test Passed")
        else:
            print("❌ L2 Penalty Grad Function Test Failed")
            print(f"Max Diff: {np.max(np.abs(grad_output-grad_correct))}")
            
    except Exception as e:
        print(f"❌ L2 Penalty Test Error: {e}")

if __name__ == '__main__':
    run_tests()
