import torch
import time
import json
import os
from datetime import datetime
from pipeline.train import Trainer

def verify_gpu():
    """Verify GPU availability and print device information"""
    if torch.cuda.is_available():
        print(f"CUDA is available!")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return True
    else:
        print("CUDA is not available. Running on CPU.")
        return False

def run_all(num_phases=20, classes_per_phase=5):
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Create directories for each label mode
    # label_modes = ["random_labels"]
    label_modes = ["raw", "one_hot", "multi_hot", "one_hot_self_concat", "random_labels"]
    for mode in label_modes:
        os.makedirs(f'results/{mode}', exist_ok=True)
    
    # Start timing
    total_start_time = time.time()
    
    # Verify GPU and clear cache
    has_gpu = verify_gpu()
    if has_gpu:
        torch.cuda.empty_cache()
    
    device = torch.device("cuda" if has_gpu else "cpu")
    
    # Define all label modes to test
    # Dictionary to store timing and metrics
    results = {
        "run_info": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "gpu": has_gpu,
            "num_phases": num_phases,
            "classes_per_phase": classes_per_phase
        },
        "label_modes": {}
    }
    
    for mode in label_modes:
        print(f"\n=== Running experiments with label mode: {mode} ===")
        
        # Time each label mode
        mode_start_time = time.time()
        
        trainer = Trainer(device, label_mode=mode)
        trainer.run_experiment(num_phases=num_phases, classes_per_phase=classes_per_phase, plot_at_end=False)
        
        # Record time for this mode
        mode_time_taken = time.time() - mode_start_time
        print(f"Time taken for {mode}: {mode_time_taken:.2f} seconds ({mode_time_taken/60:.2f} minutes)")
        
        # Store metrics for this mode
        results["label_modes"][mode] = {
            "time_taken": mode_time_taken,
            "accuracy": trainer.tracker.metrics["incremental"]["accuracy"],
            "num_classes_history": trainer.tracker.num_classes_history[mode],
            "dead_neurons": {},
            "effective_rank": {}
        }
        
        # Add detailed layer metrics
        for layer_name in trainer.tracker.tracked_layers:
            results["label_modes"][mode]["dead_neurons"][layer_name] = trainer.tracker.metrics["incremental"]["dead_neurons"][layer_name]
            results["label_modes"][mode]["effective_rank"][layer_name] = trainer.tracker.metrics["incremental"]["effective_rank"][layer_name]
        
        # Clear GPU cache between experiments
        if has_gpu:
            torch.cuda.empty_cache()
    
    # Generate all plots at the end
    print("\nGenerating all plots at the end...")
    plotting_start = time.time()
    if label_modes:
        # Use the last trainer's tracker to generate plots
        trainer.tracker.plot_results()
    plotting_time = time.time() - plotting_start
    print(f"Plotting completed in {plotting_time:.2f} seconds")
    
    # Total time taken
    total_time_taken = time.time() - total_start_time
    print(f"\nTotal time taken: {total_time_taken:.2f} seconds ({total_time_taken/60:.2f} minutes)")
    
    # Add total time to results
    results["run_info"]["total_time"] = total_time_taken
    
    # Save metrics to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = f"results/metrics_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Metrics saved to {metrics_file}")
    
    # Also save all metrics using the ExperimentTracker's method
    # This provides a more detailed record of metrics
    if label_modes:
        # Use the last trainer's tracker to save all metrics
        trainer.tracker.save_all_metrics()

if __name__ == "__main__":
    run_all(num_phases=10, classes_per_phase=10)



