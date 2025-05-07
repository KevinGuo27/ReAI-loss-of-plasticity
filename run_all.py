import torch
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

def run_all():
    # Verify GPU and clear cache
    has_gpu = verify_gpu()
    if has_gpu:
        torch.cuda.empty_cache()
    
    device = torch.device("cuda" if has_gpu else "cpu")
    
    # Define all label modes to test
    label_modes = ["one_hot", "raw", "multi_hot", "one_hot_self_concat", "random_labels"]
    
    for mode in label_modes:
        print(f"\n=== Running experiments with label mode: {mode} ===")
        trainer = Trainer(device, label_mode=mode)
        trainer.run_experiment(num_phases=20, classes_per_phase=5)
        
        # Clear GPU cache between experiments
        if has_gpu:
            torch.cuda.empty_cache()

if __name__ == "__main__":
    run_all()



