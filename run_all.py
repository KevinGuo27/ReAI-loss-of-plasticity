import torch
from pipeline.train import Trainer

def run_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define all label modes to test
    label_modes = ["raw", "one_hot", "multi_hot", "one_hot_self_concat", "random_labels"]
    
    for mode in label_modes:
        print(f"\n=== Running experiments with label mode: {mode} ===")
        trainer = Trainer(device, label_mode=mode)
        trainer.run_experiment(num_phases=20, classes_per_phase=5)

if __name__ == "__main__":
    run_all()



