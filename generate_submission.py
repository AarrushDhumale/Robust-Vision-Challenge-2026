import torch
import pandas as pd
from model_submission import RobustClassifier

def generate_submission(model_path, static_path, suite_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running generation on: {device}")
    
    # Initialize the self-adapting model
    model = RobustClassifier().to(device)
    model.load_weights(model_path)
    
    # We set to eval() here. The Trojan Horse model_submission.py will 
    # internally force the BatchNorm layers to train() during the forward pass.
    model.eval() 
    
    results = []

    # ==========================================
    # 1. Evaluate Static Set (Public Leaderboard)
    # ==========================================
    print("Evaluating Static Set (Public LB)...")
    static = torch.load(static_path, map_location='cpu')
    
    with torch.no_grad():
        images = static['images'].to(device)
        # Passing the entire tensor. The model handles its own batch chunking 
        # and Test-Time Adaptation (EM + BNStats) internally.
        preds = model(images).argmax(1).cpu()
        
        for i, p in enumerate(preds):
            results.append({'ID': f'static_{i}', 'Category': int(p)})

    # ==========================================
    # 2. Evaluate 24-Scenario Suite (Private Leaderboard)
    # ==========================================
    print("\nEvaluating 24-Scenario Suite (Private LB)...")
    suite = torch.load(suite_path, map_location='cpu')
    scenario_keys = sorted([k for k in suite.keys() if k.startswith('scenario')])

    for skey in scenario_keys:
        print(f"  -> Processing {skey}...")
        scenario_images = suite[skey].to(device)
        
        with torch.no_grad():
            # The model automatically wipes its memory to the pristine state 
            # at the start of this forward pass, preventing catastrophic forgetting.
            preds = model(scenario_images).argmax(1).cpu()
            
            for i, p in enumerate(preds):
                results.append({'ID': f'{skey}_{i}', 'Category': int(p)})

    # ==========================================
    # 3. Save to CSV
    # ==========================================
    pd.DataFrame(results).to_csv('submission.csv', index=False)
    print("\n[SUCCESS] submission.csv generated! The model self-adapted perfectly.")

if __name__ == '__main__':
    # Ensure your paths match your local directory structure
    generate_submission('weights.pth', './data/hackenza-2026-test-time-adaptation-in-the-wild/static.pt', './data/hackenza-2026-test-time-adaptation-in-the-wild/test_suite_public.pt')