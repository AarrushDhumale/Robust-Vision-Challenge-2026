import torch
import pandas as pd
from model_submission import RobustClassifier
import time

def generate_submission(model, static_path, suite_path, device):
    results = []
    
    # ==========================================
    # MISSION 1: SURVIVE STATIC.PT (Public LB)
    # ==========================================
    print("\n[Phase A] Deploying to target domain: static.pt")
    start_time = time.time()
    
    static_data = torch.load(static_path)
    # Extract images, shape is [B, 1, 28, 28]
    static_images = static_data['images'].to(device)
    
    with torch.no_grad():
        # The model's forward pass triggers BNStats and EM automatically
        preds = model(static_images).argmax(1).cpu()
        
        # --- HEALTH CHECK ---
        unique, counts = torch.unique(preds, return_counts=True)
        distribution = dict(zip(unique.numpy(), counts.numpy()))
        print(f" -> Predicted Distribution: {distribution}")
        
        for i, p in enumerate(preds):
            results.append({'ID': f'static_{i}', 'Category': int(p)})
            
    print(f" -> static.pt complete in {time.time() - start_time:.2f}s")

    # ==========================================
    # MISSION 2: SURVIVE THE TEST SUITE (Private LB Prep)
    # ==========================================
    print("\n[Phase B] Deploying to Hidden Matrix (24 Scenarios)...")
    suite = torch.load(suite_path)
    scenario_keys = sorted([k for k in suite.keys() if k.startswith('scenario')])

    for skey in scenario_keys:
        print(f" Adapting to {skey}...")
        scenario_images = suite[skey].to(device)
        
        with torch.no_grad():
            # Because of your memory reset in model_submission.py, 
            # the model starts completely fresh for every new scenario.
            preds = model(scenario_images).argmax(1).cpu()
            
            # --- HEALTH CHECK ---
            unique, counts = torch.unique(preds, return_counts=True)
            distribution = dict(zip(unique.numpy(), counts.numpy()))
            # Just print the number of unique classes predicted to ensure no Mode Collapse
            print(f"    Unique Classes Predicted: {len(unique)}/10")
            
            for i, p in enumerate(preds):
                results.append({'ID': f'{skey}_{i}', 'Category': int(p)})

    # ==========================================
    # MISSION 3: COMPILE CSV
    # ==========================================
    print("\n[Phase C] Writing submission.csv...")
    pd.DataFrame(results).to_csv('submission.csv', index=False)
    print("Done! Ready for Kaggle upload.")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Deployment on {device}...")
    
    # Load the base weights from Phase 1 Bootcamp
    model = RobustClassifier()
    model.load_weights('weights.pth')
    model = model.to(device)
    # Crucial: the main model stays in eval, but your forward pass selectively overrides BN layers
    model.eval() 
    
    # Verify these paths match your local directory structure
    static_pt = './data/hackenza-2026-test-time-adaptation-in-the-wild/static.pt'
    suite_pt = './data/hackenza-2026-test-time-adaptation-in-the-wild/test_suite_public.pt'
    
    generate_submission(model, static_pt, suite_pt, device)