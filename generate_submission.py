import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from model_submission import RobustClassifier

def setup_tent(model):
    """Forces BN to use target batch stats and enables gradients only on BN parameters."""
    model.train()
    model.requires_grad_(False)
    
    params = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.weight.requires_grad = True
            m.bias.requires_grad = True
            params.extend([m.weight, m.bias])
            
    return optim.Adam(params, lr=1e-3)

def adapt_and_predict(model, images_tensor, device, steps=2):
    """Executes Covariate Alignment (TENT) and Label Shift Estimation (EM)."""
    dataset = TensorDataset(images_tensor)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    
    optimizer = setup_tent(model)
    target_prior = torch.ones(10, device=device) / 10.0 # Initialize uniform prior
    
    for step in range(steps):
        all_preds = []
        for batch in loader:
            imgs = batch[0].to(device)
            logits = model(imgs)
            
            # EM Step: Adjust logits based on estimated target prior
            adjusted_logits = logits + torch.log(target_prior + 1e-6) - torch.log(torch.tensor(0.1, device=device))
            probs = F.softmax(adjusted_logits, dim=1)
            
            # Update running prior (Moving Average)
            batch_prior = probs.mean(dim=0)
            target_prior = 0.9 * target_prior + 0.1 * batch_prior.detach()
            
            # TENT Step: Entropy Minimization
            entropy_loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            optimizer.zero_grad()
            entropy_loss.backward()
            optimizer.step()
            
            if step == steps - 1:
                all_preds.append(probs.argmax(dim=1).cpu())
                
    return torch.cat(all_preds)

def main(model_path, static_path, suite_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    
    # --- PHASE 2: Public Leaderboard (Static Shift) ---
    print("Evaluating Static Set (Public LB)...")
    model = RobustClassifier().to(device)
    model.load_weights(model_path)
    
    static_data = torch.load(static_path, map_location='cpu')
    static_preds = adapt_and_predict(model, static_data['images'], device)
    
    for i, p in enumerate(static_preds):
        results.append({'ID': f'static_{i}', 'Category': int(p)})
        
    # --- PHASE 3: Private Leaderboard (The 24 Scenarios) ---
    print("\nEvaluating 24-Scenario Suite (Private LB)...")
    suite = torch.load(suite_path, map_location='cpu')
    scenario_keys = sorted([k for k in suite.keys() if k.startswith('scenario')])
    
    for skey in scenario_keys:
        print(f"  -> Adapting to {skey}...")
        # CRITICAL HARD RESET: Reload pristine weights to prevent catastrophic forgetting
        model = RobustClassifier().to(device)
        model.load_weights(model_path)
        
        scenario_images = suite[skey]
        preds = adapt_and_predict(model, scenario_images, device)
        
        for i, p in enumerate(preds):
            results.append({'ID': f'{skey}_{i}', 'Category': int(p)})

    # Save to CSV
    pd.DataFrame(results).to_csv('submission.csv', index=False)
    print("\n[SUCCESS] submission.csv generated and correctly formatted!")

if __name__ == '__main__':
    # Ensure source_toxic.pt, static.pt, val_sanity.pt, and test_suite_public.pt are in the directory
    main('weights.pth', 'static.pt', 'test_suite_public.pt')