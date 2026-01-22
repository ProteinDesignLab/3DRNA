import os

import numpy as np
import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm

import common.atoms
import common.run_manager
import seq_des.models as models
import seq_des.util.data as pdb_data

N_CHI_BINS = 36
CHI_BINS = np.linspace(-180, 180, num=N_CHI_BINS)
num_resis = 4
num_atoms = len(common.atoms.atoms)
num_bb = len(common.atoms.bb_elem)

"""Script to evaluate 3D CNN on local residue-centered environments with autoregressive rotamer chi angle prediction"""
n = 4
ncb = len(pdb_data.CHI_BINS)


def collate_fn(inputs):
    if isinstance(inputs[0], dict):
        output = {}
        for name in inputs[0].keys():
                if isinstance(inputs[0][name], dict):
                    output[name] = collate_fn([input[name] for input in inputs if input is not None])
                elif isinstance(inputs[0][name], np.ndarray):
                    output[name] = torch.stack([torch.tensor(input[name]) for input in inputs if input is not None], dim=0)
                elif isinstance(inputs[0][name], torch.Tensor):
                    output[name] = torch.stack([input[name] for input in inputs if input is not None], dim=0)
                else:
                    output[name] = [input[name] for input in inputs if input is not None]
                if len(output[name]) == 0:
                    return None
        return output
    else:
        return inputs

def main(args):
    torch.cuda.empty_cache()

    # Create output directories if they don't exist
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    if torch.cuda.is_available() and args.cuda:
        print("Running model on GPU")
        print("Using", torch.cuda.device_count(), "GPUs")
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()
    else:
        print("Running model on CPU")
        device = torch.device('cpu')

    c = len(common.atoms.atoms)

    classifier = models.seqPred(nf=args.nf, nic=c + 1)

    # Use --model argument, add .pt extension if not present
    model_file = args.model
    if not model_file:
        raise ValueError("Must specify --model for evaluation")
    if not model_file.endswith('.pt'):
        model_file = f'{model_file}.pt'
    
    model_path = os.path.join(args.model_dir, model_file)
    chkpt = torch.load(model_path, map_location=device)

    classifier.load_state_dict(chkpt['model'])
    classifier = classifier.to(device)
    classifier.eval()

    dataset = pdb_data.PDBDataset(coords_dir=args.test_coords_dir,
                                   voxel_size=args.voxel_size, bb_only=args.bb_only, set_len=16326)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    acc = []
    # Evaluate
    with torch.no_grad():
        for it, data_ in enumerate(tqdm(dataloader)):
            if data_ is None:
                continue
            try:
                X, y = data_['input'], data_['res_labels']
            except Exception:
                continue
            y = common.atoms.label_res_rna_dict[y.long().to(device).item()]
            X = X.float().to(device)
            out_pred, chi_pred = classifier.get_feat(X)
            chi_predicted = CHI_BINS[torch.argmax(chi_pred, 1).item()]
            chi_real_idx = data_['chi_binned'][0, 0].item()
            chi_real = CHI_BINS[chi_real_idx]

            pred = common.atoms.label_res_rna_dict[torch.argmax(out_pred, 1).item()]
            acc.append([data_['pdb'][0], data_['chain'][0], data_['res_id'][0], y, pred, chi_real, chi_predicted, out_pred])

    df = pd.DataFrame(acc)
    df.columns = ['pdb', 'chain', 'res_idx', 'wt', 'predicted', 'chi_real', 'chi_pred', 'logits']
    model_name = model_file.replace('.pt', '')
    output_path = os.path.join(args.log_dir, f'{model_name}.csv')
    df.to_csv(output_path)
    print(f"Evaluation results saved to {output_path}")    

if __name__ == "__main__":

    manager = common.run_manager.RunManager()

    manager.parse_args()
    args = manager.args
    main(args)
