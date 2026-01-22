import os

import pandas as pd
import torch
import wandb
from tqdm import tqdm

import common.atoms
import common.run_manager
import seq_des.models as models
import seq_des.sampler as sampler


def main():
    manager = common.run_manager.RunManager()
    manager.parse_args()
    args = manager.args

    # Create output directories if they don't exist
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.wandb_path, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    df = pd.read_csv(args.test_csv)

    args.chain = df['chain_id'][args.input_index]
    the_pdb = df['PDB_code'][args.input_index]
    args.pdb = os.path.join(args.pdb_dir, f'{the_pdb}_clean.cif')

    wandb.init(
        project="RNA Sequence Design",
        dir=args.wandb_path,
        config=vars(args)
    )

    if torch.cuda.is_available() and args.cuda:
        device = torch.device('cuda')
        print("Running on GPU")
    else:
        device = torch.device('cpu')
        print("Running on CPU")

    # Instantiate models - use model_list if provided, otherwise use model
    model_files = args.model_list if args.model_list else [args.model]
    if not model_files or model_files == [""]:
        raise ValueError("Must specify either --model or --model_list")
    
    classifiers = []
    for model_file in model_files:
        # If path is relative, join with model_dir; otherwise use as-is
        if not os.path.isabs(model_file):
            model_path = os.path.join(args.model_dir, model_file)
        else:
            model_path = model_file
        classifier = models.seqPred(nf=args.nf, nic=len(common.atoms.atoms) + 1)
        checkpoint = torch.load(model_path, map_location=device)
        classifier.load_state_dict(checkpoint['model'])
        classifier = classifier.to(device)
        classifier.eval()
        classifiers.append(classifier)

    # Set up design sampler
    design_sampler = sampler.Sampler(args, classifiers, device=device)

    # Initialize sampler
    design_sampler.init()

    # Initialize design_sampler sequence with baseline model prediction or random/poly-alanine/poly-valine initial sequence
    design_sampler.init_seq()

    # Run design
    with torch.no_grad():
        for i in tqdm(range(1, int(args.n_iters)), desc='running design'):
            design_sampler.step(i)

    # Save final model
    design_sampler.pose.dump_pdb(f'{args.log_dir}/curr_final.pdb')

if __name__ == "__main__":
    main()