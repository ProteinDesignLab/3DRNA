import os
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from tqdm import tqdm
import wandb

import common.atoms
import common.run_manager
import seq_des.models as models
import seq_des.util.acc_util as acc_util
import seq_des.util.data as pdb_data

num_resis = len(common.atoms.label_res_rna_dict)
num_atoms = len(common.atoms.atoms)
num_bb = len(common.atoms.bb_elem)
ncb = pdb_data.N_CHI_BINS

"""Script to train 3D CNN on local RNA residue-centered environments"""


def circular_mean_absolute_error(chi_hat, chi):
    """Compute the difference in angles while considering angle wrapping."""
    diff = torch.atan2(torch.sin(chi_hat - chi), torch.cos(chi_hat - chi))
    return torch.abs(diff)

def test(model, test_gen, test_dataloader, criterion, chi_criterion, device, max_it=1e6, n_iters=10, use_chi_bin=False):
    n_iters = min(max_it, n_iters)

    losses, avg_acc, avg_chi_1_acc, avg_chi_1_loss, avg_type_acc = ([] for i in range(5))

    with torch.no_grad():
        for i in range(n_iters):
            try:
                test_data = next(test_gen)
            except StopIteration:
                test_gen = enumerate(iter(test_dataloader))
                test_data = next(test_gen)
            out = step(model, test_data[1], criterion, chi_criterion, device, mode='test', use_chi_bin=use_chi_bin)

            if out is None:
                continue

            # Append losses and accuracies to lists
            for x, y in zip([losses, avg_acc, avg_chi_1_loss, avg_chi_1_acc, avg_type_acc], out):
                x.append(y)

        print("\ntest: loss", np.mean(losses), "acc", np.mean(avg_acc), "type", np.mean(avg_type_acc),
              "chi loss", np.mean(avg_chi_1_loss), "chi acc", np.mean(avg_chi_1_acc))

    return test_gen, [np.mean(losses).item(),
                      np.mean(avg_acc).item(),
                      np.mean(avg_chi_1_loss).item(),
                      np.mean(avg_chi_1_acc).item(),
                      np.mean(avg_type_acc).item()]


def step(model, out, criterion, chi_1_criterion, device, mode='train', use_chi_bin=False):
    if out is None or len(out) != 4:
        return None

    # Note: chi_angles are in degrees
    X, y, chi_angles, chis_binned = out['input'], out['res_labels'], out['chi_angles'], out['chi_binned']

    bs_i = len(y)
    y = y.long().to(device)
    chi_angles = chi_angles.long().to(device)
    X = X.float().to(device)

    if use_chi_bin:
        chis_binned = chis_binned.to(device)
    else:
        chi_angles = chi_angles.to(device)

    res_onehot = torch.zeros(size=(bs_i, num_resis), dtype=torch.int8).to(device)
    res_onehot.scatter_(1, y, 1)

    out_pred = model(X, res_onehot)

    y = y.squeeze()
    res_loss = criterion(out_pred[0], y)
    acc = acc_util.get_acc(out_pred[0], y)
    type_acc = acc_util.get_base_type_acc(out_pred[0], y)

    if use_chi_bin:
        chi_1_loss = chi_1_criterion(out_pred[1], chis_binned[:, 0])
        chi_1_acc = acc_util.get_acc(out_pred[1], chis_binned[:, 0], ignore_idx=-1)
    else:
        chi_1_loss = circular_mean_absolute_error(out_pred[1], torch.deg2rad(chi_angles[:, 0]))
        chi_1_loss = torch.pow(chi_1_loss, 2).cpu().detach().numpy()  # L2 loss
        chi_1_loss = chi_1_loss.mean()  # mean
        chi_1_acc = chi_1_loss

    if mode == 'test':
        res_loss = res_loss.item()
        chi_1_loss = chi_1_loss.item()

    return [res_loss, acc, chi_1_loss, chi_1_acc, type_acc]

def collate_fn(inputs: List[Any]) -> Any:
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
    os.makedirs(args.wandb_path, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    if torch.cuda.is_available() and args.cuda:
        print("Training model on GPU")
        print("Using", torch.cuda.device_count(), "GPUs")
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()
    else:
        print("Training model on CPU")
        device = torch.device('cpu')

    if args.bb_only:
        c = len(common.atoms.bb_elem)
    else:
        c = len(common.atoms.atoms)
        trdata = 'grnade_train/pts'
        tedata = 'grnade_test'
        vdata = 'grnade_val'

    # Load dataset
    train_dataset = pdb_data.PDBDataset(coords_dir=f'{args.coord_dir}/{trdata}', noise=args.noise,
                                        voxel_size=args.voxel_size, bb_only=args.bb_only)

    test_dataset = pdb_data.PDBDataset(coords_dir=f'{args.coord_dir}/{vdata}', noise=0,
                                       voxel_size=args.voxel_size, bb_only=args.bb_only)

    train_dataset.len = 1315689
    test_dataset.len = 7466

    nic = c + 1
    model = models.seqPred(nic=nic, nf=args.nf, use_chi_bin=args.use_chi_bin)
    model.apply(models.init_ortho_weights)
    model.to(device)

    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.reg)
    optimizer.zero_grad()

    # Load pre-trained model
    if args.model != "":
        # If path is relative, join with model_dir; otherwise use as-is
        model_path = args.model
        if not os.path.isabs(model_path):
            model_path = os.path.join(args.model_dir, model_path)
        chkpt = torch.load(model_path, map_location=device)
        model.load_state_dict(chkpt['model'])


    # Parallelize over available GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).to(device)

    # Create dataloader
    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batchSize, num_workers=4, prefetch_factor=2,
                                        shuffle=False, pin_memory=True, collate_fn=collate_fn)
    test_dataloader = data.DataLoader(test_dataset, batch_size=args.batchSize, shuffle=False, pin_memory=True,
                                      collate_fn=collate_fn)

    test_gen = enumerate(iter(test_dataloader))

    # Training parameters
    validation_frequency = args.validation_frequency
    save_frequency = args.save_frequency
    ld = len(train_dataloader)

    # Start a new wandb run to track this script
    wandb.init(
        settings=wandb.Settings(_service_wait=300),
        project="RNA SeqDes - Training",
        dir=args.wandb_path,
        config=vars(args)
    )
    print(wandb.config)

    model.train()
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(common.atoms.res_weights)).to(device)
    chi_1_criterion = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.Tensor(common.atoms.chi_weights)).to(device)

    print(f"Total params: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Iterate through epochs
    for epoch in range(0, args.epochs):
        # Iterate through dataset
        for it, train_data in enumerate(tqdm(train_dataloader, desc="training epoch %0.2d" % epoch, miniters=500)):
            out = step(model, train_data, criterion, chi_1_criterion, device, mode='train', use_chi_bin=args.use_chi_bin)
            if out is None:
                continue

            # Calculate training loss
            train_loss = out[0] + args.weight_chi * out[2]  # res_loss + chi_1_loss
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if it % 25 == 0:
                wandb.log({"train_loss": out[0].item(),
                           "train_acc": out[1],
                           "train_chi_1_loss": out[2].item(),
                           "train_chi_1_acc": out[3],
                           "train_type_acc": out[4]})

            if it % validation_frequency == 0 or it == ld - 1:
                # After iterating the dataset run validation
                print(f'Running validation: {it}')

                # Saving models
                if it > 0:
                    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                    torch.save(state, os.path.join(args.model_dir, "seq_RNA_curr.pt"))

                    if it % save_frequency == 0:
                        torch.save(state, os.path.join(args.model_dir, f"seq_RNA_epoch_{epoch}_{it}.pt"))

                # Run evaluation
                model.eval()
                with torch.no_grad():
                    test_gen, test_out = test(model, test_gen, test_dataloader, criterion, chi_1_criterion, device,
                                              max_it=ld, n_iters=min(args.n_iters, ld), use_chi_bin=args.use_chi_bin)
                    wandb.log({"test_loss": test_out[0],
                               "test_acc": test_out[1],
                               "test_chi_1_loss": test_out[2],
                               "test_chi_1_acc": test_out[3],
                               "test_type_acc": test_out[4]})
                model.train()

if __name__ == "__main__":
    manager = common.run_manager.RunManager()
    manager.parse_args()
    args = manager.args
    main(args)
