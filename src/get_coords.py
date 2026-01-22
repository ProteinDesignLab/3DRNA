import os

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

import common.run_manager
import seq_des.util.data as pdb_data

# >python get_coords.py --pdb_dir <pdb folder> --input_data <bgsu_csvfile_path> --save_dir <where you want to save the output>


def main():
    manager = common.run_manager.RunManager()
    manager.parse_args()  # args=[] if running on jupyter
    args = manager.args

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    print("Loading Dataset")
    dataset = pdb_data.get_dataset(input_data=args.input_data, pdb_dir=args.pdb_dir)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    gen = enumerate(iter(dataloader))

    n = 0

    print("iterating dataset")
    for it in tqdm(range(len(dataloader)), desc="loading and saving coords"):
        out = next(gen)[1]  # get next residue values in the data

        if len(out) == 0 or out is None:
            print("Out is None")
            continue

        pdb, chain, out_data = out
        res_ids, output_coor, output_data, res_label, chis = out_data
        res_label = torch.squeeze(res_label, 0)
        if res_label.dim() == 0:
            continue
        res_label = torch.unsqueeze(res_label, 1)
        chis = torch.squeeze(chis)
        for res_id, out_coor, out_data, out_res, out_chi in zip(res_ids, output_coor, output_data, res_label, chis):
            torch.save((pdb, res_id[1], res_id[0], out_coor, out_data, out_res, out_chi),
                       f"{args.save_dir}/data_{n}.pt")
            n += 1
    print(f'Saved {n} coords')

if __name__ == "__main__":
    main()