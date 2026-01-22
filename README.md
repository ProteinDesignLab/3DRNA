# 3DRNA

A deep learning framework for designing RNA sequences from 3D structural environments using 3D convolutional neural networks (CNNs). The model predicts RNA nucleotide identities (A, C, G, U) and chi angles from local residue-centered structural environments extracted from RNA crystal structures.

## Overview

This project implements a structure-based RNA sequence design pipeline that:

1. **Extracts local structural environments** from RNA crystal structures (PDB/CIF files)
2. **Model training** to predict nucleotide identities and chi angles from voxelized structural environments
3. **RNA sequence design** for target structures using trained models

The model uses a 3D CNN architecture that processes voxelized atomic environments centered on each nucleotide to predict:
- **Nucleotide type**: A, C, G, or U
- **Chi angles**: Torsion angles for RNA backbone conformations

## Project Structure

```
3DRNA/
├── src/                    # Main source code
│   ├── common/            # Common utilities
│   │   ├── atoms.py       # Atom and residue definitions
│   │   ├── logger.py      # Logging utilities
│   │   └── run_manager.py # Argument parsing and run management
│   ├── data/              # Data processing modules
│   │   ├── models.py      # Neural network model definitions
│   │   ├── sampler.py     # Sequence design sampler
│   │   └── util/          # Data utilities
│   ├── get_coords.py      # Extract coordinates from PDB files
│   ├── train.py           # Training script
│   ├── run.py             # Sequence design/inference script
│   └── evals.py           # Model evaluation script
├── input/                 # Input PDB/CIF files
├── output/                # Output designed structures
├── splits/                # Train/test data splits
├── ipybs/                 # Jupyter notebooks for analysis
└── README.md
```

## Installation

### Dependencies

- Python 3.x
- PyTorch
- NumPy
- BioPython
- PyRosetta (for structure manipulation)
- wandb (for experiment tracking)
- tqdm
- pandas

## Usage

### Step 1: Extract Coordinates

Extract nucleotide-centered structural environments from PDB/CIF files:

```bash
python src/get_coords.py \
    --pdb_dir <path_to_pdb_directory> \
    --input_data <path_to_csv_with_pdb_info> \
    --save_dir <output_directory>
```

The input CSV should contain columns for PDB codes and chain IDs. Output files are saved as PyTorch tensors (`.pt` files) containing voxelized structural environments.

### Step 2: Model Training

Train the 3DCNN on extracted structural environments:

```bash
python src/train.py \
    --coord_dir <directory_with_coordinate_files> \
    --log_dir <checkpoint_directory> \
    --wandb_path <wandb_log_directory> \
    --batchSize <batch_size> \
    --epochs <num_epochs> \
    --lr <learning_rate> \
    --nf <number_of_filters> \
    --voxel_size <voxel_size> \
    --use_chi_bin <0_or_1> \
    --weight_chi <chi_loss_weight> \
    --cuda
```

**Key training parameters:**
- `--coord_dir`: Directory containing preprocessed coordinate files (default: `data/coords`)
- `--log_dir`: Directory to save model checkpoints (default: `output` - created automatically)
- `--wandb_path`: Directory for wandb experiment logs (default: `logs` - created automatically)
- `--batchSize`: Batch size for training (default: 128)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--nf`: Number of filters in the CNN (default: 32)
- `--voxel_size`: Size of voxels for discretization (default: 0.5)
- `--use_chi_bin`: Use binned chi angles instead of continuous values (default: 1)
- `--weight_chi`: Weight for chi angle loss term (default: 1.0)
- `--bb_only`: Use only backbone atoms (0 or 1, default: 0)
- `--model`: Path to pretrained model for resuming training (optional)
- `--validation_frequency`: How often to validate during training (default: 50)
- `--save_frequency`: How often to save model checkpoints (default: 100)

### Step 3: Design Sequences

Design RNA sequences for target structures:

```bash
python src/run.py \
    --pdb <path_to_target_structure.pdb> \
    --chain <chain_id> \
    --model <model_filename.pt> \
    --model_dir <directory_with_models> \
    --n_iters <number_of_iterations> \
    --log_dir <output_directory> \
    --wandb_path <wandb_log_directory> \
    --cuda
```

**For ensemble design (averages multiple models):**
```bash
python src/run.py \
    --pdb <path_to_target_structure.pdb> \
    --model_list model1.pt model2.pt model3.pt \
    --n_iters <number_of_iterations> \
    --cuda
```

**Key design parameters:**
- `--pdb`: Input structure file (PDB or CIF format)
- `--chain`: Chain ID to design
- `--model`: Single model filename for design (default: uses `--model_list` if provided)
- `--model_list`: Space-separated list of model filenames for ensemble design (averages predictions)
- `--model_dir`: Directory containing model checkpoints (default: `models`)
- `--n_iters`: Number of design iterations
- `--log_dir`: Output directory for designed structures (default: `output` - created automatically)
- `--wandb_path`: Directory for wandb logs (default: `logs` - created automatically)
- `--threshold`: Probability threshold for mutations
- `--anneal`: Use simulated annealing
- `--seed`: Random seed for reproducibility

**For batch design from CSV:**
```bash
python src/run.py \
    --test_csv <path_to_csv_with_pdb_codes> \
    --pdb_dir <directory_with_pdb_files> \
    --input_index <row_index_in_csv> \
    --model <model_filename.pt> \
    --n_iters <number_of_iterations> \
    --cuda
```

The design process iteratively:
1. Extracts local environments for each nucleotide
2. Predicts residue probabilities using the trained model
3. Samples new sequences based on predictions
4. Updates the structure and repeats

### Step 4: Evaluate Models

Evaluate trained models on test data:

```bash
python src/evals.py \
    --model <model_filename.pt> \
    --model_dir <directory_with_models> \
    --test_coords_dir <test_data_directory> \
    --log_dir <output_directory> \
    --voxel_size <voxel_size> \
    --bb_only <0_or_1> \
    --cuda
```

**Key evaluation parameters:**
- `--model`: Model filename (with or without `.pt` extension, e.g., `conditional_model_f1.pt` or `conditional_model_f1`)
- `--model_dir`: Directory containing model checkpoints (default: `models`)
- `--test_coords_dir`: Directory containing test coordinate files (default: `data/coords/test`)
- `--log_dir`: Output directory for evaluation results CSV (default: `output` - created automatically)
- `--voxel_size`: Size of voxels for discretization (default: 0.5)
- `--bb_only`: Use only backbone atoms (0 or 1, default: 0)
- `--cuda`: Use GPU if available

The evaluation script outputs a CSV file with predictions and ground truth labels for each residue.

## Data Format

### Input Structures

- PDB or CIF format RNA structures
- Structures should contain standard RNA atoms (C1', C2', C3', C4', C5', O2', O3', O4', O5', P, OP1, OP2)

### Coordinate Files

Preprocessed coordinate files (`.pt` format) contain:
- Voxelized atomic environments (40×40×40 voxel grids)
- Residue labels (A, C, G, U)
- Chi angles
- Atom type information

### Data Splits

Train/test splits are provided in the `splits/` directory:
- `rna_train.csv`, `rna_test.csv`: Main dataset splits
- `BGSUdataset_train.csv`, `BGSUdataset_test.csv`: BGSU dataset splits

## Model Architecture

The model (`seqPred`) uses a 3D CNN architecture:

1. **Feature extraction**: Multiple 3D convolutional layers with batch normalization and dropout
2. **Residue prediction**: Predicts nucleotide identity (A/C/G/U) from structural features
3. **Chi angle prediction**: Predicts chi angles conditioned on residue type and structural features

The model can operate in two modes:
- **Binned chi angles**: Predicts chi angles as discrete bins (36 bins, 10° each)
- **Continuous chi angles**: Predicts chi angles as continuous values using circular regression

## Path Configuration

All paths are configurable via command-line arguments. The following path-related arguments are available:

**Data paths:**
- `--pdb_dir`: Directory containing input PDB/CIF files (default: `data/pdb`)
- `--coord_dir`: Directory containing preprocessed coordinate files (default: `data/coords`)
- `--test_coords_dir`: Directory containing test coordinate files (default: `data/coords/test`)
- `--save_dir`: Directory for saving extracted coordinates (default: `data/coords` - created automatically)
- `--input_data`: Path to CSV file with training data (default: `data/train.csv`)
- `--test_csv`: Path to CSV file with test PDB codes and chain IDs (default: `data/test.csv`)

**Model paths:**
- `--model_dir`: Directory for all model checkpoints - training saves models here, all scripts load models from here (default: `models` - created automatically)
- `--model`: Model filename for resuming training, evaluation, or single-model design (e.g., `seq_RNA_epoch_0_100.pt` or `conditional_model_f1.pt`)
- `--model_list`: Model filenames for ensemble design (averages predictions from multiple models). If not specified, uses `--model` (e.g., `conditional_model_f1.pt conditional_model_f2.pt`)

**Output paths:**
- `--log_dir`: Output directory for results and checkpoints (default: `output` - created automatically)
- `--wandb_path`: Directory for wandb experiment logs (default: `logs` - created automatically)

All paths can be overridden via command-line arguments to adapt to different environments.

## Output

Designed sequences are saved as PDB files in the specified output directory:
- `*_start.pdb`: Initial sequence
- `*_curr.pdb`: Current sequence during design
- `*_final.pdb`: Final designed sequence

Evaluation results are saved as CSV files in the log directory with columns:
- `pdb`, `chain`, `res_idx`: Structure identifiers
- `wt`, `predicted`: Ground truth and predicted nucleotide types
- `chi_real`, `chi_pred`: Ground truth and predicted chi angles
- `logits`: Model output logits

## References

This project implements structure-based RNA sequence design using deep learning. The approach is similar to protein sequence design methods (Anand et. al., Nature Communications) but adapted for RNA structures and nucleotide prediction.
