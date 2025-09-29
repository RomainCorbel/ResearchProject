import os, os.path as osp, json, yaml, torch
from dataset import Dataset
from models.MLP import MLP       # change to GraphSAGE or GUNet later
from train import main           # <- your simplified training loop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load dataset names ---
root_dir = os.getcwd()
dataset_dir = osp.join(root_dir, "Dataset")

with open(osp.join(dataset_dir, "manifest.json"), "r") as f:
    manifest = json.load(f)

# --- Pick only 10 training cases to test quickly ---
train_names = manifest["full_train"][:8]    # 8 for training
val_names   = manifest["full_train"][8:10]  # 2 for validation

# --- Build datasets ---
train_ds, norm = Dataset(train_names, norm=True, surface_only=True, pressure_only=True)
val_ds, _       = Dataset(val_names, norm=False, coef_norm=norm, surface_only=True, pressure_only=True)

print(f"Train: {len(train_ds)} cases, Val: {len(val_ds)} cases")

# --- Load model hyperparameters ---
hparams = yaml.safe_load(open(osp.join(root_dir, "params.yaml")))["MLP"]
net = MLP(hparams["encoder"] + hparams["decoder"][1:], batch_norm=hparams.get("bn_bool", True))

# --- Train ---
model = main(device, train_ds, val_ds, net, hparams, path="runs/test_run",
             criterion="MSE", name_mod="MLP")
