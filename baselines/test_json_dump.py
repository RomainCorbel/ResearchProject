
from train import NumpyEncoder
import numpy as np, torch, json

payload = {
    "train_loss_surf": np.float32(0.0123),
    "train_loss_surf_var": np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32),
    "train_loss_vol": np.float32(0.0456),
    "train_loss_vol_var": np.array([0.05, 0.06, 0.07, 0.08], dtype=np.float32),
    "val_loss_surf": np.float32(0.0234),
    "val_loss_surf_var": np.array([0.02, 0.03, 0.01, 0.04], dtype=np.float32),
    "val_loss_vol": np.float32(0.0345),
    "val_loss_vol_var": np.array([0.03, 0.02, 0.05, 0.06], dtype=np.float32),
    "nb_parameters": np.int64(19988),
    "time_elapsed": np.float64(123.45),
    "torch_scalar": torch.tensor(1.23),
    "torch_tensor": torch.tensor([1.0, 2.0, 3.0]),
}
s = json.dumps(payload, cls=NumpyEncoder, indent=2)
print("✅ JSON dump success. Preview:")
print(s[:200].replace("\n"," ") + " ...")
