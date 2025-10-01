import subprocess
import sys
from itertools import product


MODELS = ["MLP", "PointNet", "GraphSAGE", "GUNet"]
TASKS = ["scarce"]


def run(cmd):
    print("=" * 80)
    print(">>", " ".join(cmd))
    print("=" * 80, flush=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERREUR] La commande a échoué : {e}\n--- On continue avec la suivante ---")

if __name__ == "__main__":
    py = sys.executable or "python"
    for model, task in product(MODELS, TASKS):
        cmd = [py, "main.py", model, "-t", task, "-s", "1"]  # weight & nmodel = valeurs par défaut
        run(cmd)
