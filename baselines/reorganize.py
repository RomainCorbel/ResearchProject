"""import numpy as np

def reorganize(in_order_points, out_order_points, quantity_to_reordered):
    n = out_order_points.shape[0]
    idx = np.zeros(n)
    for i in range(n):
        cond = (out_order_points[i] == in_order_points)
        cond = cond[:, 0]*cond[:, 1]        
        idx[i] = np.argwhere(cond)[0][0]
    idx = idx.astype('int')

    assert (in_order_points[idx] == out_order_points).all()

    return quantity_to_reordered[idx]"""

import numpy as np
from scipy.spatial import cKDTree

def reorganize(in_order_points, out_order_points, quantity_to_reordered):
    """
    Réordonne `quantity_to_reordered` défini sur in_order_points pour suivre
    l'ordre de out_order_points.

    - Tente d'abord une correspondance exacte (rapide).
    - Si ça échoue, utilise un plus proche voisin (KDTree) pour gérer
      les légers décalages numériques / sous-échantillonnages.
    """
    in_order_points  = np.asarray(in_order_points)
    out_order_points = np.asarray(out_order_points)

    n = out_order_points.shape[0]
    idx = np.empty(n, dtype=np.int64)

    # 1) tentative: correspondance exacte via dict
    lookup = {tuple(p): i for i, p in enumerate(map(tuple, in_order_points))}
    exact_ok = True
    for i, p in enumerate(map(tuple, out_order_points)):
        j = lookup.get(p, None)
        if j is None:
            exact_ok = False
            break
        idx[i] = j

    # 2) fallback: nearest neighbor si l'égalité stricte échoue
    if not exact_ok:
        tree = cKDTree(in_order_points)
        # Pour tous les points de sortie, on prend l'indice du plus proche point d'entrée
        _, nn = tree.query(out_order_points, k=1)
        idx = nn.astype(np.int64)

    return quantity_to_reordered[idx]
