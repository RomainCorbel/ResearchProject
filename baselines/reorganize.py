'''import numpy as np

def reorganize(in_order_points, out_order_points, quantity_to_reordered):
    n = out_order_points.shape[0]
    idx = np.zeros(n)
    for i in range(n):
        cond = (out_order_points[i] == in_order_points)
        cond = cond[:, 0]*cond[:, 1]        
        idx[i] = np.argwhere(cond)[0][0]
    idx = idx.astype('int')

    assert (in_order_points[idx] == out_order_points).all()

    return quantity_to_reordered[idx]'''

import numpy as np

def reorganize(in_order_points, out_order_points, quantity_to_reordered, atol=1e-8, rtol=1e-10):
    n = out_order_points.shape[0]
    idx = np.empty(n, dtype=int)

    for i in range(n):
        # match all coordinates within tolerance (works for any dimensionality)
        cond = np.all(np.isclose(in_order_points, out_order_points[i], atol=atol, rtol=rtol), axis=1)
        hits = np.flatnonzero(cond)

        if hits.size == 0:
            # fall back to nearest neighbor so we don’t crash, but still guard by tol
            diffs = in_order_points - out_order_points[i]
            d2 = np.einsum('ij,ij->i', diffs, diffs)
            j = int(np.argmin(d2))
            if d2[j] > atol * atol:
                raise ValueError(
                    f"No match for out_order_points[{i}] within atol={atol}/rtol={rtol}; "
                    f"nearest distance={np.sqrt(d2[j])}."
                )
            idx[i] = j
        else:
            # if multiple, choose the closest geometrically
            if hits.size > 1:
                candidates = in_order_points[hits]
                d2 = np.sum((candidates - out_order_points[i])**2, axis=1)
                idx[i] = hits[int(np.argmin(d2))]
            else:
                idx[i] = hits[0]

    # tolerance-based sanity check
    assert np.allclose(in_order_points[idx], out_order_points, atol=atol, rtol=0)

    return quantity_to_reordered[idx]
