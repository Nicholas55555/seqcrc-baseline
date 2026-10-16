"""The two performance patches to cods (patches/cods-reproduction.patch) must not change results.

1. Boolean-mask selection replaces a Python list comprehension over boxes: same rows, same order,
   same empty case.
2. _RunningSumList keeps sum(losses) up to date in O(1); it must match torch.sum(torch.stack(...))
   after any sequence of appends and single-element overwrites (as the optimizers do).
"""

import random

import torch

from cods.od.optim import _fast_correct_risk, _RunningSumList


def _original_selection(pred_cls_i, confidences_i, thr):
    sel = [x for x, c in zip(pred_cls_i, confidences_i) if c >= thr]
    return torch.stack(sel) if len(sel) > 0 else torch.tensor([]).float()


def _patched_selection(pred_cls_i, confidences_i, thr):
    sel = list(pred_cls_i[confidences_i >= thr])
    return torch.stack(sel) if len(sel) > 0 else torch.tensor([]).float()


def test_mask_selection_identical():
    g = torch.Generator().manual_seed(0)
    for n_boxes in [0, 1, 2, 7, 50, 300]:
        cls = torch.rand(n_boxes, 91, generator=g)
        conf = torch.rand(n_boxes, generator=g)
        for thr in [0.0, 1e-3, 0.074, 0.5, 0.999, 1.0, 1.5]:
            a = _original_selection(cls, conf, thr)
            b = _patched_selection(cls, conf, thr)
            assert a.shape == b.shape
            assert torch.equal(a, b)


def test_running_sum_matches_stack_sum():
    rng = random.Random(0)
    n = 2500
    losses = _RunningSumList()
    ref = []
    for _ in range(n):
        v = torch.tensor([rng.random()]).float()
        losses.append(v)
        ref.append(v)
    for _ in range(20000):  # monotone-max updates, as in the optimizers
        i = rng.randrange(n)
        new = torch.tensor([rng.random()]).float()
        v = max(losses[i].clone(), new)
        losses[i] = v
        ref[i] = v
    exact = (torch.sum(torch.stack(ref)) + 1) / (n + 1)
    fast = _fast_correct_risk(losses, n, 1)
    assert abs(float(exact) - float(fast)) < 1e-6
    copy = losses.copy()
    assert isinstance(copy, _RunningSumList) and abs(copy.total - losses.total) < 1e-9
