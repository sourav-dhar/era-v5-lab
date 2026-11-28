"""Muon leaves rows with no gradient signal alone: an idle expert's weights, and its row of the router.

The fused MoE runs every expert on every token and gates the unchosen ones to zero, so an expert that saw no tokens
gets an all-zero gradient (not None), and its router row a gradient of about 1e-10. Newton-Schulz scales every row it
gets to full size, so without a mask both would take a normal-sized step from stale momentum on every idle step.
"""
from pathlib import Path
import sys
import unittest

import torch

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM / "harness"))
from week2.model import MoE
from week2.optim import MUON, Muon, zeropower_batched

D, E = 32, 8


def reference_step(params, bufs, lr, momentum=0.95, nesterov=True, ns_steps=5):
    """One same-shape group as Muon.step did before rows were masked (it assumes every row has a gradient)."""
    grads = [p.grad for p in params]
    torch._foreach_lerp_(bufs, grads, 1 - momentum)
    gs = torch._foreach_lerp(grads, bufs, momentum) if nesterov else bufs
    U = zeropower_batched(torch.stack(gs), ns_steps)
    alpha = -lr * max(1.0, params[0].shape[0] / params[0].shape[1]) ** 0.5
    torch._foreach_add_(params, [u.to(p.dtype) for u, p in zip(U.unbind(0), params)], alpha=alpha)


def moe():
    torch.manual_seed(0)
    cfg = {"n_embd": D, "moe": {"num_experts": E, "top_k": 2, "num_shared": 1, "ffn_hidden": 48,
                                "bias": False, "bias_gamma": 0.001}}
    return MoE(cfg).train()


def step(model, opt, idle=None, seed=0, n=64):
    """One step; `idle` gets no tokens (a large negative selection bias, which never reaches the gate values)."""
    with torch.no_grad():
        model.router_bias.zero_()
        if idle is not None:
            model.router_bias[idle] = -100.0
    x = torch.randn(n, D, generator=torch.Generator().manual_seed(seed))
    model(x, torch.ones(n, dtype=torch.bool)).pow(2).mean().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


class IdleRowTests(unittest.TestCase):
    def test_all_zero_gradient_is_treated_like_a_missing_one(self):
        p = torch.nn.Parameter(torch.randn(16, 24))
        opt = Muon([p], lr=0.02, **MUON)
        p.grad = torch.randn(16, 24)
        opt.step()
        weight, buf = p.detach().clone(), opt.state[p]["buf"].clone()
        for _ in range(5):
            p.grad = torch.zeros(16, 24)
            opt.step()
        self.assertTrue(torch.equal(p.detach(), weight))
        self.assertTrue(torch.equal(opt.state[p]["buf"], buf))

    def test_an_idle_expert_and_its_router_row_stay_put(self):
        model = moe()
        opt = Muon([p for p in model.parameters() if p.ndim == 2], lr=0.006, **MUON)
        step(model, opt)  # every expert takes tokens once, so every row has momentum
        idle = 3
        router0 = model.router.weight.detach().clone()
        expert0 = [p.detach().clone() for p in model.experts[idle].parameters()]
        busy0 = [p.detach().clone() for p in model.experts[0].parameters()]
        for s in range(1, 6):
            step(model, opt, idle=idle, seed=s)
        moved = (model.router.weight.detach() - router0).norm(dim=1)
        self.assertEqual(moved[idle].item(), 0.0)
        self.assertTrue(bool((moved[[i for i in range(E) if i != idle]] > 0).all()))
        for p, p0 in zip(model.experts[idle].parameters(), expert0):
            self.assertTrue(torch.equal(p.detach(), p0))
        self.assertTrue(any(not torch.equal(p.detach(), p0) for p, p0 in zip(model.experts[0].parameters(), busy0)))

    def test_rows_with_a_gradient_update_exactly_as_before(self):
        torch.manual_seed(1)
        params = [torch.nn.Parameter(torch.randn(16, 24)) for _ in range(3)]
        ref = [torch.nn.Parameter(p.detach().clone()) for p in params]
        opt = Muon(params, lr=0.02, **MUON)
        bufs = [torch.zeros(16, 24) for _ in ref]
        for _ in range(4):
            grads = [torch.randn(16, 24) for _ in params]
            for p, q, g in zip(params, ref, grads):
                p.grad, q.grad = g.clone(), g.clone()
            opt.step()
            with torch.no_grad():
                reference_step(ref, bufs, 0.02)
        for p, q in zip(params, ref):
            self.assertTrue(torch.equal(p.detach(), q.detach()))

    def test_a_zero_first_step_stays_finite_and_unchanged(self):
        p = torch.nn.Parameter(torch.randn(8, 12))
        weight = p.detach().clone()
        opt = Muon([p], lr=0.02, **MUON)
        p.grad = torch.zeros(8, 12)
        opt.step()
        self.assertTrue(torch.equal(p.detach(), weight))


if __name__ == "__main__":
    unittest.main()
