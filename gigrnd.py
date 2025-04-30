#!/usr/bin/env python

"""
Random variate generator for the generalized inverse Gaussian distribution.
Reference: L Devroye. Random variate generation for the generalized inverse Gaussian distribution.
           Statistics and Computing, 24(2):239-246, 2014.

"""


import math
from numpy import random
import torch


def psi(x, alpha, lam):
    f = -alpha*(math.cosh(x)-1.0)-lam*(math.exp(x)-x-1.0)
    return f


def dpsi(x, alpha, lam):
    f = -alpha*math.sinh(x)-lam*(math.exp(x)-1.0)
    return f


def g(x, sd, td, f1, f2):
    if (x >= -sd) and (x <= td):
        f = 1.0
    elif x > td:
        f = f1
    elif x < -sd:
        f = f2

    return f


def gigrnd(p, a, b):
    # setup -- sample from the two-parameter version gig(lam,omega)
    p = float(p); a = float(a); b = float(b)
    lam = p
    omega = math.sqrt(a*b)

    if lam < 0:
        lam = -lam
        swap = True
    else:
        swap = False

    alpha = math.sqrt(math.pow(omega,2)+math.pow(lam,2))-lam

    # find t
    x = -psi(1.0, alpha, lam)
    if (x >= 0.5) and (x <= 2.0):
        t = 1.0
    elif x > 2.0:
        if (alpha == 0) and (lam == 0):
            t = 1.0
        else:
            t = math.sqrt(2.0/(alpha+lam))
    elif x < 0.5:
        if (alpha == 0) and (lam == 0):
            t = 1.0
        else:
            t = math.log(4.0/(alpha+2.0*lam))

    # find s
    x = -psi(-1.0, alpha, lam)
    if (x >= 0.5) and (x <= 2.0):
        s = 1.0
    elif x > 2.0:
        if (alpha == 0) and (lam == 0):
            s = 1.0
        else:
            s = math.sqrt(4.0/(alpha*math.cosh(1)+lam))
    elif x < 0.5:
        if (alpha == 0) and (lam == 0):
            s = 1.0
        elif alpha == 0:
            s = 1.0/lam
        elif lam == 0:
            s = math.log(1.0+1.0/alpha+math.sqrt(1.0/math.pow(alpha,2)+2.0/alpha))
        else:
            s = min(1.0/lam, math.log(1.0+1.0/alpha+math.sqrt(1.0/math.pow(alpha,2)+2.0/alpha)))

    # find auxiliary parameters
    eta = -psi(t, alpha, lam)
    zeta = -dpsi(t, alpha, lam)
    theta = -psi(-s, alpha, lam)
    xi = dpsi(-s, alpha, lam)

    p = 1.0/xi
    r = 1.0/zeta

    td = t-r*eta
    sd = s-p*theta
    q = td+sd

    # random variate generation
    while True:
        U = random.random()
        V = random.random()
        W = random.random()
        if U < q/(p+q+r):
            rnd = -sd+q*V
        elif U < (q+r)/(p+q+r):
            rnd = td-r*math.log(V)
        else:
            rnd = -sd+p*math.log(V)

        f1 = math.exp(-eta-zeta*(rnd-t))
        f2 = math.exp(-theta+xi*(rnd+s))
        if W*g(rnd, sd, td, f1, f2) <= math.exp(psi(rnd, alpha, lam)):
            break

    # transform back to the three-parameter version gig(p,a,b)
    rnd = math.exp(rnd)*(lam/omega+math.sqrt(1.0+math.pow(lam,2)/math.pow(omega,2)))
    if swap:
        rnd = 1.0/rnd

    rnd = rnd/math.sqrt(a/b)
    return rnd


# PyTorch-based vectorized generalized inverse Gaussian sampler
def vectorized_gigrnd(lam, a, b, max_iter=1000):
    """
    Vectorized sampler for the generalized inverse Gaussian distribution using PyTorch.
    lam, a, b: torch tensors of same shape.
    Returns: torch tensor of same shape with GIG samples.
    """
    device = lam.device
    # ensure lam, a, b have same shape by explicitly broadcasting scalars
    if lam.dim() == 0:
        lam = lam.expand_as(a)
    if a.dim() == 0:
        a = a.expand_as(lam)
    if b.dim() == 0:
        b = b.expand_as(lam)
    # absolute shape and sign swap mask
    p_abs = lam.abs()
    swap_mask = lam < 0

    # parameters
    omega = torch.sqrt(a * b)
    alpha = torch.sqrt(omega**2 + p_abs**2) - p_abs

    # define psi and dpsi for tensors
    def psi_t(x):
        return -alpha * (torch.cosh(x) - 1) - p_abs * (torch.exp(x) - x - 1)
    def dpsi_t(x):
        return -alpha * torch.sinh(x) - p_abs * (torch.exp(x) - 1)

    ones = torch.ones_like(alpha)
    # compute t
    x = -psi_t(ones)
    cond1 = (x >= 0.5) & (x <= 2.0)
    cond2 = x > 2.0
    t = torch.where(cond1, ones,
                    torch.where(cond2,
                                torch.where((alpha==0)&(p_abs==0), ones, torch.sqrt(2.0/(alpha + p_abs))),
                                torch.where((alpha==0)&(p_abs==0), ones, torch.log(4.0/(alpha + 2.0*p_abs))))
                   )
    # compute s
    x = -psi_t(-ones)
    cond1s = (x >= 0.5) & (x <= 2.0)
    cond2s = x > 2.0
    s = torch.where(
        cond1s,
        ones,
        torch.where(
            cond2s,
            torch.where((alpha == 0) & (p_abs == 0), ones,
                        torch.sqrt(4.0 / (alpha * torch.cosh(ones) + p_abs))
                       ),
            torch.where(
                (alpha == 0) & (p_abs == 0),
                ones,
                torch.where(alpha == 0,
                            1.0 / p_abs,
                            torch.log(1.0 + 1.0 / alpha + torch.sqrt(1.0 / alpha**2 + 2.0 / alpha))
                           )
            )
        )
    )

    # auxiliary parameters
    eta = -psi_t(t)
    zeta = -dpsi_t(t)
    theta = -psi_t(-s)
    xi = dpsi_t(-s)

    p_param = 1.0 / xi
    r = 1.0 / zeta
    td = t - r * eta
    sd = s - p_param * theta
    q = td + sd

    # sampling loop
    shape = lam.shape
    samples = torch.zeros(shape, device=device)
    accepted = torch.zeros(shape, dtype=torch.bool, device=device)
    iter_count = 0
    while (~accepted).any() and iter_count < max_iter:
        U = torch.rand(shape, device=device)
        V = torch.rand(shape, device=device)
        W = torch.rand(shape, device=device)
        denom = p_param + q + r
        mask1 = U < q/denom
        mask2 = (U >= q/denom) & (U < (q + r)/denom)
        # propose candidate
        cand = torch.zeros(shape, device=device)
        cand = torch.where(mask1, -sd + q * V, cand)
        cand = torch.where(mask2, td - r * torch.log(V), cand)
        cand = torch.where(~mask1 & ~mask2, -sd + p_param * torch.log(V), cand)

        # acceptance test
        f1 = torch.exp(-eta - zeta * (cand - t))
        f2 = torch.exp(-theta + xi * (cand + s))
        env_mask = (cand >= -sd) & (cand <= td)
        gval = torch.where(env_mask, ones, torch.where(cand > td, f1, f2))

        accept = W * gval <= torch.exp(psi_t(cand))
        new_accept = accept & ~accepted
        samples = torch.where(new_accept, cand, samples)
        accepted = accepted | accept
        iter_count += 1

    # transform back to GIG(p, a, b)
    out = torch.exp(samples) * (p_abs/omega + torch.sqrt(1.0 + p_abs**2/omega**2))
    out = torch.where(swap_mask, 1.0/out, out)
    return out / torch.sqrt(a/b)


