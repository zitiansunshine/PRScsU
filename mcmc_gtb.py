#!/usr/bin/env python

"""
Markov Chain Monte Carlo (MCMC) sampler for polygenic prediction with continuous shrinkage (CS) priors.

"""


import torch
import gigrnd


def mcmc(a, b, phi, sst_dict, n, ld_blk, blk_size, n_iter, n_burnin, thin, chrom, out_dir, beta_std, write_psi, write_pst, seed):
    print('... MCMC ...')

    # device and seeds for torch tensors
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"... Using device: {device} ...")
    if device.type == 'cuda':
        try:
            gpu_name = torch.cuda.get_device_name(device)
        except Exception:
            gpu_name = 'Unknown CUDA device'
        print(f"... CUDA device name: {gpu_name} ...")
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # derived stats to torch
    p = len(sst_dict['SNP'])
    beta_mrg = torch.tensor(sst_dict['BETA'], dtype=torch.float32, device=device).reshape(p,1)
    maf = torch.tensor(sst_dict['MAF'], dtype=torch.float32, device=device).reshape(p,1)
    n_pst = (n_iter - n_burnin) // thin
    # move LD blocks to torch
    ld_blk = [torch.tensor(bl, dtype=torch.float32, device=device) for bl in ld_blk]
    n_blk = len(ld_blk)
    # initialization with torch
    beta = torch.zeros((p,1), device=device)
    psi = torch.ones((p,1), device=device)
    sigma = 1.0
    if phi is None:
        phi = 1.0; phi_updt = True
    else:
        phi_updt = False
    if write_pst == 'TRUE':
        beta_pst = torch.zeros((p, n_pst), device=device)
    beta_est = torch.zeros((p,1), device=device)
    psi_est = torch.zeros((p,1), device=device)
    sigma_est = 0.0
    phi_est = 0.0

    # MCMC
    pp = 0
    for itr in range(1,n_iter+1):
        if itr % 100 == 0:
            print('--- iter-' + str(itr) + ' ---')

        mm = 0; quad = 0.0
        for kk in range(n_blk):
            if blk_size[kk] == 0:
                continue
            else:
                idx_blk = list(range(mm, mm+blk_size[kk]))
                # posterior precision matrix & Cholesky
                dinvt = ld_blk[kk] + torch.diag(1.0/psi[idx_blk].flatten())
                chol = torch.linalg.cholesky(dinvt, upper=True)
                # solve R^T y = beta_mrg by transposing R (R is upper-triangular) and treating as lower-triangular
                beta_tmp = torch.linalg.solve_triangular(chol.transpose(-2,-1), beta_mrg[idx_blk], upper=False, left=True)
                # add Gaussian noise with std sqrt(sigma/n)
                beta_tmp = beta_tmp + (sigma/n)**0.5 * torch.randn_like(beta_tmp)
                # solve R x = beta_tmp
                beta_blk = torch.linalg.solve_triangular(chol, beta_tmp, upper=True, left=True)
                beta[idx_blk] = beta_blk
                # accumulate quadratic form
                quad += (beta_blk.T @ dinvt @ beta_blk).item()
                mm += blk_size[kk]

        # update sigma via GPU-accelerated Gamma sampler
        term1 = n/2.0 * (1.0 - 2.0*torch.sum(beta*beta_mrg) + quad)
        term2 = n/2.0 * torch.sum(beta**2/psi)
        err = torch.max(term1, term2)
        # correct rate=err for Gamma(shape=(n+p)/2, scale=1/err)
        sigma = (1.0 / torch.distributions.Gamma(concentration=(n+p)/2.0, rate=err).sample()).item()
        # update local shrinkage via vectorized GIG
        delta = torch.distributions.Gamma(concentration=(a+b), rate=(psi+phi)).sample()
        psi = gigrnd.vectorized_gigrnd(torch.tensor(a-0.5, device=device), 2.0*delta, (n*beta**2/sigma))
        psi = torch.minimum(psi, torch.ones_like(psi))
        # update global shrinkage if needed
        if phi_updt:
            # correct rate=phi+1.0 for Gamma(1, scale=1/(phi+1.0))
            w = torch.distributions.Gamma(concentration=1.0, rate=(phi+1.0)).sample().item()
            # correct rate=sum(delta)+w for Gamma(p*b+0.5, scale=1/(sum(delta)+w))
            phi = torch.distributions.Gamma(concentration=(p*b+0.5), rate=(delta.sum().item()+w)).sample().item()

        # posterior
        if (itr>n_burnin) and (itr % thin == 0):
            beta_est = beta_est + beta/n_pst
            psi_est = psi_est + psi/n_pst
            sigma_est = sigma_est + sigma/n_pst
            phi_est = phi_est + phi/n_pst

            if write_pst == 'TRUE':
                beta_pst[:,[pp]] = beta
                pp += 1

    # convert standardized beta to per-allele beta
    if beta_std == 'FALSE':
        beta_est /= torch.sqrt(2.0*maf*(1.0-maf))

        if write_pst == 'TRUE':
            beta_pst /= torch.sqrt(2.0*maf*(1.0-maf))


    # write posterior effect sizes
    if phi_updt == True:
        eff_file = out_dir + '_pst_eff_a%d_b%.1f_phiauto_chr%d.txt' % (a, b, chrom)
    else:
        eff_file = out_dir + '_pst_eff_a%d_b%.1f_phi%1.0e_chr%d.txt' % (a, b, phi, chrom)

    with open(eff_file, 'w') as ff:
        if write_pst == 'TRUE':
            for snp, bp, a1, a2, beta in zip(sst_dict['SNP'], sst_dict['BP'], sst_dict['A1'], sst_dict['A2'], beta_pst):
                ff.write(('%d\t%s\t%d\t%s\t%s' + '\t%.6e'*n_pst + '\n') % (chrom, snp, bp, a1, a2, *beta))
        else:
            for snp, bp, a1, a2, beta in zip(sst_dict['SNP'], sst_dict['BP'], sst_dict['A1'], sst_dict['A2'], beta_est):
                ff.write('%d\t%s\t%d\t%s\t%s\t%.6e\n' % (chrom, snp, bp, a1, a2, beta))

    # write posterior estimates of psi
    if write_psi == 'TRUE':
        if phi_updt == True:
            psi_file = out_dir + '_pst_psi_a%d_b%.1f_phiauto_chr%d.txt' % (a, b, chrom)
        else:
            psi_file = out_dir + '_pst_psi_a%d_b%.1f_phi%1.0e_chr%d.txt' % (a, b, phi, chrom)

        with open(psi_file, 'w') as ff:
            for snp, psi in zip(sst_dict['SNP'], psi_est):
                ff.write('%s\t%.6e\n' % (snp, psi))

    # print estimated phi
    if phi_updt == True:
        print('... Estimated global shrinkage parameter: %1.2e ...' % phi_est )

    print('... Done ...')


