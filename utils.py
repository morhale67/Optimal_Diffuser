from pytorch_lasso.lasso.linear import sparse_encode
import torch


def breg_rec(diffuser_batch, bucket_batch, batch_size):
    recs_container = torch.zeros((batch_size, diffuser_batch.shape[2]))
    for rec_ind in range(batch_size):
        niter_out = 1  # 50
        niter_in = 1  # 3
        mu = 10  # 0.01
        lamda = 0.3
        rec = sparse_encode(bucket_batch[rec_ind], diffuser_batch[rec_ind], maxiter=1, niter_inner=1, alpha=lamda,
                            algorithm='split-bregman')

        recs_container = recs_container.clone()
        recs_container[rec_ind] = rec

    return recs_container