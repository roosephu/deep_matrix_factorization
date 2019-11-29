import numpy as np

import lunzi as lz
import torch


class FLAGS(lz.BaseFLAGS):
    n = 100
    rank = 5
    symmetric = False
    gt_path = ''


@lz.main(FLAGS)
@FLAGS.inject
def main(n, rank, gt_path, symmetric):
    r = rank
    U = np.random.randn(n, r).astype(np.float32)
    if symmetric:
        V = U
    else:
        V = np.random.randn(n, r).astype(np.float32)
    w_gt = U.dot(V.T) / np.sqrt(r)
    w_gt = w_gt / np.linalg.norm(w_gt, 'fro') * n

    oracle_sv = np.linalg.svd(w_gt, compute_uv=False)
    lz.log.info("singular values = %s, Fro(w) = %.3f", oracle_sv[:r], np.linalg.norm(w_gt, ord='fro'))
    torch.save(torch.from_numpy(w_gt), gt_path)


if __name__ == '__main__':
    main()
