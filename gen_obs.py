import numpy as np
import torch

import lunzi as lz


class FLAGS(lz.BaseFLAGS):
    n = 100
    gt_path = ''
    obs_path = ''
    problem = ''
    n_train_samples = 0

    @classmethod
    def finalize(cls):
        if cls.problem == 'matrix-sensing':
            cls.obs_path = f'datasets/mat-sensing/{cls.n_train_samples}.pt'
        elif cls.problem == 'matrix-completion':
            cls.obs_path = f'datasets/mat-cmpl/{cls.n_train_samples}.pt'


@lz.main(FLAGS)
@FLAGS.inject
def main(n, problem, n_train_samples, gt_path, obs_path, _log):
    w_gt = torch.load(gt_path)

    with torch.no_grad():
        if problem == 'matrix-completion':
            indices = torch.multinomial(torch.ones(n * n), n_train_samples, replacement=False)
            us, vs = indices // n, indices % n
            ys_ = w_gt[us, vs]
            assert 0.8 <= ys_.pow(2).mean().sqrt() <= 1.2
            torch.save([(us, vs), ys_], obs_path)
        elif problem == 'matrix-sensing':
            xs = torch.randn(n_train_samples, n, n) / n
            ys_ = (xs * w_gt).sum(dim=-1).sum(dim=-1)
            assert 0.8 <= ys_.pow(2).mean().sqrt() <= 1.2
            torch.save([xs, ys_], obs_path)
        else:
            raise ValueError(f'unexpected problem: {problem}')
    _log.warning('[%s] Saved %d samples to %s', problem, n_train_samples, obs_path)


if __name__ == '__main__':
    main()
