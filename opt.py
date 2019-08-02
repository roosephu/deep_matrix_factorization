import torch
from torch.optim.optimizer import Optimizer


class GroupRMSprop(Optimizer):
    """A different version of RMSprop optimizer with a global learning rate adjusting.
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-6):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, alpha=alpha, eps=eps, adjusted_lr=lr)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            state = self.state

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                state['square_avg'] = torch.tensor(0.)

            square_avg = state['square_avg']
            alpha = group['alpha']
            square_avg.mul_(alpha)

            state['step'] += 1

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('GroupRMSprop does not support sparse gradients')

                square_avg.add_((1 - alpha) * grad.pow(2).sum().cpu().float())

            avg = square_avg.div(1 - alpha**state['step']).sqrt_().add_(group['eps'])
            lr = group['lr'] / avg
            group['adjusted_lr'] = lr

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                p.data.add_(-lr.to(grad.device) * grad)

        return loss
