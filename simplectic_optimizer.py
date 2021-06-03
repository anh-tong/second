""" 
A Pytorch implementation of Simplectic Gradient Adjustment Optimizer in the paper `The Mechanics of n-Player Differential Games`
https://arxiv.org/abs/1802.05642
"""
from typing import Callable, List, Optional

import torch
from torch.autograd import grad
from torch.optim import Optimizer, SGD, Adam, RMSprop

class SGA(Optimizer):
    """Simplectic Gradient Adjustment Optimizer"""
    
    def __init__(
        self, 
        list_params,
        lr,
        reg_params=1.,
        base_optimizer:Optional[str]='SGD',
        use_signs=True) -> None:
        
        """
        Args:
            list_params: list of list parameters. The length of this list should be the same as the number of players
            lr: learning rate
            reg_params: regularized hyperparameter of adjustment term
            base_optimizer: accept options including SGD, Adam, RMSprop (default hyperparameters for all these optimizers for now)
            use_signs: whether or not turn on "aligned" option. `use_signs=True` converges faster according to the paper
        """
        
        self.list_params = list_params
        default = {}
        params = []
        for lp in list_params:
            params.extend(lp)
        self.all_params = params
        super().__init__(params, default)
        
        if base_optimizer == 'SGD':
            self._optim = SGD(params, lr=lr)
        elif base_optimizer == 'Adam':
            self._optim = Adam(params, lr=lr)
        elif base_optimizer == 'RMSprop':
            self._optim = RMSprop(params, lr=lr)
        else:
            raise ValueError("Only support base_optimizer= SGD, Adam, RMSprop")
        
        self.reg_params = reg_params
        self.use_signs = use_signs
        
    def step(self, closure: Callable[[], List[float]]) -> Optional[List[float]]:
        """
        Args:
            closure: A closure that returns a list of losses from n players
        """
        
        if closure is None:
            raise ValueError("SGA requires to pass a closure!")

        losses = closure()
        
        assert len(losses) == len(self.list_params)
        
        all_grads = []
        for loss, parameter in zip(losses, self.list_params):
            g = grad(
                loss,
                parameter,
                create_graph=True,
                retain_graph=True
            )
            all_grads.extend(g)
        
        # First part: transposed Jacobian-vector product
        Ht_xi = grad(
            all_grads,
            self.all_params,
            all_grads,
            retain_graph=True,
            allow_unused=True
        )
        
        # Second part: Jacobian vector product
        ws = [torch.zeros_like(g, requires_grad=True) for g in all_grads]
        jacobian = grad(
            all_grads,
            self.all_params,
            grad_outputs=ws,
            create_graph=True,
            allow_unused=True
        )
        # make None to zero
        jacobian = [torch.zeros_like(p) if j is None else j
                    for p, j in zip(self.all_params, jacobian)]
        H_xi = grad(
            jacobian,
            ws,
            grad_outputs=all_grads,
            allow_unused=True
        )
        
        # Adjustment term
        At_xi = [0.5*(ht - h) for ht, h in zip(Ht_xi, H_xi)]
        
        # compute the aligned signs
        if self.use_signs:
            grad_dot_h = sum([(g * h).sum() for g, h in zip(all_grads, Ht_xi)])
            at_dot_h = sum([(a * h).sum() for a, h in zip(At_xi, Ht_xi)])
            mult = grad_dot_h * at_dot_h
            lambda_ = torch.sign(mult / len(all_grads) + 0.1) * self.reg_params
        else:
            lambda_ = self.reg_params
            
        # modified gradient with adjustment
        gradients = [g.detach().clone() + lambda_ * a for g, a in zip(all_grads, At_xi)]
        
        # call base optimizer to update
        self._optim.zero_grad()
        for param, g in zip(self.all_params, gradients):
            param.grad = g
        
        self._optim.step()
        
        return losses
    
if __name__ == "__main__":
    
    """ A simple test for the optimizer"""
    import torch.nn as nn

    class SimpleModel(nn.Module):
        
        def __init__(self) -> None:
            super().__init__()
            
            self.x = nn.Parameter(torch.ones((1,)))
            self.y = nn.Parameter(torch.ones((1,)))
        
        def loss_1(self):
            
            return 0.5 * self.x ** 2 + 10. * self.x * self.y
        
        def loss_2(self):
            
            return 0.5 * self.y ** 2 - 10. * self.x * self.y
        
        
        def list_params(self):
            
            return [[self.x], [self.y]]
        

    model = SimpleModel()

    optimizer = SGA(model.list_params(), lr=0.01)

    for i in range(30):
        
        def compute_loss():
            return [model.loss_1(), model.loss_2()]

        losses = optimizer.step(compute_loss)
        losses = [loss.item() for loss in losses]
        print(f"Iter {i} \t Losses: {losses}")
        
    print(model.x.data, model.y.data)
    # We can see the parameters converge quickly to Nash equilibrium at (0,0)