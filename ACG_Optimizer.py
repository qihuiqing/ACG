# coding: utf-8
'''
Created on 2021年1月21日
@author: Chyi
'''
import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required

#optimizer
def tuple_divide_scalar(xs, y):
    return tuple([x / y for x in xs])

def tuple_subtract(xs, ys):
    return tuple([x - y for (x, y) in zip(list(xs), list(ys))])

def grad_T(ys, xs, grad_xs, create_graph, allow_unused): 
    us = tuple([torch.zeros_like(x, requires_grad=True) + float(1) for x in xs])
    dydxs = torch.autograd.grad(outputs=ys, inputs=xs, grad_outputs=us, create_graph=True)
    dysdx = torch.autograd.grad(outputs=dydxs, inputs=us, grad_outputs=grad_xs, create_graph=create_graph, allow_unused=allow_unused)
    return dysdx

def jacobian_vec(ys, xs, vs):
    return grad_T(ys, xs, grad_xs=vs, create_graph=True, allow_unused=True)

def jacobian_transpose_vec(ys, xs, vs):
    dydxs = torch.autograd.grad(ys, xs, grad_outputs=vs, create_graph=True, allow_unused=True)
    dydxs = [torch.zeros_like(x) if dydx is None else dydx for x, dydx in zip(xs, dydxs)]
    return tuple(dydxs)

def _dot(x, y):
    dot_list = []
    for xx, yy in zip(x, y):
        dot_list.append(torch.sum(torch.mul(xx, yy)))
    return sum(dot_list)


class SymplecticOptimizer(Optimizer):
    """Optimizer that corrects for rotational components in gradients."""
    def __init__(self, params, lr=required, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, 
                        centered=centered, weight_decay=weight_decay)
        super(SymplecticOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SymplecticOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    @torch.no_grad()
    def step(self, loss, closure=None):
        vars_ = [self.param_groups[i]['params'] for i in range(len(self.param_groups))]
        gen_loss = loss[0]
        disc_loss = loss[1]
        gen_vars = vars_[0]
        disc_vars = vars_[1]
        vars_s = gen_vars + disc_vars
        length = len(vars_s)
        with torch.enable_grad():
            gen_grads = torch.autograd.grad(gen_loss, gen_vars, create_graph=True, allow_unused=True)
            disc_grads = torch.autograd.grad(disc_loss, disc_vars, create_graph=True, allow_unused=True)
            grads = gen_grads + disc_grads
            h_v = jacobian_vec(grads, vars_s, grads)
            ht_v = jacobian_transpose_vec(grads, vars_s, grads)
            at_v = tuple_divide_scalar(tuple_subtract(ht_v, h_v), 2.)
        
        grad_dot_h = _dot(grads, ht_v)
        at_v_dot_h = _dot(at_v, ht_v)
        mult = grad_dot_h * at_v_dot_h
        lambda_ = torch.sign(mult / length + 0.1)
        apply_vec = [g + lambda_ * ag 
                     for (g, ag) in zip(grads, at_v)
                     if at_v is not None]

        gp_i = 0
        for group in self.param_groups:
            p_i = 0
            l = len(group["params"])
            for p in group["params"]:
                d_p = apply_vec[gp_i*l + p_i]
                p_i += 1
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p)
                
                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                square_avg.mul_(alpha).addcmul_(d_p, d_p, value=1 - alpha)

                avg = square_avg.sqrt().add_(group['eps'])

                p.addcdiv_(d_p, avg, value=-group['lr'])


            gp_i += 1
        
        return apply_vec, vars_s


class myoptimizer(Optimizer):
    """Optimizer that corrects for rotational components in gradients."""

    # def __init__(self, params, lr=required, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):

    def __init__(self, params, lr=required, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, 
                        centered=centered, weight_decay=weight_decay)
        super(myoptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(myoptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)
    
    
    @torch.no_grad()
    def step(self, alpha_vec, closure=None):
        l = 0
        for group in self.param_groups:
            for p in group["params"]:
                m = p.shape
                if len(m) == 2:
                    length = m[0] * m[1]
                else:
                    length = m[0]
                d_p = alpha_vec[l:l+length].view(m)
                l += length
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p)
                
                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                square_avg.mul_(alpha).addcmul_(d_p, d_p, value=1 - alpha)

                avg = square_avg.sqrt().add_(group['eps'])

                p.addcdiv_(d_p, avg, value=-group['lr'])
    

##RMSprop
class myRMSProp(Optimizer):
    """Optimizer that corrects for rotational components in gradients."""
    def __init__(self, params, lr=required, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, 
                        centered=centered, weight_decay=weight_decay)
        super(myRMSProp, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(myRMSProp, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    @torch.no_grad()
    def step(self, loss, closure=None):
        vars_ = [self.param_groups[i]['params'] for i in range(len(self.param_groups))]
        gen_loss = loss[0]
        disc_loss = loss[1]
        gen_vars = vars_[0]
        disc_vars = vars_[1]
        vars_s = gen_vars + disc_vars
        with torch.enable_grad():
            gen_grads = torch.autograd.grad(gen_loss, gen_vars, create_graph=True, allow_unused=True)
            disc_grads = torch.autograd.grad(disc_loss, disc_vars, create_graph=True, allow_unused=True)
            grads = gen_grads + disc_grads
        
        apply_vec = grads

        gp_i = 0
        for group in self.param_groups:
            p_i = 0
            l = len(group["params"])
            for p in group["params"]:
                d_p = apply_vec[gp_i*l + p_i]
                p_i += 1
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p)
                
                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                square_avg.mul_(alpha).addcmul_(d_p, d_p, value=1 - alpha)

                avg = square_avg.sqrt().add_(group['eps'])

                p.addcdiv_(d_p, avg, value=-group['lr'])


            gp_i += 1
        
        return apply_vec, vars_s
