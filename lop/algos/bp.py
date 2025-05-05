import torch
import torch.nn.functional as F
from torch import optim
import numpy as np

class Backprop(object):
    def __init__(self, net, step_size=0.001, loss='mse', opt='sgd', beta_1=0.9, beta_2=0.999, weight_decay=0.0,
                 to_perturb=False, perturb_scale=0.1, device='cpu', momentum=0, ef_lambda=0.01):
        self.net = net
        self.to_perturb = to_perturb
        self.perturb_scale = perturb_scale
        self.device = device
        self.ef_lambda = ef_lambda

        # define the optimizer
        if opt == 'sgd':
            self.opt = optim.SGD(self.net.parameters(), lr=step_size, weight_decay=weight_decay, momentum=momentum)
        elif opt == 'adam':
            self.opt = optim.Adam(self.net.parameters(), lr=step_size, betas=(beta_1, beta_2),
                                  weight_decay=weight_decay)
        elif opt == 'adamW':
            self.opt = optim.AdamW(self.net.parameters(), lr=step_size, betas=(beta_1, beta_2),
                                   weight_decay=weight_decay)

        # define the loss function
        self.loss = loss
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[self.loss]

        # Placeholder
        self.previous_features = None
    
    def effective_rank(self, m):
        sv = torch.linalg.svdvals(m)
        # print("Effective rank: ", sv)
        norm_sv = sv / torch.sum(torch.abs(sv))
        entropy = torch.tensor(0.0, dtype=torch.float32, device=sv.device)
        for p in norm_sv:
            if p > 0.0:
                entropy -= p * torch.log(p)

        effective_rank = torch.tensor(np.e) ** entropy
        return effective_rank.to(torch.float32)

    def _print_grads(self):
        for name, param in self.net.named_parameters():
            if param.grad is not None:
                # you can replace .norm() with param.grad.abs().mean(), etc.
                print(f"{name:40s} grad norm = {param.grad.norm().item():.6f}")
            else:
                print(f"{name:40s} grad is None")

    def learn(self, x, target):
        """
        Learn using one step of gradient-descent
        :param x: input
        :param target: desired output
        :return: loss
        """
        self.opt.zero_grad()
        output, features = self.net.predict(x)
        self.previous_features = features
        er_terms = [self.effective_rank(f) for f in features]
        # print("Effective rank terms: ", er_terms)
        # print("Effective rank terms: ", [f.item() for f in er_terms])
        ef_reg = torch.stack(er_terms).mean()
        loss_reg = - self.ef_lambda * ef_reg
        # loss_reg.backward()
        # print("=== Gradients after effective‐rank backward ===")
        # self._print_grads()
        # self.opt.step()

        # Phase 2: accuracy update
        loss = self.loss_func(output, target) + loss_reg
        loss.backward()
        # print("=== Gradients after accuracy backward ===")
        # self._print_grads()
        self.opt.step()

        if self.to_perturb:
            self.perturb()
        if self.loss == 'nll':
            return loss.detach(), output.detach()
        return loss.detach()

    def perturb(self):
        with torch.no_grad():
            for i in range(int(len(self.net.layers)/2)+1):
                self.net.layers[i * 2].bias +=\
                    torch.empty(self.net.layers[i * 2].bias.shape, device=self.device).normal_(mean=0, std=self.perturb_scale)
                self.net.layers[i * 2].weight +=\
                    torch.empty(self.net.layers[i * 2].weight.shape, device=self.device).normal_(mean=0, std=self.perturb_scale)