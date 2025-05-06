import torch
import torch.nn.functional as F
from torch import optim
import numpy as np
from collections import deque

class EffectiveRank(object):
    def __init__(self, net, step_size=0.001, loss='mse', opt='sgd', beta_1=0.9, beta_2=0.999, weight_decay=0.0,
                 to_perturb=False, perturb_scale=0.1, device='cpu', momentum=0, ef_lambda=0.01, rank_interval=100,):
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

        self.opt_er = optim.SGD(self.net.parameters(), lr=0.01, weight_decay=weight_decay, momentum=momentum)
        # define the loss function
        self.loss = loss
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[self.loss]
        self.x_buffer: deque[torch.Tensor] = deque(maxlen=rank_interval)
        self.rank_interval = rank_interval

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
        self.x_buffer.append(x.detach())
        self.previous_features = features
        # print("Effective rank terms: ", er_terms)
        # print("Effective rank terms: ", [f.item() for f in er_terms])
        # loss_reg.backward()
        # print("=== Gradients after effective‐rank backward ===")
        # self._print_grads()
        # self.opt.step()

        # Phase 2: accuracy update
        loss = self.loss_func(output, target)
        loss.backward()
        # print("=== Gradients after accuracy backward ===")
        # self._print_grads()
        self.opt.step()

        if len(self.x_buffer) == self.rank_interval:
            # stack → big batch (B=rank_interval × minibatch)
            big_x = torch.cat(list(self.x_buffer), dim=0)

            self.opt_er.zero_grad()                        # fresh grad pass
            _, feats = self.net.predict(big_x)          # recompute features

            er_terms = [self.effective_rank(f) for f in feats]
            ef_reg   = torch.stack(er_terms).mean()
            loss_er  = -self.ef_lambda * ef_reg         # maximise ER ⇒ min -ER
            loss_er.backward()
            self.opt_er.step()

            self.x_buffer.clear()  

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