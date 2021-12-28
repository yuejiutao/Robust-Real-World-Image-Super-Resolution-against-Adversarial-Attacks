import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

def make_model(**kwargs):
    return IFGSM(model=kwargs['srmodel'],modelname=kwargs['modelname'])


class IFGSM(nn.Module):
    def __init__(self, model ,modelname):
        super().__init__()
        self.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
        self.model = model.to(self.device)  
        self.modelname=modelname
        self.loss_func = nn.MSELoss().to(self.device)
        # self.loss_func = nn.L1Loss().to(self.device)

    def generate(self, x, GT, **params):
        from datetime import datetime
        self.parse_params(**params)
        labels = GT
        adv_x = self.attack(x, labels)

        return adv_x

    def parse_params(self, scala=4 ,eps=8.0, iter_eps=8.0, nb_iter=50, clip_min=0.0, clip_max=255.0, C=0.0,
                     clip_eps_min=-8.0, clip_eps_max=8.0,
                     y=None, ord=np.inf, rand_init=False, flag_target=False):
        self.scala=4
        self.eps = eps
        self.iter_eps = iter_eps
        self.nb_iter = nb_iter
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.y = y
        self.ord = ord
        self.rand_init = rand_init
        self.flag_target = flag_target
        self.clip_eps_min = clip_eps_min
        self.clip_eps_max = clip_eps_max
        self.C = C

    def sigle_step_attack(self, x, pertubation, labels):
        adv_x = x + pertubation
        # get the gradient of x
        adv_x = Variable(adv_x)
        adv_x.requires_grad = True

        if self.modelname=='CDC_MC':
            preds,_ ,judloss,Flag=self.model(adv_x)
            preds=preds[-1]
        else:
            preds = self.model(adv_x)

        if self.flag_target:
            loss = -self.loss_func(preds, labels)
        else:
            loss = self.loss_func(preds, labels)

        self.model.zero_grad()
        loss.backward()


        grad = adv_x.grad.data
        pertubation = self.iter_eps / self.nb_iter * torch.sign(grad)

        adv_x=adv_x.detach()+pertubation
        x=x.detach()
        pertubation =torch.clamp(adv_x-x,self.clip_eps_min,self.clip_eps_max)

        return pertubation

    def attack(self, x, labels):
        self.model.eval()

        if self.rand_init:
            x_tmp = x + torch.Tensor(np.random.uniform(-self.eps, self.eps, x.shape)).type_as(x).cuda()
            print("*** init with a random noise ")
        else:
            x_tmp = x

        pertubation = torch.zeros(x.shape).type_as(x).to(self.device)

        for i in range(self.nb_iter):
            pertubation = self.sigle_step_attack(x_tmp, pertubation=pertubation, labels=labels)
            

        adv_x = x + pertubation
        adv_x = torch.clamp(adv_x, self.clip_min, self.clip_max)

        return adv_x