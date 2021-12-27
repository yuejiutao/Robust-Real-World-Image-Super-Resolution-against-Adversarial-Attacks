import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


def make_model(**kwargs):
    return IFGSM(model=kwargs['srmodel'], modelname=kwargs['modelname'])


class IFGSM(nn.Module):
    def __init__(self, model, modelname):
        super().__init__()
        self.model = model  # 必须是pytorch的model
        self.modelname = modelname
        self.device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

    def generate(self, x, GT, pert, **params):
        self.parse_params(**params)
        labels = GT
        adv_x = self.attack(x, labels, pert)
        return adv_x

    def parse_params(self, scala=4, eps=8.0, iter_eps=8.0, nb_iter=50, clip_min=0.0, clip_max=255.0, C=0.0,
                     clip_eps_min=-8.0, clip_eps_max=8.0,
                     y=None, ord=np.inf, rand_init=False, flag_target=False):
        self.scala = 4
        self.eps = eps
        self.iter_eps = iter_eps
        self.nb_iter = nb_iter
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.y = y
        self.ord = ord
        self.rand_init = rand_init
        self.model.to(self.device)
        self.flag_target = flag_target
        self.clip_eps_min = clip_eps_min
        self.clip_eps_max = clip_eps_max
        self.C = C

    def sigle_step_attack(self, x0, x_now, labels):
        adv_x = x_now
        adv_x = Variable(adv_x)
        adv_x.requires_grad = True
        # loss_func = nn.MSELoss()
        loss_func = nn.L1Loss()

        if self.modelname == 'HGSR-MHR':
            preds, SR_maps = self.model(adv_x)
            preds = preds[-1]
        else:
            preds = self.model(adv_x)

        if self.flag_target:
            loss = -loss_func(preds, labels)
        else:
            loss = loss_func(preds, labels)
            # label_mask=torch_one_hot(labels)
            #
            # correct_logit=torch.mean(torch.sum(label_mask * preds,dim=1))
            # wrong_logit = torch.mean(torch.max((1 - label_mask) * preds, dim=1)[0])
            # loss=-F.relu(correct_logit-wrong_logit+self.C)

        self.model.zero_grad()
        loss.backward()
        grad = adv_x.grad.data

        # get the pertubation of an iter_eps
        pertubation = self.iter_eps / self.nb_iter * np.sign(grad.cpu())
        adv_x = adv_x.cpu().detach().numpy() + pertubation.cpu().numpy()
        # now ,the adv_x is x(n+1)
        adv_x = np.clip(adv_x, self.clip_min, self.clip_max)

        x0 = x0.cpu().detach().numpy()
        pertubation = np.clip(adv_x - x0, self.clip_eps_min, self.clip_eps_max) - x0
        # pertubation = clip_pertubation(pertubation, self.ord, self.eps)

        return adv_x, pertubation

        # adv_x=x+pertubation
        # # get the gradient of x
        # adv_x=Variable(adv_x)
        # adv_x.requires_grad = True
        #
        # loss_func=nn.CrossEntropyLoss()
        # preds=self.model(adv_x)
        # if self.flag_target:
        #     loss =-loss_func(preds,labels)
        # else:
        #     loss=loss_func(preds,labels)
        #     # label_mask=torch_one_hot(labels)
        #     #
        #     # correct_logit=torch.mean(torch.sum(label_mask * preds,dim=1))
        #     # wrong_logit = torch.mean(torch.max((1 - label_mask) * preds, dim=1)[0])
        #     # loss=-F.relu(correct_logit-wrong_logit+self.C)
        #
        # self.model.zero_grad()
        # loss.backward()
        # grad=adv_x.grad.data
        # #get the pertubation of an iter_eps
        # pertubation=self.iter_eps*np.sign(grad)
        # adv_x=adv_x.cpu().detach().numpy()+pertubation.cpu().numpy()
        # x=x.cpu().detach().numpy()
        #
        # pertubation=np.clip(adv_x,self.clip_min,self.clip_max)-x
        # pertubation=clip_pertubation(pertubation,self.ord,self.eps)
        #
        #
        # return pertubation

    def attack(self, x, pert, labels):
        labels = labels.to(self.device)
        x = x.to(self.device)
        self.model = self.model.to(self.device)
        self.model.eval()

        # if self.rand_init:
        #     x_tmp = x + torch.Tensor(np.random.uniform(-self.eps, self.eps, x.shape)).type_as(x).cuda()
        # else:
        #     x_tmp = x
        pertubation = torch.zeros(x.shape).type_as(x).to(self.device)

        x_now = x_tmp

        # print("*** x_now device ",x_now.device)
        # print("*** type xnow",type(x_now))

        for i in range(self.nb_iter):
            x_now, pertubation = self.sigle_step_attack(x, x_now, labels=labels)
            x_now = torch.from_numpy(x_now)
            x_now = x_now.to(self.device)
            # print("*** xnow device and type ",x_now.device,type(x_now))

            # pertubation=self.sigle_step_attack(x_tmp,pertubation=pertubation,labels=labels)
            # pertubation=torch.Tensor(pertubation).type_as(x).to(self.device)
        # adv_x=x+pertubation
        # adv_x=adv_x.cpu().detach().numpy()

        adv_x = x_now
        adv_x = adv_x.cpu().detach().numpy()
        adv_x = np.clip(adv_x, self.clip_min, self.clip_max)

        adv_x = torch.from_numpy(adv_x)
        return adv_x
