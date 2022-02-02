import torch
import torch.nn.utils as torch_utils
import collections

from utils import *
# from models.utils import *
from data_loader.FastDataLoader import FastDataLoader
import trainers.lr_scheduler as lr_scheduler

from data_loader.datasets import Train_datasets
from data_loader.datasets import Test_datasets
from data_loader.data_sampler import DistIterSampler


class baseTrainer():
    def __init__(self, config):
        self.config = config
        self.is_train = config.is_train

        self.network = None
        self.results = collections.OrderedDict()

        self.schedulers = []
        self.optimizers = []

    def get_itr_per_epoch(self, state):
        if state == 'train':
            return len(self.data_loader_train) * self.itr_inc[state]
        else:
            return len(self.data_loader_eval) * self.itr_inc[state]

    def _set_optim(self, lr = None):
        if self.rank <= 0: print(toGreen('Building Optim...'))
        lr = self.config.lr_init if lr is None else lr

        self.optimizer = torch.optim.Adam([
            {'params': self.network.parameters(), 'lr': self.config.lr_init, 'lr_init': self.config.lr_init, 'betas':(0.9, 0.999), 'eps':1e-8}
            ], eps= 1e-8, lr=lr, betas=(self.config.beta1, 0.999))

        self.optimizers.append(self.optimizer)

    def _set_lr_scheduler(self):
        if self.rank <= 0: print(toGreen('Loading Learning Rate Scheduler...'))
        if self.config.LRS == 'CA':
            if self.rank <= 0: print(toRed('\tCosine annealing scheduler...'))
            for optimizer in self.optimizers:
                # if self.rank <= 0: print(toRed('\t\twarmup_itr: {}'.format(self.config.warmup_itr)))
                # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.total_itr-self.config.warmup_itr, self.config.lr_min)
                # self.schedulers.append(GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=self.config.warmup_itr, after_scheduler=scheduler_cosine))
                self.schedulers.append(
                lr_scheduler.CosineAnnealingLR_Restart(
                    optimizer, self.config.T_period, eta_min= self.config.eta_min,
                    restarts= self.config.restarts, weights= self.config.restart_weights))

        elif self.config.LRS == 'LD':
            if self.rank <= 0: print(toRed('\tLR dacay scheduler...'))
            for optimizer in self.optimizers:
                self.schedulers.append(
                    lr_scheduler.LR_decay(
                        optimizer, decay_period = self.config.decay_period,
                        decay_rate = self.config.decay_rate))

    def _set_dataloader(self, is_train=True):
        if self.rank <= 0: print(toGreen('Loading Data Loader...'))

        self.dataset_train = Train_datasets(self.config) if is_train else None
        self.dataset_eval = Test_datasets(self.config, is_valid = True if is_train else False)

        if self.config.dist == True:
            self.sampler_train = DistIterSampler(self.dataset_train, self.ws, self.rank) if is_train else None
            self.sampler_eval = DistIterSampler(self.dataset_eval, self.ws, self.rank, is_train=False)
        else:
            self.sampler_train = None
            self.sampler_eval = None

        self.data_loader_train = self._create_dataloader(self.dataset_train, sampler = self.sampler_train, is_train = True) if is_train else None
        self.data_loader_eval = self._create_dataloader(self.dataset_eval, sampler = self.sampler_eval, is_train = False)

    def _create_dataloader(self, dataset, sampler, is_train, wif = None, drop_last = False):
        if is_train:
            shuffle = False if self.config.dist else True

            data_loader = FastDataLoader(
                            dataset,
                            batch_size=self.config.batch_size if is_train else self.config.batch_size_test,
                            shuffle=shuffle,
                            num_workers=self.config.thread_num,
                            sampler=sampler,
                            drop_last=drop_last,
                            worker_init_fn = wif,
                            pin_memory=True)

        else:
            shuffle = False
            data_loader = torch.utils.data.DataLoader(
                            dataset,
                            batch_size=self.config.batch_size if is_train else self.config.batch_size_test,
                            shuffle=shuffle,
                            num_workers=self.config.thread_num,
                            sampler=sampler,
                            # sampler=None,
                            drop_last=False,
                            worker_init_fn = wif,
                            pin_memory=False)

        return data_loader

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warmup
        lr_groups_l: list for lr_groups. each for a optimizer"""
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler"""
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def _update_learning_rate(self, cur_itr):
        for scheduler in self.schedulers:
            scheduler.step()

        lrs = {}
        for i, optimizer in enumerate(self.optimizers):
            # lr = [v['lr'] for v in optimizer.param_groups]
            for j, v in enumerate(optimizer.param_groups):
                lrs['lr{}-{}'.format(i, j)] = v['lr']

        return lrs

    def _set_visuals(self, inputs, outs, errs):
        self.visuals = collections.OrderedDict()

    def get_network(self):
        return self.network

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def print_network(self):
        print(self.network)

    def get_training_state(self, epoch):
        """Save training state during training, which will be used for resuming"""
        state = {'epoch': epoch, 'itr': self.itr_global, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())

        return state

    def resume_training(self, resume_state):
        """Resume the optimizers and schedulers for training"""
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        self.itr_global = resume_state['itr']

        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
            # for g in self.optimizers[i].param_groups:
            #     g['lr'] = self.config.lr_init
            #     g['lr_init'] = self.config.lr_init
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)
            if 'MultiStepLR_Restart' in type(self.schedulers[i]).__name__:
                self.schedulers[i].reset_param(self.config.resetarts, self.config.restart_weights)
            elif 'CosineAnnealingLR_Restart' in type(self.schedulers[i]).__name__:
                self.schedulers[i].reset_param(self.config.T_period, self.config.restarts, self.config.restart_weights)
            elif 'LR_decay' in type(self.schedulers[i]).__name__:
                self.schedulers[i].reset_param(self.config.decay_period, self.config.decay_rate)
            elif 'LR_decay_progressive' in type(self.schedulers[i]).__name__:
                self.schedulers[i].reset_param(self.config.decay_period, self.config.decay_rate)
            elif 'GradualWarmupScheduler' in type(self.schedulers[i]).__name__:
                from warmup_scheduler import GradualWarmupScheduler
                scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizers[i], self.config.total_itr-self.config.warmup_itr, self.config.lr_min)
                self.schedulers[i] = GradualWarmupScheduler(self.optimizers[i], multiplier=1, total_epoch=self.config.warmup_itr, after_scheduler=scheduler_cosine)
                for j in range(self.itr_global['train']):
                    self.schedulers[i].step()

    def _update(self, errs):

        self.optimizer.zero_grad()
        errs['total'].backward()
        torch_utils.clip_grad_norm_(self.network.parameters(), self.config.gc)

        # total_norm = 0
        # for name, p in self.network.named_parameters():
        #     if p.grad != None:
        #         param_norm = p.grad.data.norm(2)
        #         total_norm += param_norm.item() ** 2
        #     elif p.requires_grad != False:
        #         if self.rank <= 0: print('grad_none: ', name)

        # total_norm = total_norm ** (1. / 2)

        self.optimizer.step()

        log = self._update_learning_rate(self.itr_global['train'])
        # log['gnorm'] = total_norm

        return log

    def _update_amp(self, errs):

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(errs['total']).backward()

        self.scaler.unscale_(self.optimizer)
        torch_utils.clip_grad_norm_(self.network.parameters(), self.config.gc)

        # total_norm = 0
        # for name, p in self.network.named_parameters():
        #     if p.grad != None:
        #         param_norm = p.grad.data.norm(2)
        #         total_norm += param_norm.item() ** 2
        #     elif p.requires_grad != False:
        #         if self.rank <= 0: print('grad_none: ', name)

        # total_norm = total_norm ** (1. / 2)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        log = self._update_learning_rate(self.itr_global['train'])
        # log['gnorm'] = total_norm

        return log