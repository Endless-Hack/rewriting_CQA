# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
from torch import nn

import os
import pickle
import copy
import random
from tqdm import tqdm
from datasets import Dataset
from models import CP, ComplEx, TransE, RESCAL, TuckER
from regularizers import F2, N3
from utils import avg_both, setup_optimizer, get_git_revision_hash, set_seed
import handle_rules as hr 
# import wandb

current_path = os.path.dirname(__file__)
type1_rule_ids, type2_rule_ids = hr.grounding_rule()

# 采样grounding rule负样本
def grounding_negative_sample(rules_list: list):
    with open(current_path + '/../../data/FB15k-betae/id2rel.pkl', 'rb') as f:
        id_to_rel = pickle.load(f)
    with open(current_path + '/../../data/FB15k-betae/id2ent.pkl', 'rb') as f:
        id_to_ent = pickle.load(f)
    rel_num = len(id_to_rel)
    ent_num = len(id_to_ent)

    neg_rules_list = copy.deepcopy(rules_list)
    # 随机替换一个rel
    for rule in neg_rules_list:
        rule[0][1] = random.randint(0, rel_num)

    return neg_rules_list

neg_type1_rules = grounding_negative_sample(type1_rule_ids)
neg_type2_rules = grounding_negative_sample(type2_rule_ids)
# type1_rule:1512
# type2_rule:35
rule1_heads = []
rule1_bodys = []
for rule in type1_rule_ids:
    rule1_heads.append(rule[0])
    rule1_bodys.append(rule[1])
# rule_heads = torch.tensor(rule_heads).long().cuda()
# rule_bodys = torch.tensor(rule_bodys).long().cuda()

neg_rule1_heads = []
neg_rule1_bodys = []
for rule in neg_type1_rules:
    neg_rule1_heads.append(rule[0])
    neg_rule1_bodys.append(rule[1])
# neg_rule_heads = torch.tensor(neg_rule_heads).long().cuda()
# neg_rule_bodys = torch.tensor(neg_rule_bodys).long().cuda()

rule2_heads = []
rule2_body1s = []
rule2_body2s = []
for rule in type2_rule_ids:
    rule2_heads.append(rule[0])
    rule2_body1s.append(rule[1])
    rule2_body2s.append(rule[2])

neg_rule2_heads = []
neg_rule2_body1s = []
neg_rule2_body2s = []
for rule in neg_type2_rules:
    neg_rule2_heads.append(rule[0])
    neg_rule2_body1s.append(rule[1])
    neg_rule2_body2s.append(rule[2])


def setup_ds(opt):
    dataset_opt = {k: v for k, v in opt.items() if k in ['dataset', 'device', 'cache_eval', 'reciprocal']}
    dataset = Dataset(dataset_opt)
    return dataset


def setup_model(opt):
    if opt['model'] == 'TransE':
        model = TransE(opt['size'], opt['rank'], opt['init'])
    elif opt['model'] == 'ComplEx':
        model = ComplEx(opt['size'], opt['rank'], opt['init'])
    elif opt['model'] == 'TuckER':
        model = TuckER(opt['size'], opt['rank'], opt['rank_r'], opt['init'], opt['dropout'])
    elif opt['model'] == 'RESCAL':
        model = RESCAL(opt['size'], opt['rank'], opt['init'])
    elif opt['model'] == 'CP':
        model = CP(opt['size'], opt['rank'], opt['init'])
    model.to(opt['device'])
    return model


def setup_loss(opt):
    if opt['world'] == 'sLCWA+bpr':
        loss = nn.BCEWithLogitsLoss(reduction='mean')
    elif opt['world'] == 'sLCWA+set':
        pass
    elif opt['world'] == 'LCWA':
        loss = nn.CrossEntropyLoss(reduction='mean')
    return loss


def setup_regularizer(opt):
    if opt['regularizer'] == 'F2':
        regularizer =  F2(opt['lmbda'])
    elif opt['regularizer'] == 'N3':
        regularizer = N3(opt['lmbda'])
    return regularizer


def _set_exp_alias(opt):
    suffix = '{}_{}_Rank{}_Reg{}_Lmbda{}_W{}'.format(opt['dataset'], opt['model'], opt['rank'], opt['regularizer'], opt['lmbda'], opt['w_rel'])
    alias = opt['alias'] + suffix
    return alias


def _set_cache_path(path_template, dataset, alias):
    if path_template is not None:
        cache_path = path_template.format(dataset=dataset, alias=alias)
        if not os.path.exists(cache_path):
            print("cache_path: ", cache_path)
            os.makedirs(cache_path, exist_ok=True)
    else:
        cache_path = None
    return cache_path


class KBCEngine(object):
    def __init__(self, opt):
        self.seed = opt['seed']
        set_seed(int(self.seed))
        self.alias = _set_exp_alias(opt)
        self.cache_eval = _set_cache_path(opt['cache_eval'], opt['dataset'], self.alias)
        self.model_cache_path = _set_cache_path(opt['model_cache_path'], opt['dataset'], self.alias)
        opt['cache_eval'] = self.cache_eval
        # dataset
        self.dataset = setup_ds(opt)
        opt['size'] = self.dataset.get_shape()
        # model
        self.model = setup_model(opt)
        self.optimizer = setup_optimizer(self.model, opt['optimizer'], opt['learning_rate'], opt['decay1'], opt['decay2'])
        self.loss = setup_loss(opt)
        opt['loss'] = self.loss
        self.batch_size = opt['batch_size']
        # regularizer
        self.regularizer = setup_regularizer(opt)
        self.device = opt['device']
        self.max_epochs = opt['max_epochs']
        self.world = opt['world']
        self.num_neg = opt['num_neg']
        self.score_rel = opt['score_rel']
        self.score_rhs = opt['score_rhs']
        self.score_lhs = opt['score_lhs']
        self.w_rel = opt['w_rel']
        self.w_lhs = opt['w_lhs']
        self.opt = opt
        self._epoch_id = 0
        self.writer = open(self.model_cache_path+"/train.log", "a+")
        '''
        wandb.init(project="ssl-relation-prediction", 
                    group=opt['experiment_id'], 
                    tags=opt['run_tags'],
                    notes=opt['run_notes'])
        wandb.config.update(opt)
        wandb.watch(self.model, log='all', log_freq=10000)
        wandb.run.summary['is_done'] = False
        '''
        # print('Git commit ID: {}'.format(get_git_revision_hash()))
        self.rule1_heads = rule1_heads
        self.rule1_bodys = rule1_bodys
        self.neg_rule1_heads = neg_rule1_heads
        self.neg_rule1_bodys = neg_rule1_bodys

        self.rule2_heads = rule2_heads
        self.rule2_body1s = rule2_body1s
        self.rule2_body2s = rule2_body2s
        self.neg_rule2_heads = neg_rule2_heads
        self.neg_rule2_body1s = neg_rule2_body1s
        self.neg_rule2_body2s = neg_rule2_body2s
        
    def episode(self):
        best_valid_mrr, init_epoch_id, step_idx = 0, 0, 0
        exp_train_sampler = self.dataset.get_sampler('train')
        
        for e in range(init_epoch_id, self.max_epochs):
            # wandb.run.summary['epoch_id'] = e
            self.model.train()
            print("-----Start training triples-----")
            pbar = tqdm(total=exp_train_sampler.size)
            while exp_train_sampler.is_epoch(e): # iterate through all batchs inside an epoch
                pbar.update(self.batch_size)
                if self.world == 'LCWA':
                    input_batch_train = exp_train_sampler.batchify(self.batch_size,
                                                                    self.device)
                    predictions, factors = self.model.forward(input_batch_train, score_rel=self.score_rel, score_rhs=self.score_rhs, score_lhs=self.score_lhs)
                    
                    if self.score_rel and self.score_rhs and self.score_lhs:
                        # print('----1----')
                        l_fit = self.loss(predictions[0], input_batch_train[:, 2]) \
                                + self.w_rel * self.loss(predictions[1], input_batch_train[:, 1]) \
                                + self.w_lhs * self.loss(predictions[2], input_batch_train[:, 0])
                    elif self.score_rel and self.score_rhs:
                        # print('----2----')
                        l_fit = self.loss(predictions[0], input_batch_train[:, 2]) + self.w_rel * self.loss(predictions[1], input_batch_train[:, 1])
                    elif self.score_lhs and self.score_rel:
                        # print('----3----')
                        pass
                    elif self.score_rhs and self.score_lhs: # standard
                        # print('----4----')
                        l_fit = self.loss(predictions[0], input_batch_train[:, 2]) + self.loss(predictions[1], input_batch_train[:, 0])
                    elif self.score_rhs: # only rhs
                        # print('----5----')
                        l_fit = self.loss(predictions, input_batch_train[:, 2])
                    elif self.score_rel:
                        # print('----6----')
                        l_fit = self.loss(predictions, input_batch_train[:, 1])
                    elif self.score_lhs:
                        # print('----7----')
                        pass
                    
                    l_reg, l_reg_raw, avg_lmbda = self.regularizer.penalty(input_batch_train, factors) # Note: this shouldn't be included into the computational graph of lambda update
                elif self.world == 'sLCWA+bpr':
                    pos_train, neg_train, label = exp_train_sampler.batchify(self.batch_size,
                                                                                self.device,
                                                                                num_neg=self.num_neg)
                    predictions, factors = self.model.forward_bpr(pos_train, neg_train)
                    l_fit = self.loss(predictions, label)
                    l_reg, l_reg_raw, avg_lmbda = self.regularizer.penalty(
                        torch.cat((pos_train, neg_train), dim=0),
                        factors)
                
                l = l_fit + l_reg
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                    
                '''
                if ((step_idx % 1000 == 0 and step_idx > 1000) or (step_idx <= 1000 and step_idx % 100 == 0)): # reduce logging frequency to accelerate 
                    wandb.log({'step_wise/train/l': l.item()}, step=step_idx)
                    wandb.log({'step_wise/train/l_fit': l_fit.item()}, step=step_idx)
                    wandb.log({'step_wise/train/l_reg': l_reg.item()}, step=step_idx)
                    wandb.log({'step_wise/train/l_reg_raw': l_reg_raw.item()}, step=step_idx)
                '''
                step_idx += 1

            print("-----Start training type1 rules-----")
            rule1_loss = nn.MarginRankingLoss(margin=0.2, reduction='mean')
            with tqdm(total=len(type1_rule_ids)) as bar:
                # 获取负样本
                bar.set_description('rule1 loss')

                rule1_heads = torch.tensor(self.rule1_heads).long().cuda()
                rule1_bodys = torch.tensor(self.rule1_bodys).long().cuda()

                neg_rule1_heads = torch.tensor(self.neg_rule1_heads).long().cuda()
                neg_rule1_bodys = torch.tensor(self.neg_rule1_bodys).long().cuda()
                rule1_begin = 0
                grounding1_batch = 5

                # grounding type1 rule训练
                while rule1_begin < len(type1_rule_ids):
                    if rule1_begin + grounding1_batch < len(type1_rule_ids):
                        rule_head_batch = rule1_heads[  # tensor:[100, 3]
                                        rule1_begin: rule1_begin + grounding1_batch]
                        rule_body_batch = rule1_bodys[  # tensor:[100, 3]
                                        rule1_begin: rule1_begin + grounding1_batch]
                        neg_rule_head_batch = neg_rule1_heads[  # tensor:[100, 3]
                                            rule1_begin: rule1_begin + grounding1_batch]
                        neg_rule_body_batch = neg_rule1_bodys[  # tensor:[100, 3]
                                            rule1_begin: rule1_begin + grounding1_batch]
                    else:
                        rule_head_batch = rule1_heads[  # tensor:[100, 3]
                                        rule1_begin: -1]
                        rule_body_batch = rule1_bodys[  # tensor:[100, 3]
                                        rule1_begin: -1]
                        neg_rule_head_batch = neg_rule1_heads[  # tensor:[100, 3]
                                            rule1_begin: -1]
                        neg_rule_body_batch = neg_rule1_bodys[  # tensor:[100, 3]
                                            rule1_begin: -1]

                    # 负样本采样
                    neg_head_truth = self.triple_truth_value(neg_rule_head_batch)
                    neg_body_truth = self.triple_truth_value(neg_rule_body_batch)
                    body_truth = self.triple_truth_value(rule_body_batch)
                    head_truth = self.triple_truth_value(rule_head_batch)
                    # print(head_truth)
                    # t-norm
                    rule_truth = self.t_norm_equation(head_truth, body_truth)
                    rule_neg_truth = self.t_norm_equation(neg_head_truth, neg_body_truth)
                    # print('after t-norm', rule_neg_truth.shape) #(batch_size,1)

                    y = torch.full_like(body_truth, fill_value=1)
                    l_sum = rule1_loss(rule_truth, rule_neg_truth, y)

                    self.optimizer.zero_grad()
                    l_sum.backward()
                    self.optimizer.step()

                    rule1_begin += grounding1_batch

                    # 进度条
                    bar.update(grounding1_batch)
                    bar.set_postfix(loss=f'{l_sum.item():.5f}')
                bar.close()

            print("-----Start training type2 rules-----")
            rule2_loss = nn.MarginRankingLoss(margin=0.2, reduction='mean')
            with tqdm(total=len(type2_rule_ids)) as bar:
                bar.set_description('rule2 loss')

                rule2_heads = torch.tensor(self.rule2_heads).long().cuda()
                rule2_body1s = torch.tensor(self.rule2_body1s).long().cuda()
                rule2_body2s = torch.tensor(self.rule2_body2s).long().cuda()
                neg_rule2_heads = torch.tensor(self.neg_rule2_heads).long().cuda()
                neg_rule2_body1s = torch.tensor(self.neg_rule2_body1s).long().cuda()
                neg_rule2_body2s = torch.tensor(self.neg_rule2_body2s).long().cuda()

                rule2_begin = 0
                grounding2_batch = 1

                # grounding type2 rule训练
                while rule2_begin < len(type2_rule_ids):
                    if rule2_begin <= len(type2_rule_ids):
                        rule_head_batch = rule2_heads[  # tensor:[100, 3]
                                        rule2_begin:rule2_begin + grounding2_batch]
                        rule_body1_batch = rule2_body1s[  # tensor:[100, 3]
                                        rule2_begin:rule2_begin + grounding2_batch]
                        rule_body2_batch = rule2_body2s[  # tensor:[100, 3]
                                        rule2_begin:rule2_begin + grounding2_batch]
                        neg_rule_head_batch = neg_rule2_heads[  # tensor:[100, 3]
                                            rule2_begin:rule2_begin + grounding2_batch]
                        neg_rule_body1_batch = neg_rule2_body1s[  # tensor:[100, 3]
                                            rule2_begin:rule2_begin + grounding2_batch]
                        neg_rule_body2_batch = neg_rule2_body2s[  # tensor:[100, 3]
                                            rule2_begin:rule2_begin + grounding2_batch]
                    else:
                        rule_head_batch = rule2_heads[  # tensor:[100, 3]
                                          rule2_begin: -1]
                        rule_body1_batch = rule2_body1s[  # tensor:[100, 3]
                                           rule2_begin: -1]
                        rule_body2_batch = rule2_body2s[  # tensor:[100, 3]
                                           rule2_begin: -1]
                        neg_rule_head_batch = neg_rule2_heads[  # tensor:[100, 3]
                                              rule2_begin: -1]
                        neg_rule_body1_batch = neg_rule2_body1s[  # tensor:[100, 3]
                                               rule2_begin: -1]
                        neg_rule_body2_batch = neg_rule2_body2s[  # tensor:[100, 3]
                                               rule2_begin: -1]

                    # 负样本采样
                    neg_head_truth = self.triple_truth_value(neg_rule_head_batch)
                    neg_body1_truth = self.triple_truth_value(neg_rule_body1_batch)
                    neg_body2_truth = self.triple_truth_value(neg_rule_body2_batch)
                    head_truth = self.triple_truth_value(rule_head_batch)
                    body1_truth = self.triple_truth_value(rule_body1_batch)
                    body2_truth = self.triple_truth_value(rule_body2_batch)

                    # print(neg_head_truth.shape)     #(batch_size, 1)
                    # t-norm
                    rule_truth = self.t_norm_equation(head_truth, self.t_norm(body1_truth, body2_truth))
                    rule_neg_truth = self.t_norm_equation(neg_head_truth,
                                                        self.t_norm(neg_body1_truth, neg_body2_truth))

                    y = torch.full_like(body1_truth, fill_value=1)
                    l_sum = rule2_loss(rule_truth, rule_neg_truth, y)

                    self.optimizer.zero_grad()
                    # 异常检测开启
                    # torch.autograd.set_detect_anomaly(True)
                    # # 反向传播时检测是否有异常值，定位code
                    # with torch.autograd.detect_anomaly():
                    #     l_sum.backward()
                    l_sum.backward()
                    self.optimizer.step()

                    rule2_begin += grounding2_batch

                    # 进度条
                    bar.update(grounding2_batch)
                    bar.set_postfix(loss=f'{l_sum.item():.5f}')
                bar.close()

            if e % self.opt['valid'] == 0:
                self.model.eval()
                res_all, res_all_detailed = [], []
                for split in self.dataset.splits:
                    res_s = self.dataset.eval(model=self.model, 
                                              split=split, 
                                              n_queries=-1 if split != 'train' else 1000, # subsample 5000 triples for computing approximated training MRR
                                              n_epochs=e)
                    res_all.append(avg_both(res_s[0], res_s[1]))
                    res_all_detailed.append(res_s[2])
                    
                res = dict(zip(self.dataset.splits, res_all))
                res_detailed = dict(zip(self.dataset.splits, res_all_detailed))
                
                print("\t Epoch: ", e)
                self.writer.write("Epoch: "+str(e)+"\n")
                for split in self.dataset.splits:
                    print("\t {}: {}".format(split.upper(), res[split]))
                    self.writer.write("{}: {}\n".format(split.upper(), res[split]))
                    '''
                    wandb.log({'step_wise/{}/mrr'.format(split): res[split]['MRR']}, step=step_idx)
                    wandb.log({'step_wise/{}/hits@1'.format(split): res[split]['hits@[1,3,10]'][0]}, step=step_idx)
                    '''
                split = 'meta_valid' if 'meta_valid' in self.dataset.splits else 'valid'
                self.model.checkpoint(model_cache_path=self.model_cache_path, epoch_id=str(e))
                if res[split]['MRR'] > best_valid_mrr:
                    best_valid_mrr = res[split]['MRR']
                    self.model.checkpoint(model_cache_path=self.model_cache_path, epoch_id='best_valid')
                    if self.opt['cache_eval'] is not None:
                        for s in self.dataset.splits:
                            for m in ['lhs', 'rhs']:
                                torch.save(res_detailed[s][m], 
                                           self.opt['cache_eval']+'{s}_{m}.pt'.format(s=s, m=m))
                    '''
                    wandb.run.summary['best_valid_mrr'] = best_valid_mrr
                    wandb.run.summary['best_valid_epoch'] = e
                    wandb.run.summary['corr_test_mrr'] = res['test']['MRR']
                    wandb.run.summary['corr_test_hits@1'] = res['test']['hits@[1,3,10]'][0]
                    wandb.run.summary['corr_test_hits@3'] = res['test']['hits@[1,3,10]'][1]
                    wandb.run.summary['corr_test_hits@10'] = res['test']['hits@[1,3,10]'][2]
                    '''
            if best_valid_mrr == 1:
                print('MRR 1, diverged!')
                break
            if best_valid_mrr > 0 and best_valid_mrr < 2e-4:
                if l_reg_raw.item() < 1e-4:
                    print('0 embedding weight, diverged!')
                    break

        self.model.eval()
        mrrs, hits, _ = self.dataset.eval(self.model, 'test', -1)
        print("\n\nTEST : MRR {} Hits {}".format(mrrs, hits))
        self.writer.write("Best valid mrr: "+str(best_valid_mrr))
        # wandb.run.summary['is_done'] = True
    
    def triple_truth_value(self, batch_example: torch.LongTensor):
        # predictions为预测得到的尾实体：[100,14951]
        # factors用来做正则化: [3,100,1000]
        # predictions, factors = self.model.forward(input_batch, neg_batch)
        score = self.model.score(batch_example)

        # truth = batch_example[:, 2]
        # l_fit = loss(predictions, truth)
        # l_reg = self.regularizer.forward(factors)
        # l = l_fit + l_reg
        l = torch.sigmoid(score)
        return l

    def rule_truth_value(self, rule: list):
        pass

    def t_norm(self, tens_1: torch.Tensor, tens_2: torch.Tensor) -> torch.Tensor:
        # print('In t-norm',(tens_1*tens_2).shape) #(batch_size,1)
        return tens_1 * tens_2

    def t_norm_equation(self, tens_1: torch.Tensor, tens_2: torch.Tensor) -> torch.Tensor:
        y = torch.full_like(tens_1, fill_value=1)
        return tens_1 * tens_2 - tens_2 + y