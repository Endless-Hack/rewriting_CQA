# -*-coding : utf-8-*-
# __author__ : Administrator
# date : 2023/2/5

import pickle
import random
import os

current_path = os.path.dirname(__file__)


def grounding_rule():
    type1_rules, type2_rules = load_rules()

    with open(current_path+'/../../data/FB15k-betae/id2rel.pkl','rb') as f:
        id_to_rel = pickle.load(f)
    with open(current_path+'/../../data/FB15k-betae/id2ent.pkl','rb') as f:
        id_to_ent = pickle.load(f)

    type1_rule_ids = []
    type2_rule_ids = []

    for rule in type1_rules:
        rule = rule.split('<=')
        rule_triples = []
        entity_map = dict()
        for i in range(len(rule)):
            rule_body = rule[i].strip()
            start = rule_body.find('(')
            end = rule_body.find(')')
            if start==-1 or end == -1:
                continue
            rule_entities = rule_body[start+1 : end].split(',')
            rule_rel = [k for k,v in id_to_rel.items() if v==("+" + rule_body[0 : start])][0]
            rule_triples.append([rule_entities[0], rule_rel, rule_entities[1]])
            # 维护一个哈希map 实体 -> id
            for i in range(len(rule_entities)):
                if entity_map.get(rule_entities[i]) == None:
                    # 包含固定实体的规则
                    if '/m' in rule_entities[i]:
                        entity_map[rule_entities[i]] = [k for k, v in id_to_ent.items() if v == rule_entities[i]][0]
                    else:  # grounding 只有一个
                        entity_map[rule_entities[i]] = random.randint(0, len(id_to_ent))
        rule_triples[0][0] = entity_map[rule_triples[0][0]]
        rule_triples[0][2] = entity_map[rule_triples[0][2]]
        rule_triples[1][0] = entity_map[rule_triples[1][0]]
        rule_triples[1][2] = entity_map[rule_triples[1][2]]
        type1_rule_ids.append(rule_triples)

    for rule in type2_rules:
        rule = rule.split('<=')
        rule_body = rule[1].split(', ')
        rule = [rule[0], rule_body[0], rule_body[1]]
        rule_triples = []
        entity_map = dict()
        for i in range(len(rule)):
            rule_body = rule[i].strip()
            start = rule_body.find('(')
            end = rule_body.find(')')
            if start == -1 or end == -1:
                continue
            rule_entities = rule_body[start + 1: end].split(',')
            rule_rel = [k for k, v in id_to_rel.items() if v == ("+" + rule_body[0: start])][0]
            rule_triples.append([rule_entities[0], rule_rel, rule_entities[1]])
            for i in range(len(rule_entities)):
                if entity_map.get(rule_entities[i]) == None:
                    if '/m' in rule_entities[i]:
                        entity_map[rule_entities[i]] = [k for k, v in id_to_ent.items() if v == rule_entities[i]][0]
                    else:
                        entity_map[rule_entities[i]] = random.randint(0, len(id_to_ent))   
        rule_triples[0][0] = entity_map[rule_triples[0][0]]
        rule_triples[0][2] = entity_map[rule_triples[0][2]]
        rule_triples[1][0] = entity_map[rule_triples[1][0]]
        rule_triples[1][2] = entity_map[rule_triples[1][2]]
        rule_triples[2][0] = entity_map[rule_triples[2][0]]
        rule_triples[2][2] = entity_map[rule_triples[2][2]]
        type2_rule_ids.append(rule_triples)

    # del type1_rule_ids[1512]
    # del type2_rule_ids[35]
    # print(type1_rule_ids)
    # print(len(type1_rule_ids))

    return type1_rule_ids, type2_rule_ids

def load_rules():
    path1 = '/rule/FB15k_type1_rules.txt'
    path2 = '/rule/FB15k_type2_rules.txt'

    with open(current_path+path1, 'r') as f:
        type1_rules = f.readlines()

    with open(current_path+path2, 'r') as f:
        type2_rules = f.readlines()

    return type1_rules, type2_rules

if __name__ == '__main__':
    grounding_rule()