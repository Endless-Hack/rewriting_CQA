import pickle
import os
from collections import defaultdict
import argparse
import time
# import matplotlib
import matplotlib.pyplot as plt

# 设置全局字体为 SimHei（黑体）
plt.rcParams['font.sans-serif'] = ['AaGuDianKeBenSong-2']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# print(matplotlib.get_cachedir())

# hyper-parameters
rewriting_depth = 10
conf_shreshold = 0.95
# 重写后的查询
rewriting_queries = defaultdict(set)
# 规则集合
rule_list = []  # (-)rule_body -> rule_head, conf
# 规则字典
rule_dict = defaultdict(list)
# 执行平均时间
avg_execute_time = []
# 查询类型
query_types = ['1p','2p','3p','2i','3i','ip','pi','2u','up','inp','pin','pni','2in','3in']

# A(X, Y) -> B(Y, X)类型的规则转换为 (X, Y) -> (X, Y)
def handle_rules(rules):
    rule_bodies = []
    rule_heads = [] 
    tmp_list = []
    for rule in rules:
        conf = rule.split("\t")[0]
        rule_head = rule.split("\t")[-1].split("<=")[0].strip() # (X,Y)
        rule_body = rule.split("\t")[-1].split("<=")[1].strip() # (X,Y) or (Y,X)
        rule_bodies.append(rule_body)
        rule_heads.append(rule_head)        
        # print(rule_body, "->", rule_head)   
        rule_body_ents = rule_body.strip().split("(")[-1]
        rule_body_pred = rule_body.strip().split("(")[0]
        rule_head_ents = rule_head.strip().split("(")[-1]
        rule_head_pred = rule_head.strip().split("(")[0]
        # implication规则
        if rule_head_ents == rule_body_ents:
            if rule_head_pred != rule_body_pred:
                if [eval(rule_body_pred), eval(rule_head_pred)] not in tmp_list and eval(conf) >= conf_shreshold:
                    tmp_list.append([eval(rule_body_pred), eval(rule_head_pred)])
                    rule_list.append([eval(rule_body_pred), eval(rule_head_pred), eval(conf)])
                if eval(rule_head_pred)%2==1 and eval(rule_body_pred)%2==1:
                    if [eval(rule_body_pred)-1, eval(rule_head_pred)-1] not in tmp_list and eval(conf) >= conf_shreshold:
                        tmp_list.append([eval(rule_body_pred)-1, eval(rule_head_pred)-1])
                        rule_list.append([eval(rule_body_pred)-1, eval(rule_head_pred)-1, eval(conf)])
                elif eval(rule_head_pred)%2==1 and eval(rule_body_pred)%2==0:
                    if [eval(rule_body_pred)+1, eval(rule_head_pred)-1] not in tmp_list and eval(conf) >= conf_shreshold:
                        tmp_list.append([eval(rule_body_pred)+1, eval(rule_head_pred)-1])
                        rule_list.append([eval(rule_body_pred)+1, eval(rule_head_pred)-1, eval(conf)])
                elif eval(rule_head_pred)%2==0 and eval(rule_body_pred)%2==0:
                    if [eval(rule_body_pred)+1, eval(rule_head_pred)+1] not in tmp_list and eval(conf) >= conf_shreshold:
                        tmp_list.append([eval(rule_body_pred)+1, eval(rule_head_pred)+1])
                        rule_list.append([eval(rule_body_pred)+1, eval(rule_head_pred)+1, eval(conf)])
                elif eval(rule_head_pred)%2==0 and eval(rule_body_pred)%2==1:
                    if [eval(rule_body_pred)-1, eval(rule_head_pred)+1] not in tmp_list and eval(conf) >= conf_shreshold:
                        tmp_list.append([eval(rule_body_pred)-1, eval(rule_head_pred)+1])
                        rule_list.append([eval(rule_body_pred)-1, eval(rule_head_pred)+1, eval(conf)])
        # inverse规则 A(X,Y) -> B(Y,X)
        else:
            if rule_head_pred != rule_body_pred:
                # 71 -> 339
                if eval(rule_head_pred)%2==1 and eval(rule_body_pred)%2==1:
                    if [eval(rule_body_pred), eval(rule_head_pred)-1] not in tmp_list and eval(conf) >= conf_shreshold:
                        tmp_list.append([eval(rule_body_pred), eval(rule_head_pred)-1])
                        rule_list.append([eval(rule_body_pred), eval(rule_head_pred)-1, eval(conf)])
                    if [eval(rule_body_pred)-1, eval(rule_head_pred)] not in tmp_list and eval(conf) >= conf_shreshold:
                        tmp_list.append([eval(rule_body_pred)-1, eval(rule_head_pred)])
                        rule_list.append([eval(rule_body_pred)-1, eval(rule_head_pred), eval(conf)])
                # 70 -> 339
                elif(eval(rule_head_pred)%2==1 and eval(rule_body_pred)%2==0):
                    if [eval(rule_body_pred), eval(rule_head_pred)-1] not in tmp_list and eval(conf) >= conf_shreshold:
                        tmp_list.append([eval(rule_body_pred), eval(rule_head_pred)-1])
                        rule_list.append([eval(rule_body_pred), eval(rule_head_pred)-1, eval(conf)])
                    if [eval(rule_body_pred)+1, eval(rule_head_pred)] not in tmp_list and eval(conf) >= conf_shreshold:
                        tmp_list.append([eval(rule_body_pred)+1, eval(rule_head_pred)])
                        rule_list.append([eval(rule_body_pred)+1, eval(rule_head_pred), eval(conf)])
                # 70 -> 338
                elif(eval(rule_head_pred)%2==0 and eval(rule_body_pred)%2==0):
                    if [eval(rule_body_pred), eval(rule_head_pred)+1] not in tmp_list and eval(conf) >= conf_shreshold:
                        tmp_list.append([eval(rule_body_pred), eval(rule_head_pred)+1])
                        rule_list.append([eval(rule_body_pred), eval(rule_head_pred)+1, eval(conf)])
                    if [eval(rule_body_pred)+1, eval(rule_head_pred)] not in tmp_list and eval(conf) >= conf_shreshold:
                        tmp_list.append([eval(rule_body_pred)+1, eval(rule_head_pred)])
                        rule_list.append([eval(rule_body_pred)+1, eval(rule_head_pred), eval(conf)])
                # 71 -> 338
                else:
                    if [eval(rule_body_pred), eval(rule_head_pred)+1] not in tmp_list and eval(conf) >= conf_shreshold:
                        tmp_list.append([eval(rule_body_pred), eval(rule_head_pred)+1])
                        rule_list.append([eval(rule_body_pred), eval(rule_head_pred)+1, eval(conf)])
                    if [eval(rule_body_pred)-1, eval(rule_head_pred)] not in tmp_list and eval(conf) >= conf_shreshold:
                        tmp_list.append([eval(rule_body_pred)-1, eval(rule_head_pred)])
                        rule_list.append([eval(rule_body_pred)-1, eval(rule_head_pred), eval(conf)])

def make_rule_dict():
    for rule in rule_list:
        rule_body = rule[0]
        rule_head = rule[1]
        rule_conf = rule[-1]
        rule_dict[rule_head].append((rule_body, rule_conf))

def rewrite_queries(dataset):
    with open("%s/test-queries.pkl"%dataset, "rb") as f:
        test_queries = pickle.load(f)

        queries_1p = test_queries[('e', ('r',))]    #set 集合
        rewrite_queries_1p(queries_1p, ('e', ('r',)))
        queries_2p = test_queries[('e', ('r', 'r'))]
        rewrite_queries_2p(queries_2p, ('e', ('r', 'r')))
        queries_3p = test_queries[('e', ('r', 'r', 'r'))]
        rewrite_queries_3p(queries_3p, ('e', ('r', 'r', 'r')))
        queries_2i = test_queries[(('e', ('r',)), ('e', ('r',)))]
        rewrite_queries_2i(queries_2i, (('e', ('r',)), ('e', ('r',))))
        queries_3i = test_queries[(('e', ('r',)), ('e', ('r',)), ('e', ('r',)))]
        rewrite_queries_3i(queries_3i, (('e', ('r',)), ('e', ('r',)), ('e', ('r',))))
        queries_ip = test_queries[((('e', ('r',)), ('e', ('r',))), ('r',))]
        rewrite_queries_ip(queries_ip, ((('e', ('r',)), ('e', ('r',))), ('r',)))
        queries_pi = test_queries[(('e', ('r', 'r')), ('e', ('r',)))]
        rewrite_queries_pi(queries_pi, (('e', ('r', 'r')), ('e', ('r',))))
        queries_2u = test_queries[(('e', ('r',)), ('e', ('r',)), ('u',))]
        rewrite_queries_2u(queries_2u, (('e', ('r',)), ('e', ('r',)), ('u',)))
        queries_up = test_queries[((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))]
        rewrite_queries_up(queries_up, ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)))
        queries_inp = test_queries[((('e', ('r',)), ('e', ('r', 'n'))), ('r',))]
        rewrite_queries_inp(queries_inp, ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)))
        queries_pin = test_queries[(('e', ('r', 'r')), ('e', ('r', 'n')))]
        rewrite_queries_pin(queries_pin, (('e', ('r', 'r')), ('e', ('r', 'n'))))
        queries_pni = test_queries[(('e', ('r', 'r', 'n')), ('e', ('r',)))]
        rewrite_queries_pni(queries_pni, (('e', ('r', 'r', 'n')), ('e', ('r',))))
        queries_2in = test_queries[(('e', ('r',)), ('e', ('r', 'n')))]
        rewrite_queries_2in(queries_2in, (('e', ('r',)), ('e', ('r', 'n'))))
        queries_3in = test_queries[(('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')))]
        rewrite_queries_3in(queries_3in, (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))))
    
def rewrite_queries_1p(queries_1p, query_structure):
    print("-----handling 1p-----")
    start_time = time.time()
    nums = float(len(queries_1p))
    print("1p 查询的个数为：", nums)
    for query in queries_1p:
        # print(query)
        rewriting_queries[query] = {(query, 1.0, query_structure)}
        unique_queries = {query}
        last_rewriting = []
        this_rewriting = [(query, 1.0)]
        while len(rewriting_queries[query]) < rewriting_depth:
            last_rewriting = this_rewriting
            this_rewriting = []
            for each_query in last_rewriting:
                if len(rewriting_queries[query]) >= rewriting_depth:
                    break
                rel = each_query[0][1][0]
                ent = each_query[0][0]
                query_conf = each_query[-1]
                # 哈希表优化
                for each_rule in rule_dict[rel]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if (ent, (body_pred,)) not in unique_queries:
                        rewriting_queries[query].add(((ent, (body_pred,)), query_conf*rule_conf, query_structure))
                        this_rewriting.append(((ent, (body_pred,)), query_conf*rule_conf))
                        unique_queries.add((ent, (body_pred,)))
            if(len(this_rewriting) == 0):
                break
        if len(rewriting_queries[query]) > 1:
            print(rewriting_queries[query])
    end_time = time.time()
    exe_time = end_time - start_time
    print("1p 重写执行平均时间：", exe_time/nums)
    avg_execute_time.append(exe_time/nums*1000)

def rewrite_queries_2p(queries_2p, query_structure):
    # ('e', ('r', 'r'))
    print("-----handling 2p-----")
    start_time = time.time()
    nums = float(len(queries_2p))
    print("2p 查询的个数为：", nums)
    for query in queries_2p:
        # print(query)
        rewriting_queries[query] = {(query, 1.0, query_structure)}
        unique_queries = {query}
        last_rewriting = []
        this_rewriting = [(query, 1.0)]
        while len(rewriting_queries[query]) < rewriting_depth:
            last_rewriting = this_rewriting
            this_rewriting = []
            for each_query in last_rewriting:
                if len(rewriting_queries[query]) >= rewriting_depth:
                    break
                rel1 = each_query[0][1][0]
                rel2 = each_query[0][1][1]
                ent = each_query[0][0]
                query_conf = each_query[-1]
                # 与第一个rel匹配
                for each_rule in rule_dict[rel1]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if (ent, (body_pred, rel2)) not in unique_queries:
                        rewriting_queries[query].add(((ent, (body_pred, rel2)), query_conf*rule_conf, query_structure))
                        this_rewriting.append(((ent, (body_pred, rel2)), query_conf*rule_conf))
                        unique_queries.add((ent, (body_pred, rel2)))
                # 与第二个rel匹配
                for each_rule in rule_dict[rel2]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if (ent, (rel1, body_pred)) not in unique_queries:
                        rewriting_queries[query].add(((ent, (rel1, body_pred)), query_conf*rule_conf, query_structure))
                        this_rewriting.append(((ent, (rel1, body_pred)), query_conf*rule_conf))
                        unique_queries.add((ent, (rel1, body_pred)))
            if(len(this_rewriting) == 0):
                break
        if len(rewriting_queries[query]) > 1:
            print(rewriting_queries[query])
    end_time = time.time()
    exe_time = end_time - start_time
    print("2p 重写执行平均时间：", exe_time/nums)
    avg_execute_time.append(exe_time/nums*1000)

def rewrite_queries_3p(queries_3p, query_structure):
    # ('e', ('r', 'r', 'r'))
    print("-----handling 3p-----")
    start_time = time.time()
    nums = float(len(queries_3p))
    print("3p 查询的个数为：", nums)
    for query in queries_3p:
        # print(query)
        rewriting_queries[query] = {(query, 1.0, query_structure)}
        unique_queries = {query}
        last_rewriting = []
        this_rewriting = [(query, 1.0)]
        while len(rewriting_queries[query]) < rewriting_depth:
            last_rewriting = this_rewriting
            this_rewriting = []
            for each_query in last_rewriting:
                if len(rewriting_queries[query]) >= rewriting_depth:
                    break
                rel1 = each_query[0][1][0]
                rel2 = each_query[0][1][1]
                rel3 = each_query[0][1][2]
                ent = each_query[0][0]
                query_conf = each_query[-1]
                # 与第一个rel匹配
                for each_rule in rule_dict[rel1]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if (ent, (body_pred, rel2, rel3)) not in unique_queries:
                        rewriting_queries[query].add(((ent, (body_pred, rel2, rel3)), query_conf*rule_conf, query_structure))
                        this_rewriting.append(((ent, (body_pred, rel2, rel3)), query_conf*rule_conf))
                        unique_queries.add((ent, (body_pred, rel2, rel3)))
                # 与第二个rel匹配
                for each_rule in rule_dict[rel2]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if (ent, (rel1, body_pred, rel3)) not in unique_queries:
                        rewriting_queries[query].add(((ent, (rel1, body_pred, rel3)), query_conf*rule_conf, query_structure))
                        this_rewriting.append(((ent, (rel1, body_pred, rel3)), query_conf*rule_conf))
                        unique_queries.add((ent, (rel1, body_pred, rel3)))
                # 与第三个rel匹配
                for each_rule in rule_dict[rel3]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if (ent, (rel1, rel2, body_pred)) not in unique_queries:
                        rewriting_queries[query].add(((ent, (rel1, rel2, body_pred)), query_conf*rule_conf, query_structure))
                        this_rewriting.append(((ent, (rel1, rel2, body_pred)), query_conf*rule_conf))
                        unique_queries.add((ent, (rel1, rel2, body_pred)))
            if(len(this_rewriting) == 0):
                break
        if len(rewriting_queries[query]) > 1:
            print(rewriting_queries[query])
    end_time = time.time()
    exe_time = end_time - start_time
    print("3p 重写执行平均时间：", exe_time/nums)
    avg_execute_time.append(exe_time/nums*1000)

def rewrite_queries_2i(queries_2i, query_structure):
    # (('e', ('r',)), ('e', ('r',)))
    print("-----handling 2i-----")
    start_time = time.time()
    nums = float(len(queries_2i))
    print("2i 查询的个数为：", nums)
    for query in queries_2i:
        # print(query)
        rewriting_queries[query] = {(query, 1.0, query_structure)}
        unique_queries = {query}
        last_rewriting = []
        this_rewriting = [(query, 1.0)]
        while len(rewriting_queries[query]) < rewriting_depth:
            last_rewriting = this_rewriting
            this_rewriting = []
            for each_query in last_rewriting:
                if len(rewriting_queries[query]) >= rewriting_depth:
                    break
                rel1 = each_query[0][0][1][0]
                rel2 = each_query[0][1][1][0]
                ent1 = each_query[0][0][0]
                ent2 = each_query[0][1][0]
                query_conf = each_query[-1]
                # print(ent1, rel1, ent2, rel2, query_conf)
                # 与第一个rel匹配
                for each_rule in rule_dict[rel1]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if ((ent1, (body_pred,)), (ent2, (rel2,)))  not in unique_queries:
                        rewriting_queries[query].add((((ent1, (body_pred,)), (ent2, (rel2,))), query_conf*rule_conf, query_structure))
                        this_rewriting.append((((ent1, (body_pred,)), (ent2, (rel2,))), query_conf*rule_conf))
                        unique_queries.add(((ent1, (body_pred,)), (ent2, (rel2,))))
                # 与第二个rel匹配
                for each_rule in rule_dict[rel2]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if ((ent1, (rel1,)), (ent2, (body_pred,))) not in unique_queries:
                        rewriting_queries[query].add((((ent1, (rel1,)), (ent2, (body_pred,))), query_conf*rule_conf, query_structure))
                        this_rewriting.append((((ent1, (rel1,)), (ent2, (body_pred,))), query_conf*rule_conf))
                        unique_queries.add(((ent1, (rel1,)), (ent2, (body_pred,))))
            if(len(this_rewriting) == 0):
                break
        if len(rewriting_queries[query]) > 1:
            print(rewriting_queries[query])
    end_time = time.time()
    exe_time = end_time - start_time
    print("2i 重写执行平均时间：", exe_time/nums)
    avg_execute_time.append(exe_time/nums*1000)

def rewrite_queries_3i(queries_3i, query_structure):
    # (('e', ('r',)), ('e', ('r',)), ('e', ('r',)))
    print("-----handling 3i-----")
    start_time = time.time()
    nums = float(len(queries_3i))
    print("3i 查询的个数为：", nums)
    for query in queries_3i:
        # print(query)
        rewriting_queries[query] = {(query, 1.0, query_structure)}
        unique_queries = {query}
        last_rewriting = []
        this_rewriting = [(query, 1.0)]
        while len(rewriting_queries[query]) < rewriting_depth:
            last_rewriting = this_rewriting
            this_rewriting = []
            for each_query in last_rewriting:
                if len(rewriting_queries[query]) >= rewriting_depth:
                    break
                rel1 = each_query[0][0][1][0]
                rel2 = each_query[0][1][1][0]
                rel3 = each_query[0][2][1][0]
                ent1 = each_query[0][0][0]
                ent2 = each_query[0][1][0]
                ent3 = each_query[0][2][0]
                query_conf = each_query[-1]
                # print(ent1, rel1, ent2, rel2, ent3, rel3, query_conf)
                # 与第一个rel匹配
                for each_rule in rule_dict[rel1]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if ((ent1, (body_pred,)), (ent2, (rel2,)), (ent3, (rel3,)))  not in unique_queries:
                        rewriting_queries[query].add((((ent1, (body_pred,)), (ent2, (rel2,)), (ent3, (rel3,))), query_conf*rule_conf, query_structure))
                        this_rewriting.append((((ent1, (body_pred,)), (ent2, (rel2,)), (ent3, (rel3,))), query_conf*rule_conf))
                        unique_queries.add(((ent1, (body_pred,)), (ent2, (rel2,)), (ent3, (rel3,))))
                # 与第二个rel匹配
                for each_rule in rule_dict[rel2]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if ((ent1, (rel1,)), (ent2, (body_pred,)), (ent3, (rel3,))) not in unique_queries:
                        rewriting_queries[query].add((((ent1, (rel1,)), (ent2, (body_pred,)), (ent3, (rel3,))), query_conf*rule_conf, query_structure))
                        this_rewriting.append((((ent1, (rel1,)), (ent2, (body_pred,)), (ent3, (rel3,))), query_conf*rule_conf))
                        unique_queries.add(((ent1, (rel1,)), (ent2, (body_pred,)), (ent3, (rel3,))))
                # 与第三个rel匹配
                for each_rule in rule_dict[rel3]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if ((ent1, (rel1,)), (ent2, (rel2,)), (ent3, (body_pred,))) not in unique_queries:
                        rewriting_queries[query].add((((ent1, (rel1,)), (ent2, (rel2,)), (ent3, (body_pred,))), query_conf*rule_conf, query_structure))
                        this_rewriting.append((((ent1, (rel1,)), (ent2, (rel2,)), (ent3, (body_pred,))), query_conf*rule_conf))
                        unique_queries.add(((ent1, (rel1,)), (ent2, (rel2,)), (ent3, (body_pred,))))
            if(len(this_rewriting) == 0):
                break
        if len(rewriting_queries[query]) > 1:
            print(rewriting_queries[query])
    end_time = time.time()
    exe_time = end_time - start_time
    print("3i 重写执行平均时间：", exe_time/nums)
    avg_execute_time.append(exe_time/nums*1000)

def rewrite_queries_ip(queries_ip, query_structure):
    # ( ( ('e', ('r',)), ('e', ('r',))), ('r',))
    print("-----handling ip-----")
    start_time = time.time()
    nums = float(len(queries_ip))
    print("ip 查询的个数为：", nums)
    for query in queries_ip:
        # print(query)
        rewriting_queries[query] = {(query, 1.0, query_structure)}
        unique_queries = {query}
        last_rewriting = []
        this_rewriting = [(query, 1.0)]
        while len(rewriting_queries[query]) < rewriting_depth:
            last_rewriting = this_rewriting
            this_rewriting = []
            for each_query in last_rewriting:
                if len(rewriting_queries[query]) >= rewriting_depth:
                    break
                rel1 = each_query[0][0][0][1][0]
                rel2 = each_query[0][0][1][1][0]
                rel3 = each_query[0][1][0]
                ent1 = each_query[0][0][0][0]
                ent2 = each_query[0][0][1][0]
                query_conf = each_query[-1]
                # print(ent1, rel1, ent2, rel2, rel3, query_conf)
                # 与第一个rel匹配
                for each_rule in rule_dict[rel1]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if (((ent1, (body_pred,)), (ent2, (rel2,))), (rel3,)) not in unique_queries:
                        rewriting_queries[query].add(((((ent1, (body_pred,)), (ent2, (rel2,))), (rel3,)), query_conf*rule_conf, query_structure))
                        this_rewriting.append(((((ent1, (body_pred,)), (ent2, (rel2,))), (rel3,)), query_conf*rule_conf))
                        unique_queries.add((((ent1, (body_pred,)), (ent2, (rel2,))), (rel3,)))
                # 与第二个rel匹配
                for each_rule in rule_dict[rel2]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if (((ent1, (rel1,)), (ent2, (body_pred,))), (rel3,)) not in unique_queries:
                        rewriting_queries[query].add(((((ent1, (rel1,)), (ent2, (body_pred,))), (rel3,)), query_conf*rule_conf, query_structure))
                        this_rewriting.append(((((ent1, (rel1,)), (ent2, (body_pred,))), (rel3,)), query_conf*rule_conf))
                        unique_queries.add((((ent1, (rel1,)), (ent2, (body_pred,))), (rel3,)))
                # 与第三个rel匹配
                for each_rule in rule_dict[rel3]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if (((ent1, (rel1,)), (ent2, (rel2,))), (body_pred,)) not in unique_queries:
                        rewriting_queries[query].add(((((ent1, (rel1,)), (ent2, (rel2,))), (body_pred,)), query_conf*rule_conf, query_structure))
                        this_rewriting.append(((((ent1, (rel1,)), (ent2, (rel2,))), (body_pred,)), query_conf*rule_conf))
                        unique_queries.add((((ent1, (rel1,)), (ent2, (rel2,))), (body_pred,)))
            if(len(this_rewriting) == 0):
                break
        if len(rewriting_queries[query]) > 1:
            print(rewriting_queries[query])
    end_time = time.time()
    exe_time = end_time - start_time
    print("ip 重写执行平均时间：", exe_time/nums)
    avg_execute_time.append(exe_time/nums*1000)

def rewrite_queries_pi(queries_pi, query_structure):
    # (('e', ('r', 'r')), ('e', ('r',)))
    print("-----handling pi-----")
    start_time = time.time()
    nums = float(len(queries_pi))
    print("pi 查询的个数为：", nums)
    for query in queries_pi:
        # print(query)
        rewriting_queries[query] = {(query, 1.0, query_structure)}
        unique_queries = {query}
        last_rewriting = []
        this_rewriting = [(query, 1.0)]
        while len(rewriting_queries[query]) < rewriting_depth:
            last_rewriting = this_rewriting
            this_rewriting = []
            for each_query in last_rewriting:
                if len(rewriting_queries[query]) >= rewriting_depth:
                    break
                rel1 = each_query[0][0][1][0]
                rel2 = each_query[0][0][1][1]
                rel3 = each_query[0][1][1][0]
                ent1 = each_query[0][0][0]
                ent2 = each_query[0][1][0]
                query_conf = each_query[-1]
                # print(ent1, rel1, rel2, ent2, rel3, query_conf)
                # 与第一个rel匹配
                for each_rule in rule_dict[rel1]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if ((ent1, (body_pred, rel2)), (ent2, (rel3,))) not in unique_queries:
                        rewriting_queries[query].add((((ent1, (body_pred, rel2)), (ent2, (rel3,))), query_conf*rule_conf, query_structure))
                        this_rewriting.append((((ent1, (body_pred, rel2)), (ent2, (rel3,))), query_conf*rule_conf))
                        unique_queries.add(((ent1, (body_pred, rel2)), (ent2, (rel3,))))
                # 与第二个rel匹配
                for each_rule in rule_dict[rel2]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if ((ent1, (rel1, body_pred)), (ent2, (rel3,))) not in unique_queries:
                        rewriting_queries[query].add((((ent1, (rel1, body_pred)), (ent2, (rel3,))), query_conf*rule_conf, query_structure))
                        this_rewriting.append((((ent1, (rel1, body_pred)), (ent2, (rel3,))), query_conf*rule_conf))
                        unique_queries.add(((ent1, (rel1, body_pred)), (ent2, (rel3,))))
                # 与第三个rel匹配
                for each_rule in rule_dict[rel3]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if ((ent1, (rel1, rel2)), (ent2, (body_pred,))) not in unique_queries:
                        rewriting_queries[query].add((((ent1, (rel1, rel2)), (ent2, (body_pred,))), query_conf*rule_conf, query_structure))
                        this_rewriting.append((((ent1, (rel1, rel2)), (ent2, (body_pred,))), query_conf*rule_conf))
                        unique_queries.add(((ent1, (rel1, rel2)), (ent2, (body_pred,))))
            if(len(this_rewriting) == 0):
                break
        if len(rewriting_queries[query]) > 1:
            print(rewriting_queries[query])
    end_time = time.time()
    exe_time = end_time - start_time
    print("pi 重写执行平均时间：", exe_time/nums)
    avg_execute_time.append(exe_time/nums*1000)

def rewrite_queries_2u(queries_2u, query_structure):
    # ( (('e', ('r',)), ('e', ('r',)), ('u',) )
    print("-----handling 2u-----")
    start_time = time.time()
    nums = float(len(queries_2u))
    print("2u 查询的个数为：", nums)
    for query in queries_2u:
        # print(query)
        rewriting_queries[query] = {(query, 1.0, query_structure)}
        unique_queries = {query}
        last_rewriting = []
        this_rewriting = [(query, 1.0)]
        while len(rewriting_queries[query]) < rewriting_depth:
            last_rewriting = this_rewriting
            this_rewriting = []
            for each_query in last_rewriting:
                if len(rewriting_queries[query]) >= rewriting_depth:
                    break
                rel1 = each_query[0][0][1][0]
                rel2 = each_query[0][1][1][0]
                ent1 = each_query[0][0][0]
                ent2 = each_query[0][1][0]
                query_conf = each_query[-1]
                # print(ent1, rel1, ent2, rel2, query_conf)
                # 与第一个rel匹配
                for each_rule in rule_dict[rel1]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if ((ent1, (body_pred,)), (ent2, (rel2,)), (-1,))  not in unique_queries:
                        rewriting_queries[query].add((((ent1, (body_pred,)), (ent2, (rel2,)), (-1,)), query_conf*rule_conf, query_structure))
                        this_rewriting.append((((ent1, (body_pred,)), (ent2, (rel2,)), (-1,)), query_conf*rule_conf))
                        unique_queries.add(((ent1, (body_pred,)), (ent2, (rel2,)), (-1,)))
                # 与第二个rel匹配
                for each_rule in rule_dict[rel2]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if ((ent1, (rel1,)), (ent2, (body_pred,)), (-1,)) not in unique_queries:
                        rewriting_queries[query].add((((ent1, (rel1,)), (ent2, (body_pred,)), (-1,)), query_conf*rule_conf, query_structure))
                        this_rewriting.append((((ent1, (rel1,)), (ent2, (body_pred,)), (-1,)), query_conf*rule_conf))
                        unique_queries.add(((ent1, (rel1,)), (ent2, (body_pred,)), (-1,)))
            if(len(this_rewriting) == 0):
                break
        if len(rewriting_queries[query]) > 1:
            print(rewriting_queries[query])
    end_time = time.time()
    exe_time = end_time - start_time
    print("2u 重写执行平均时间：", exe_time/nums)
    avg_execute_time.append(exe_time/nums*1000)

def rewrite_queries_up(queries_up, query_structure):
    # ( ( ('e', ('r',)), ('e', ('r',)), ('u',) ), ('r',) )
    print("-----handling up-----")
    start_time = time.time()
    nums = float(len(queries_up))
    print("up 查询的个数为：", nums)
    for query in queries_up:
        # print(query)
        rewriting_queries[query] = {(query, 1.0, query_structure)}
        unique_queries = {query}
        last_rewriting = []
        this_rewriting = [(query, 1.0)]
        while len(rewriting_queries[query]) < rewriting_depth:
            last_rewriting = this_rewriting
            this_rewriting = []
            for each_query in last_rewriting:
                if len(rewriting_queries[query]) >= rewriting_depth:
                    break
                rel1 = each_query[0][0][0][1][0]
                rel2 = each_query[0][0][1][1][0]
                rel3 = each_query[0][0][2][0]
                ent1 = each_query[0][0][0][0]
                ent2 = each_query[0][0][1][0]
                query_conf = each_query[-1]
                # print(ent1, rel1, ent2, rel2, rel3, query_conf)
                # 与第一个rel匹配
                for each_rule in rule_dict[rel1]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if (((ent1, (body_pred,)), (ent2, (rel2,)), (-1,)), (rel3,))  not in unique_queries:
                        rewriting_queries[query].add(((((ent1, (body_pred,)), (ent2, (rel2,)), (-1,)), (rel3,)), query_conf*rule_conf, query_structure))
                        this_rewriting.append(((((ent1, (body_pred,)), (ent2, (rel2,)), (-1,)), (rel3,)), query_conf*rule_conf))
                        unique_queries.add((((ent1, (body_pred,)), (ent2, (rel2,)), (-1,)), (rel3,)))
                # 与第二个rel匹配
                for each_rule in rule_dict[rel2]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if (((ent1, (rel1,)), (ent2, (body_pred,)), (-1,)), (rel3,)) not in unique_queries:
                        rewriting_queries[query].add(((((ent1, (rel1,)), (ent2, (body_pred,)), (-1,)), (rel3,)), query_conf*rule_conf, query_structure))
                        this_rewriting.append(((((ent1, (rel1,)), (ent2, (body_pred,)), (-1,)), (rel3,)), query_conf*rule_conf))
                        unique_queries.add((((ent1, (rel1,)), (ent2, (body_pred,)), (-1,)), (rel3,)))
                # 与第三个rel匹配
                for each_rule in rule_dict[rel3]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if (((ent1, (rel1,)), (ent2, (rel2,)), (-1,)), (body_pred,)) not in unique_queries:
                        rewriting_queries[query].add(((((ent1, (rel1,)), (ent2, (rel2,)), (-1,)), (body_pred,)), query_conf*rule_conf, query_structure))
                        this_rewriting.append(((((ent1, (rel1,)), (ent2, (rel2,)), (-1,)), (body_pred,)), query_conf*rule_conf))
                        unique_queries.add((((ent1, (rel1,)), (ent2, (rel2,)), (-1,)), (body_pred,)))
            if(len(this_rewriting) == 0):
                break
        if len(rewriting_queries[query]) > 1:
            print(rewriting_queries[query])
    end_time = time.time()
    exe_time = end_time - start_time
    print("up 重写执行平均时间：", exe_time/nums)
    avg_execute_time.append(exe_time/nums*1000)

def rewrite_queries_inp(queries_inp, query_structure):
    # ( ( ('e', ('r',)), ('e', ('r', 'n')) ), ('r',) ): 'inp'  n -> -2
    print("-----handling inp-----")
    start_time = time.time()
    nums = float(len(queries_inp))
    print("inp 查询的个数为：", nums)
    for query in queries_inp:
        # print(query)
        rewriting_queries[query] = {(query, 1.0, query_structure)}
        unique_queries = {query}
        last_rewriting = []
        this_rewriting = [(query, 1.0)]
        while len(rewriting_queries[query]) < rewriting_depth:
            last_rewriting = this_rewriting
            this_rewriting = []
            for each_query in last_rewriting:
                if len(rewriting_queries[query]) >= rewriting_depth:
                    break
                rel1 = each_query[0][0][0][1][0]
                rel2 = each_query[0][0][1][1][0]
                rel3 = each_query[0][1][0]
                ent1 = each_query[0][0][0][0]
                ent2 = each_query[0][0][1][0]
                query_conf = each_query[-1]
                # print(ent1, rel1, ent2, rel2, rel3, query_conf)
                # 与第一个rel匹配
                for each_rule in rule_dict[rel1]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if (((ent1, (body_pred,)), (ent2, (rel2, -2))), (rel3,))  not in unique_queries:
                        rewriting_queries[query].add(((((ent1, (body_pred,)), (ent2, (rel2, -2))), (rel3,)), query_conf*rule_conf, query_structure))
                        this_rewriting.append(((((ent1, (body_pred,)), (ent2, (rel2, -2))), (rel3,)), query_conf*rule_conf))
                        unique_queries.add((((ent1, (body_pred,)), (ent2, (rel2, -2))), (rel3,)))
                # 与第二个rel匹配
                for each_rule in rule_dict[rel2]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if (((ent1, (rel1,)), (ent2, (body_pred, -2))), (rel3,)) not in unique_queries:
                        rewriting_queries[query].add(((((ent1, (rel1,)), (ent2, (body_pred, -2))), (rel3,)), query_conf*rule_conf, query_structure))
                        this_rewriting.append(((((ent1, (rel1,)), (ent2, (body_pred, -2))), (rel3,)), query_conf*rule_conf))
                        unique_queries.add((((ent1, (rel1,)), (ent2, (body_pred, -2))), (rel3,)))
                # 与第三个rel匹配
                for each_rule in rule_dict[rel3]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if (((ent1, (rel1,)), (ent2, (rel2, -2))), (body_pred,)) not in unique_queries:
                        rewriting_queries[query].add(((((ent1, (rel1,)), (ent2, (rel2, -2))), (body_pred,)), query_conf*rule_conf, query_structure))
                        this_rewriting.append(((((ent1, (rel1,)), (ent2, (rel2, -2))), (body_pred,)), query_conf*rule_conf))
                        unique_queries.add((((ent1, (rel1,)), (ent2, (rel2, -2))), (body_pred,)))
            if(len(this_rewriting) == 0):
                break
        if len(rewriting_queries[query]) > 1:
            print(rewriting_queries[query])
    end_time = time.time()
    exe_time = end_time - start_time
    print("inp 重写执行平均时间：", exe_time/nums)
    avg_execute_time.append(exe_time/nums*1000)

def rewrite_queries_pin(queries_pin, query_structure):
    # (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin'
    print("-----handling pin-----")
    start_time = time.time()
    nums = float(len(queries_pin))
    print("pin 查询的个数为：", nums)
    for query in queries_pin:
        # print(query)
        rewriting_queries[query] = {(query, 1.0, query_structure)}
        unique_queries = {query}
        last_rewriting = []
        this_rewriting = [(query, 1.0)]
        while len(rewriting_queries[query]) < rewriting_depth:
            last_rewriting = this_rewriting
            this_rewriting = []
            for each_query in last_rewriting:
                if len(rewriting_queries[query]) >= rewriting_depth:
                    break
                rel1 = each_query[0][0][1][0]
                rel2 = each_query[0][0][1][1]
                rel3 = each_query[0][1][1][0]
                ent1 = each_query[0][0][0]
                ent2 = each_query[0][1][0]
                query_conf = each_query[-1]
                # print(ent1, rel1, rel2, ent2, rel3, query_conf)
                # 与第一个rel匹配
                for each_rule in rule_dict[rel1]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if ((ent1, (body_pred, rel2)), (ent2, (rel3, -2))) not in unique_queries:
                        rewriting_queries[query].add((((ent1, (body_pred, rel2)), (ent2, (rel3, -2))), query_conf*rule_conf, query_structure))
                        this_rewriting.append((((ent1, (body_pred, rel2)), (ent2, (rel3, -2))), query_conf*rule_conf))
                        unique_queries.add(((ent1, (body_pred, rel2)), (ent2, (rel3, -2))))
                # 与第二个rel匹配
                for each_rule in rule_dict[rel2]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if ((ent1, (rel1, body_pred)), (ent2, (rel3, -2))) not in unique_queries:
                        rewriting_queries[query].add((((ent1, (rel1, body_pred)), (ent2, (rel3, -2))), query_conf*rule_conf, query_structure))
                        this_rewriting.append((((ent1, (rel1, body_pred)), (ent2, (rel3, -2))), query_conf*rule_conf))
                        unique_queries.add(((ent1, (rel1, body_pred)), (ent2, (rel3, -2))))
                # 与第三个rel匹配
                for each_rule in rule_dict[rel3]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if ((ent1, (rel1, rel2)), (ent2, (body_pred, -2))) not in unique_queries:
                        rewriting_queries[query].add((((ent1, (rel1, rel2)), (ent2, (body_pred, -2))), query_conf*rule_conf, query_structure))
                        this_rewriting.append((((ent1, (rel1, rel2)), (ent2, (body_pred, -2))), query_conf*rule_conf))
                        unique_queries.add(((ent1, (rel1, rel2)), (ent2, (body_pred, -2))))
            if(len(this_rewriting) == 0):
                break
        if len(rewriting_queries[query]) > 1:
            print(rewriting_queries[query])
    end_time = time.time()
    exe_time = end_time - start_time
    print("pin 重写执行平均时间：", exe_time/nums)
    avg_execute_time.append(exe_time/nums*1000)

def rewrite_queries_pni(queries_pni, query_structure):
    # (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni'
    print("-----handling pni-----")
    start_time = time.time()
    nums = float(len(queries_pni))
    print("pni 查询的个数为：", nums)
    for query in queries_pni:
        # print(query)
        rewriting_queries[query] = {(query, 1.0, query_structure)}
        unique_queries = {query}
        last_rewriting = []
        this_rewriting = [(query, 1.0)]
        while len(rewriting_queries[query]) < rewriting_depth:
            last_rewriting = this_rewriting
            this_rewriting = []
            for each_query in last_rewriting:
                if len(rewriting_queries[query]) >= rewriting_depth:
                    break
                rel1 = each_query[0][0][1][0]
                rel2 = each_query[0][0][1][1]
                rel3 = each_query[0][1][1][0]
                ent1 = each_query[0][0][0]
                ent2 = each_query[0][1][0]
                query_conf = each_query[-1]
                # print(ent1, rel1, rel2, ent2, rel3, query_conf)
                # 与第一个rel匹配
                for each_rule in rule_dict[rel1]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if ((ent1, (body_pred, rel2, -2)), (ent2, (rel3,))) not in unique_queries:
                        rewriting_queries[query].add((((ent1, (body_pred, rel2, -2)), (ent2, (rel3,))), query_conf*rule_conf, query_structure))
                        this_rewriting.append((((ent1, (body_pred, rel2, -2)), (ent2, (rel3,))), query_conf*rule_conf))
                        unique_queries.add(((ent1, (body_pred, rel2, -2)), (ent2, (rel3,))))
                # 与第二个rel匹配
                for each_rule in rule_dict[rel2]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if ((ent1, (rel1, body_pred, -2)), (ent2, (rel3,))) not in unique_queries:
                        rewriting_queries[query].add((((ent1, (rel1, body_pred, -2)), (ent2, (rel3,))) , query_conf*rule_conf, query_structure))
                        this_rewriting.append((((ent1, (rel1, body_pred, -2)), (ent2, (rel3,))) , query_conf*rule_conf))
                        unique_queries.add(((ent1, (rel1, body_pred, -2)), (ent2, (rel3,))) )
                # 与第三个rel匹配
                for each_rule in rule_dict[rel3]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if ((ent1, (rel1, rel2, -2)), (ent2, (body_pred,)))  not in unique_queries:
                        rewriting_queries[query].add((((ent1, (rel1, rel2, -2)), (ent2, (body_pred,))) , query_conf*rule_conf, query_structure))
                        this_rewriting.append((((ent1, (rel1, rel2, -2)), (ent2, (body_pred,))) , query_conf*rule_conf))
                        unique_queries.add(((ent1, (rel1, rel2, -2)), (ent2, (body_pred,))) )
            if(len(this_rewriting) == 0):
                break
        if len(rewriting_queries[query]) > 1:
            print(rewriting_queries[query])
    end_time = time.time()
    exe_time = end_time - start_time
    print("pni 重写执行平均时间：", exe_time/nums)
    avg_execute_time.append(exe_time/nums*1000)

def rewrite_queries_2in(queries_2in, query_structure):
    # (('e', ('r',)), ('e', ('r', 'n'))): '2in'
    print("-----handling 2in-----")
    start_time = time.time()
    nums = float(len(queries_2in))
    print("2in 查询的个数为：", nums)
    for query in queries_2in:
        # print(query)
        rewriting_queries[query] = {(query, 1.0, query_structure)}
        unique_queries = {query}
        last_rewriting = []
        this_rewriting = [(query, 1.0)]
        while len(rewriting_queries[query]) < rewriting_depth:
            last_rewriting = this_rewriting
            this_rewriting = []
            for each_query in last_rewriting:
                if len(rewriting_queries[query]) >= rewriting_depth:
                    break
                rel1 = each_query[0][0][1][0]
                rel2 = each_query[0][1][1][0]
                ent1 = each_query[0][0][0]
                ent2 = each_query[0][1][0]
                query_conf = each_query[-1]
                # print(ent1, rel1, ent2, rel2, query_conf)
                # 与第一个rel匹配
                for each_rule in rule_dict[rel1]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if ((ent1, (body_pred,)), (ent2, (rel2, -2)))  not in unique_queries:
                        rewriting_queries[query].add((((ent1, (body_pred,)), (ent2, (rel2, -2))), query_conf*rule_conf, query_structure))
                        this_rewriting.append((((ent1, (body_pred,)), (ent2, (rel2, -2))), query_conf*rule_conf))
                        unique_queries.add(((ent1, (body_pred,)), (ent2, (rel2, -2))))
                # 与第二个rel匹配
                for each_rule in rule_dict[rel2]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if ((ent1, (rel1,)), (ent2, (body_pred, -2))) not in unique_queries:
                        rewriting_queries[query].add((((ent1, (rel1,)), (ent2, (body_pred, -2))), query_conf*rule_conf, query_structure))
                        this_rewriting.append((((ent1, (rel1,)), (ent2, (body_pred, -2))), query_conf*rule_conf))
                        unique_queries.add(((ent1, (rel1,)), (ent2, (body_pred, -2))))
            if(len(this_rewriting) == 0):
                break
        if len(rewriting_queries[query]) > 1:
            print(rewriting_queries[query])
    end_time = time.time()
    exe_time = end_time - start_time
    print("2in 重写执行平均时间：", exe_time/nums)
    avg_execute_time.append(exe_time/nums*1000)

def rewrite_queries_3in(queries_3in, query_structure):
    # (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in'
    print("-----handling 3in-----")
    start_time = time.time()
    nums = float(len(queries_3in))
    print("3in 查询的个数为：", nums)
    for query in queries_3in:
        # print(query)
        rewriting_queries[query] = {(query, 1.0, query_structure)}
        unique_queries = {query}
        last_rewriting = []
        this_rewriting = [(query, 1.0)]
        while len(rewriting_queries[query]) < rewriting_depth:
            last_rewriting = this_rewriting
            this_rewriting = []
            for each_query in last_rewriting:
                if len(rewriting_queries[query]) >= rewriting_depth:
                    break
                rel1 = each_query[0][0][1][0]
                rel2 = each_query[0][1][1][0]
                rel3 = each_query[0][2][1][0]
                ent1 = each_query[0][0][0]
                ent2 = each_query[0][1][0]
                ent3 = each_query[0][2][0]
                query_conf = each_query[-1]
                # print(ent1, rel1, ent2, rel2, ent3, rel3, query_conf)
                # 与第一个rel匹配
                for each_rule in rule_dict[rel1]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if ((ent1, (body_pred,)), (ent2, (rel2,)), (ent3, (rel3, -2)))  not in unique_queries:
                        rewriting_queries[query].add((((ent1, (body_pred,)), (ent2, (rel2,)), (ent3, (rel3, -2))), query_conf*rule_conf, query_structure))
                        this_rewriting.append((((ent1, (body_pred,)), (ent2, (rel2,)), (ent3, (rel3, -2))), query_conf*rule_conf))
                        unique_queries.add(((ent1, (body_pred,)), (ent2, (rel2,)), (ent3, (rel3, -2))))
                # 与第二个rel匹配
                for each_rule in rule_dict[rel2]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if ((ent1, (rel1,)), (ent2, (body_pred,)), (ent3, (rel3, -2))) not in unique_queries:
                        rewriting_queries[query].add((((ent1, (rel1,)), (ent2, (body_pred,)), (ent3, (rel3, -2))), query_conf*rule_conf, query_structure))
                        this_rewriting.append((((ent1, (rel1,)), (ent2, (body_pred,)), (ent3, (rel3, -2))), query_conf*rule_conf))
                        unique_queries.add(((ent1, (rel1,)), (ent2, (body_pred,)), (ent3, (rel3, -2))))
                # 与第三个rel匹配
                for each_rule in rule_dict[rel3]:
                    body_pred = each_rule[0]
                    rule_conf = each_rule[1]
                    # 如果该查询加入过查询集
                    if ((ent1, (rel1,)), (ent2, (rel2,)), (ent3, (body_pred, -2))) not in unique_queries:
                        rewriting_queries[query].add((((ent1, (rel1,)), (ent2, (rel2,)), (ent3, (body_pred, -2))), query_conf*rule_conf, query_structure))
                        this_rewriting.append((((ent1, (rel1,)), (ent2, (rel2,)), (ent3, (body_pred, -2))), query_conf*rule_conf))
                        unique_queries.add(((ent1, (rel1,)), (ent2, (rel2,)), (ent3, (body_pred, -2))))
            if(len(this_rewriting) == 0):
                break
        if len(rewriting_queries[query]) > 1:
            print(rewriting_queries[query])
    end_time = time.time()
    exe_time = end_time - start_time
    print("3in 重写执行平均时间：", exe_time/nums)
    avg_execute_time.append(exe_time/nums*1000)

def execute_time_pic():
    print(avg_execute_time)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(query_types, avg_execute_time, color='skyblue', label='查询平均执行时间')
    ax.plot(query_types, avg_execute_time, marker='o', color='orange', label='查询平均执行时间')

    # 设置标题和标签
    ax.set_title('各种查询类型的平均执行时间', fontsize=16)
    ax.set_xlabel('查询类型', fontsize=12)
    ax.set_ylabel('平均执行时间(ms)', fontsize=12)

    # 旋转x轴标签，避免重叠
    plt.xticks(rotation=45)

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.6)

    # 添加图例
    ax.legend()

    # # 自动调整布局
    # plt.tight_layout()

    # 显示图表
    plt.show()
    plt.savefig('query_execution_time.png', dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Rewrite querys with the length-1 rules',
        usage='rewriting_queries.py [<args>] [-h | --help]'
    )
    parser.add_argument('--dataset', type=str, choices=["FB15K-237-betae", "FB15K-betae", "NELL-betae"], help="select KG")
    args = parser.parse_args()

    if args.dataset != "FB15K-betae":
        conf_shreshold = 0.5

    # 进一步处理规则
    with open("%s/length_1.txt"%args.dataset, "r") as f:
        rules = f.readlines()
    handle_rules(rules)
    # print(len(rule_list))
    make_rule_dict()

    rewrite_queries(args.dataset)

    execute_time_pic()
    
    # 将重写查询写入pkl文件
    # with open("%s/rewriting-test-queries.pkl"%args.dataset, "wb") as f: 
    #     pickle.dump(rewriting_queries, f)

    # for test
    # with open("%s/rewriting-test-queries-origin.pkl"%args.dataset, "rb") as f:
    #     data = pickle.load(f)

    # print(data[(2301, (38,))])

