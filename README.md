# 基于逻辑规则的知识图谱上的复杂查询问答研究

我们提出了三种规则融入的方法：基于规则的查询重写方法、隐式融入方法、结合隐式和显式融入规则的方法，旨在提升复杂查询问答模型的表现。
我们的代码基于Pytorch进行编写，有关环境配置参考requriments.txt文件，代码的github地址为：[rewriting_CQA](https://github.com/Endless-Hack/rewriting_CQA)。
本代码在复杂查询问答模型QTO上应用规则融入的方法，有关QTO代码的运行问题参考[QTO](https://github.com/bys0318/QTO)。

## 数据预处理
知识图谱选择领域基准的三个数据集（FB15k, FB15k-237, NELL995），下载地址： [dataset](http://snap.stanford.edu/betae/KG_data.zip)。知识图谱数据放置到data目录下（mkdir data），进入kbc目录运行preprocess_datasets.py文件（python preprocess_datasets.py）来处理下载的数据集。

规则的获取选择规则学习器[AnyBURL](https://web.informatik.uni-mannheim.de/AnyBURL/#run)，本文只考虑两种类型的规则：蕴含规则和变迁规则，得到规则后需要对规则按类型和置信度进行筛选。AnyBURL的参数设置在其代码目录下build/config-learn.properties文件中修改，本文应用的超参数设置为SNAPSHOTS_AT = 10,50,100,600; THRESHOLD_CONFIDENCE = 0.95(FB15K中为0.95，具体参数设置可参考论文);MAX_LENGTH_CYCLIC = 1;SAFE_PREFIX_MODE = true

注：基于规则的查询重写方法中规则放置的目录和命令参考data/rewriting_queries.py，隐式融入规则方法中规则放置的目录和命令kbc/src/handle_rules.py


## 基于规则的查询重写方法
基于规则的查询重写方法利用规则重写查询，将一条查询扩展成一个查询集，并提出了三种查询聚合策略，将经过复杂查询问答模型（QTO）的推理答案进行聚合，确保了答案的完整性。

查询重写的代码: data/rewriting_queries.py，需要在多跳推理之前运行，重写的查询集命名为rewriting-test-queries.pkl

根据不同的聚合策略代码分为三个分支：agg_method1（对应基于实体排名的聚合策略）、agg_based_score（对应基于max、avg的聚合策略）、agg_based_score_nosiy_or（对应noisy_or聚合策略），三个分支对QTO的推理过程的代码进行了修改，代码运行和超参数参考execute_qto.sh

多跳推理的结果会保存在results/目录中，模型参数的文件(.pt)保存在neural_adj/目录中

## 隐式融入规则的方法
QTO需要一个预训练的知识图谱嵌入模型来执行单跳推理，有关代码在kbc/src/目录下。隐式融入规则旨在知识图谱嵌入模型的训练过程中融入规则，从而使得嵌入模型中蕴含规则信息，辅助多跳推理过程。

规则目录在kbc/src/rule，格式须符合kbc/src/handle_rules.py处理条件

隐式融入规则代码的执行和超参数参考kbc/src/train_kge.sh，其中三元组和规则的batch_size需要根据不同数据集上的规则数目自己指定，脚本中的bs1，bs2即对应grounding1_batch和grounding2_batch，这里仅供参考。

代码运行结果保存在kbc/src/data/bys/Neural-KoPL/rp/models，训练中的best_valid.model保存在kbc/FB15K/中



