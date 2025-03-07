"""
    改变数据集中数据的格式，使其能适配RUGE代码
    TODO:
        1. train,valid,test: (ent_id, rel_id, ent_id) -> (ent, rel, ent)
        2. id2rel,id2ent: {id: rel} -> (id \t rel)
"""
import os
import pickle
import argparse

def change_triples(dataset):
    with open(os.path.join(dataset, "id2ent.pkl"), "rb") as f:
        id2ent = pickle.load(f)
    with open(os.path.join(dataset, "id2rel.pkl"), "rb") as f:
        id2rel = pickle.load(f)
    splits = ["train", "valid", "test"]

    for each in splits:
        data_path = os.path.join(dataset, each+".txt")
        with open(data_path, "r") as f:
            data = f.readlines()
        new_data = []
        for triple in data:
            lines = triple.split('\t')
            h = id2ent[eval(lines[0])]
            t = id2ent[eval(lines[2])]
            rel = id2rel[eval(lines[1])]
            new_data.append(h + "\t" + rel + "\t" + t + "\n")
            print(h + "\t" + rel + "\t" + t + "\n")
        with open(os.path.join(dataset, dataset + "_triples." + each), 'w') as f:
            f.writelines(new_data)

def change_maps(dataset):
    with open(os.path.join(dataset, "id2ent.pkl"), "rb") as f:
        id2ent = pickle.load(f)
    with open(os.path.join(dataset, "id2rel.pkl"), "rb") as f:
        id2rel = pickle.load(f)
    relationid = []
    entityid = []
    for index in range(len(id2rel)):
        relationid.append(str(index) + "\t" + id2rel[index] + "\n")
    for index in range(len(id2ent)):
        entityid.append(str(index) + "\t" + id2ent[index] + "\n")
    with open("relationid.txt", "w") as f:
        f.writelines(relationid)
    with open("entityid.txt", "w") as f:
        f.writelines(entityid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Rewrite querys with the length-1 rules',
        usage='rewriting_queries.py [<args>] [-h | --help]'
    )
    parser.add_argument('--dataset', type=str, choices=["FB15K-237-betae", "FB15K-betae", "NELL-betae"], help="select KG")
    args = parser.parse_args()

    # change_triples(args.dataset)
    change_maps(args.dataset)