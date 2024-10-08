import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm
from retrieval import retrieval_via_pcst
from bm25 import BM25
from src.utils.lm_modeling import load_model, load_text2embedding

model_name = 'sbert'
path = 'dataset/graphs'
# path_nodes = f'{path}/nodes'
# path_edges = f'{path}/edges'
# path_graphs = f'{path}/graphs'

cached_graph = 'dataset/retrieved/optimal_subg'
cached_desc = 'dataset/retrieved/desc'


def process():
    # os.makedirs(cached_desc, exist_ok=True)
    # os.makedirs(cached_graph, exist_ok=True)
    # dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
    # dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    dataset = pd.read_csv('/home/ubuntu/Sci-Retriever/dataset/sampleqa.csv')
    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    # encode questions
    # print('Encoding questions...')
    # q_embs = text2embedding(model, tokenizer, device, questions)
    # torch.save(q_embs, f'{path}/q_embs.pt')
    # q_embs = torch.load(f'{path}/q_embs.pt')
    correct=0
    graph_name=''
    for index in tqdm(range(len(dataset))):
        if os.path.exists(f'{cached_graph}/{index}.pt'):
            continue
        data=dataset.iloc[index]
        if graph_name!=data['graph']:
            graph_name=data['graph']
            graph = torch.load(f'/home/ubuntu/Sci-Retriever/dataset/graphs/{graph_name}.pt')
        # print(graph.content)
        #print(graph.abstract)
        # print(graph.edge_attr)
        question=data['question']
        answer=data['answer']
        print(answer)
        q_emb = text2embedding(model, tokenizer, device, question)
        print("q emb generated")
        print(graph.title)
        abstract_emb = text2embedding(model, tokenizer, device, graph.abstract[:500])
        print("abs emb generated")
        graph.x=abstract_emb
        print("Retrieving...")
        subg=retrieval_via_pcst(graph, q_emb, topk=3, topk_e=0, cost_e=0.5)
        print("Saving retrieved graph...")
        torch.save(subg, f'{cached_graph}/{index}.pt')
        print("Saved")
        desc=BM25(question,subg,topk=3) #加了一个bm25here
        print(desc)
        
        open(f'{cached_desc}/{index}.txt', 'w').write(desc)


if __name__ == '__main__':

    process()