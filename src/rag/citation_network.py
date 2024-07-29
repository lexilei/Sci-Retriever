import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import datasets
from tqdm import tqdm
from src.dataset.utils.retrieval import retrieval_via_pcst
from bm25 import BM25
from src.utils.lm_modeling import load_model, load_text2embedding

model_name = 'sbert'
path = 'dataset/graphs'
# path_nodes = f'{path}/nodes'
# path_edges = f'{path}/edges'
# path_graphs = f'{path}/graphs'

cached_graph = 'dataset/retrieved/optimal_subg'
cached_desc = 'dataset/retrieved/desc'


class CitationNetworkDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.prompt = 'Please answer the given question.'
        self.graph = None
        self.graph_type = 'Knowledge Graph'
        dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
        self.dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
        self.q_embs = torch.load(f'{path}/q_embs.pt')

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        question = f'Question: {data["question"]}\nAnswer: '
        graph = torch.load(f'{cached_graph}/{index}.pt')
        desc = open(f'{cached_desc}/{index}.txt', 'r').read()
        label = ('|').join(data['answer']).lower()

        return {
            'id': index,
            'question': question,
            'label': label,
            'graph': graph,
            'desc': desc,
        }

    def get_idx_split(self):

        # Load the saved indices
        with open(f'{path}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


def process():
    os.makedirs(cached_desc, exist_ok=True)
    os.makedirs(cached_graph, exist_ok=True)
    # dataset = datasets.load_dataset("rmanluo/RoG-webqsp")
    # dataset = datasets.concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    dataset = pd.read_csv('/home/ubuntu/Sci-Retriever/sampleqa.csv')
    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    # encode questions
    print('Encoding questions...')
    # q_embs = text2embedding(model, tokenizer, device, questions)
    # torch.save(q_embs, f'{path}/q_embs.pt')
    # q_embs = torch.load(f'{path}/q_embs.pt')
    for index in tqdm(range(len(dataset))):
        if os.path.exists(f'{cached_graph}/{index}.pt'):
            continue
        data=dataset.iloc[index]
        graph_name=data['graph']
        graph = torch.load('/home/ubuntu/Sci-Retriever/object-detection-on-coco-o.pt')
        print(graph.x)
        print(graph.edge_index)
        print(graph.edge_attr)
        question=data['question']
        answer=data['answer']
        q_emb = text2embedding(model, tokenizer, device, question)
        subg=retrieval_via_pcst(graph, q_emb, topk=3, topk_e=5, cost_e=0.5)

        desc=BM25(question,subg,topk=3) #加了一个bm25here
        print(desc)
        torch.save(subg, f'{cached_graph}/{index}.pt')
        open(f'{cached_desc}/{index}.txt', 'w').write(desc)


if __name__ == '__main__':

    process()

    # dataset = CitationNetworkDataset()

    # data = dataset[1]
    # for k, v in data.items():
    #     print(f'{k}: {v}')

    # split_ids = dataset.get_idx_split()
    # for k, v in split_ids.items():
    #     print(f'# {k}: {len(v)}')
