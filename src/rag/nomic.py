import torch
import numpy as np
from pcst_fast import pcst_fast
from torch_geometric.data.data import Data
import pandas as pd
from src.utils.lm_modeling import load_model, load_text2embedding
from transformers import AutoTokenizer, AutoModel
import re
from openai import OpenAI
import warnings
warnings.filterwarnings("ignore")

# from sentence_transformers import SentenceTransformer
from nomic import embed

def retrieval_via_pcst(graph, textual_nodes, q_emb, topk=3, topk_e=3, cost_e=0.5):
    c = 0.01
    # if len(textual_nodes) == 0 or len(textual_edges) == 0:
    #     desc = textual_nodes.to_csv(index=False) + '\n' + textual_edges.to_csv(index=False, columns=['src', 'edge_attr', 'dst'])
    #     graph = Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, num_nodes=graph.num_nodes)
    #     return graph, desc

    root = -1  # unrooted
    num_clusters = 1
    pruning = 'gw'
    verbosity_level = 0
    if topk > 0:
        n_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, graph.x)
        topk = min(topk, graph.num_nodes)
        print(topk)
        _, topk_n_indices = torch.topk(n_prizes, topk, largest=True)
        print(topk_n_indices)
        n_prizes = torch.zeros_like(n_prizes)
        n_prizes[topk_n_indices] = torch.arange(topk, 0, -1).float()
    else:
        n_prizes = torch.zeros(graph.num_nodes)

    if topk_e > 0:
        e_prizes = torch.nn.CosineSimilarity(dim=-1)(q_emb, torch.empty(0))
        topk_e = min(topk_e, e_prizes.unique().size(0))

        topk_e_values, _ = torch.topk(e_prizes.unique(), topk_e, largest=True)
        e_prizes[e_prizes < topk_e_values[-1]] = 0.0
        last_topk_e_value = topk_e
        for k in range(topk_e):
            indices = e_prizes == topk_e_values[k]
            value = min((topk_e-k)/sum(indices), last_topk_e_value)
            e_prizes[indices] = value
            last_topk_e_value = value*(1-c)
        # reduce the cost of the edges such that at least one edge is selected
        cost_e = min(cost_e, e_prizes.max().item()*(1-c/2))
    else:
        e_prizes = torch.zeros(graph.num_edges)

    costs = []
    edges = []
    vritual_n_prizes = []
    virtual_edges = []
    virtual_costs = []
    mapping_n = {}
    mapping_e = {}
    for i, (src, dst) in enumerate(graph.edge_index.T.numpy()):
        prize_e = e_prizes[i]
        if prize_e <= cost_e:
            mapping_e[len(edges)] = i
            edges.append((src, dst))
            costs.append(cost_e - prize_e)
        else:
            virtual_node_id = graph.num_nodes + len(vritual_n_prizes)
            mapping_n[virtual_node_id] = i
            virtual_edges.append((src, virtual_node_id))
            virtual_edges.append((virtual_node_id, dst))
            virtual_costs.append(0)
            virtual_costs.append(0)
            vritual_n_prizes.append(prize_e - cost_e)

    prizes = np.concatenate([n_prizes, np.array(vritual_n_prizes)])
    num_edges = len(edges)
    if len(virtual_costs) > 0:
        costs = np.array(costs+virtual_costs)
        edges = np.array(edges+virtual_edges)

    vertices, edges = pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity_level)

    selected_nodes = vertices[vertices < graph.num_nodes]
    selected_edges = [mapping_e[e] for e in edges if e < num_edges]
    virtual_vertices = vertices[vertices >= graph.num_nodes]
    if len(virtual_vertices) > 0:
        virtual_vertices = vertices[vertices >= graph.num_nodes]
        virtual_edges = [mapping_n[i] for i in virtual_vertices]
        selected_edges = np.array(selected_edges+virtual_edges)

    edge_index = graph.edge_index[:, selected_edges]
    selected_nodes = np.unique(np.concatenate([selected_nodes, edge_index[0].numpy(), edge_index[1].numpy()]))
    # print(selected_nodes)
    textual_nodes=pd.DataFrame(textual_nodes)
    n = textual_nodes.iloc[selected_nodes]
    # e = textual_edges.iloc[selected_edges]
    desc = n#.to_csv(index=False)

    mapping = {n: i for i, n in enumerate(selected_nodes.tolist())}

    x = graph.x[selected_nodes]
    edge_attr = graph.edge_attr#[selected_edges]
    src = [mapping[i] for i in edge_index[0].tolist()]
    dst = [mapping[i] for i in edge_index[1].tolist()]
    edge_index = torch.LongTensor([src, dst])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(selected_nodes))
    return data, desc

def BM25(query, contexts, topk):
    text_column = contexts.columns[0]
    text_list = contexts[text_column].tolist()
    # contexts=converter(subg)
    # Retrieve with BM25
    

# 将每个文本条目分割成句子
    def split_into_sentences(text):
        # 使用正则表达式进行简单的句子分割，可以根据需要调整
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        return sentences

    # 转换为句子的列表
    sentences_list = [sentence for text in text_list for sentence in split_into_sentences(text)]

    # print(sentences_list)
    tokenizer = AutoTokenizer.from_pretrained('facebook/spar-wiki-bm25-lexmodel-query-encoder')
    query_encoder = AutoModel.from_pretrained('facebook/spar-wiki-bm25-lexmodel-query-encoder')
    context_encoder = AutoModel.from_pretrained('facebook/spar-wiki-bm25-lexmodel-context-encoder')

    query_input = tokenizer(query, padding=True, truncation=True, return_tensors='pt')
    ctx_input = tokenizer(sentences_list, padding=True, truncation=True, return_tensors='pt')

    # Compute embeddings: take the last-layer hidden state of the [CLS] token
    query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :]
    ctx_emb = context_encoder(**ctx_input).last_hidden_state[:, 0, :]

    scores = query_emb @ ctx_emb.T
    _, topk_n_indices = torch.topk(scores, topk, largest=True)
    # print(topk_n_indices[0].tolist())
    selected_contexts = " ".join([sentences_list[idx] for idx in topk_n_indices[0].tolist()])
        # query[q_id] = selected_contexts + " " + query[q_id]

    return selected_contexts

def BM252(query, contexts, topk):

    # 转换为句子的列表

    # print(sentences_list)
    tokenizer = AutoTokenizer.from_pretrained('facebook/spar-wiki-bm25-lexmodel-query-encoder')
    query_encoder = AutoModel.from_pretrained('facebook/spar-wiki-bm25-lexmodel-query-encoder')
    context_encoder = AutoModel.from_pretrained('facebook/spar-wiki-bm25-lexmodel-context-encoder')

    query_input = tokenizer(query, padding=True, truncation=True, return_tensors='pt')
    ctx_input = tokenizer(contexts, padding=True, truncation=True, return_tensors='pt')

    # Compute embeddings: take the last-layer hidden state of the [CLS] token
    query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :]
    ctx_emb = context_encoder(**ctx_input).last_hidden_state[:, 0, :]

    scores = query_emb @ ctx_emb.T
    _, topk_n_indices = torch.topk(scores, topk, largest=True)
    # print(topk_n_indices[0].tolist())
    selected_contexts = " ".join([contexts[idx] for idx in topk_n_indices[0].tolist()])
        # query[q_id] = selected_contexts + " " + query[q_id]

    return selected_contexts

model_name = 'sbert'
path = 'Sci-Retriever/dataset/3d-point-cloud-classification-on-scanobjectnn.pt'

score=0

dataset = pd.read_csv('/home/ubuntu/Sci-Retriever/dataset/sampleqa.csv')

model, tokenizer, device = load_model[model_name]()
text2embedding = load_text2embedding[model_name]
graph=torch.load('/home/ubuntu/Sci-Retriever/dataset/3d-point-cloud-classification-on-scanobjectnn.pt')


abstract_emb= embed.text(
    texts=graph.abstract,
    model='nomic-embed-text-v1.5',
    task_type='search_document',
    dimensionality=256,
)



graph.x=abstract_emb
maxscore=0
for topk in [5]:
    score=0
    for index in range(len(dataset)):
        data=dataset.iloc[index]
        # print(data)
        question=data['question']
        answer=data['answer']
        # q_emb = text2embedding(model, tokenizer, device, "The unstructured nature of point clouds demands that local aggregation be adaptive to different local structures.")
        # abstract_emb = text2embedding(model, tokenizer, device, graph.abstract)
        q_emb= embed.text(
            texts=question,
            model='nomic-embed-text-v1.5',
            task_type='search_query',
            dimensionality=256,
        )
        print(q_emb)
    # print("finished generation now saving",graph.abstract)
    # torch.save(abstract_emb,"/home/ubuntu/Sci-Retriever/dataset/a.pt")
        
        subg,desc=retrieval_via_pcst(graph, graph.title, q_emb, topk=10, topk_e=0, cost_e=0.5)
    # file_name = "desc.txt"
    # with open(file_name, 'w') as file:
    #     file.write(desc.to_string(index=False))
        print("running bm25")
        print(desc)
        # result= BM25(question, desc, 50)
        result=desc
        with open(f"/home/ubuntu/Sci-Retriever/temp.txt", 'w') as file:
            file.write(result)

        client = OpenAI()
        completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Answer the folowing question based on the information given."},
            {"role": "user", "content": f"{question} {result}"}
        ]
        )
        
        file_name = "output.txt"
        with open(f"/home/ubuntu/Sci-Retriever/dataset/chatgptanswers/{index}.txt", 'w') as file:
            file.write(completion.choices[0].message.content)
        if answer in completion.choices[0].message.content:
            score+=1
        if score>maxscore:
            maxscore=score
            maxk=topk
    print(score/len(dataset))