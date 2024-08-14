from nomic import embed
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
from itertools import combinations

def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

# def overall_similarity(embeddings_A, embeddings_B):
#     similarities = [cosine_similarity(A, B) for A, B in zip(embeddings_A, embeddings_B)]
#     return np.mean(similarities)

def overall_similarity(embeddings):
    # Generate all pairs of embeddings
    pairs = combinations(embeddings, 2)
    
    # Calculate cosine similarity for each pair
    similarities = [cosine_similarity(A, B) for A, B in pairs]
    
    # Calculate the mean of all similarities
    return np.mean(similarities)

# 示例文本嵌入向量
embs_1=torch.load("/home/ubuntu/Sci-Retriever/dataset/a.pt")
embs_2=torch.load("/home/ubuntu/Sci-Retriever/nomic_abs_emb_768.pt")['embeddings']

# 计算总体相似性
overall_sim1 = overall_similarity(embs_1)
overall_sim2 = overall_similarity(embs_2)
print("Overall Similarity of sbert:", overall_sim1)
print("Overall Similarity of nomic:", overall_sim2)
