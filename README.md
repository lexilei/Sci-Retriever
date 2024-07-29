# Sci-QA Retriever

[![arXiv](https://img.shields.io/badge/arXiv-2402.07630-b31b1b.svg)]()

This repository contains the source code for the paper ["<u>place holder</u>"]().

We introduce **sci-retriever**, 

## Installation 

```bash
pip install 
```
step 0 preprocess
generate question embeddings for all questions
save by index

Step 1 retrieve
given a graph, a question, and a sample answer, we perform the following:
use grag to retrieve optimal subgraph
then use bm25 to retrieve relevent passsages
save the optimal subgraph and relevent passages

step 2 llm
provide llm with the prompt
