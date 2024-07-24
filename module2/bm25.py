

def BM25(query, contexts, topk):
  # Retrieve with BM25
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

  for q_id in range(len(query)):
    selected_contexts = " ".join([contexts[idx] for idx in topk_n_indices[q_id]])
    query[q_id] = selected_contexts + " " + query[q_id]

  return query

# Here is an easiest way to call LLaMA for responses.
def LLaMA_answers(model, tokenizer, prompts):

  input_ids = tokenizer_llama.encode(prompts, return_tensors="pt")
  input_ids = input_ids.to(device)
  output = model_llama.generate(input_ids, max_length=256, num_beams=4, no_repeat_ngram_size=2)
  responses = tokenizer_llama.decode(output[0], skip_special_tokens=True)

  return responses

# topk = 2
# # You can input multi-questions to perform batch computation
# query = ["Where was Marie Curie born?", "When was Marie Curie born?"]
# contexts = [
#     "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
#     "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
# ]

# prompts = BM25(query, contexts, topk)
# answers = LLaMA_answers(model_llama, tokenizer_llama, prompts)
