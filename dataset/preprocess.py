path='/home/ubuntu/Sci-Retriever/dataset/qa_qemb'
model_name = 'sbert'
def generate_q_embs():
    os.makedirs(path, exist_ok=True)
    dataset = pd.read_csv('/home/ubuntu/Sci-Retriever/dataset/qa_pairs.csv')
    model, tokenizer, device = load_model[model_name]()
    text2embedding = load_text2embedding[model_name]

    # encode questions
    print('Encoding questions...')
    # q_embs = text2embedding(model, tokenizer, device, questions)
    # torch.save(q_embs, f'{path}/q_embs.pt')
    # q_embs = torch.load(f'{path}/q_embs.pt')
    for index in tqdm(range(len(dataset))):
        if os.path.exists(f'{path}/{index}.pt'):
            continue
        data=dataset.iloc[index]
        question=data['question']
        print(question)
        answer=data['answer']
        q_emb = text2embedding(model, tokenizer, device, question)
        
        torch.save(q_emb, f'{path}/{index}.pt')
        # open(f'{cached_desc}/{index}.txt', 'w').write(desc)


if __name__ == '__main__':

    generate_q_embs()