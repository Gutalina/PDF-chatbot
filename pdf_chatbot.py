import fitz
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Step 1: Load and chunk the PDF document (Retrieval step of RAG)
def extract_chunks_from_pdf(pdf_path, chunk_size=300):
    doc = fitz.open(pdf_path)
    text = ' '.join(page.get_text() for page in doc)
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Step 2: Get embeddings for each chunk using "paraphrase-multilingual" (Encoder model for retrieval)
def get_embedding(text):
    response = requests.post('http://localhost:11434/api/embeddings', json={
        'model': 'paraphrase-multilingual',
        'prompt': text
    })
    return response.json()['embedding']

# Step 3: Use Mistral to generate an answer based on retrieved context (Generation step of RAG)
def generate_answer(context, question):
    prompt = f'Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely and do not refer to example code:'
    response = requests.post('http://localhost:11434/api/generate', json={
        'model': 'mistral',
        'prompt': prompt,
        'stream': False
    })
    text = response.json()['response']
    for stop_phrase in ['In your example code', 'As shown above', 'As mentioned earlier']:
        if stop_phrase in text:
            text = text.split(stop_phrase)[0].strip()
    return text.strip()

print(' Loading and processing PDF...')
chunks = extract_chunks_from_pdf('Nic_NlpBooklet_20240507.pdf')  # Load your PDF here
print(f' Extracted {len(chunks)} chunks from PDF.')

print(' Embedding chunks...')
vectors = [get_embedding(chunk) for chunk in chunks]
print(' Embedding complete.')

# Step 4: Find the most relevant chunk for the user's question (R - retrieval)
def search_context(question, top_k=1):
    q_vec = np.array(get_embedding(question)).reshape(1, -1)
    sims = cosine_similarity(q_vec, np.array(vectors))[0]
    best_indices = sims.argsort()[::-1][:top_k]
    return '\n\n'.join([chunks[i] for i in best_indices])

# FastAPI app setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class Query(BaseModel):
    question: str

@app.post('/chat')
def chat(query: Query):
    context = search_context(query.question)
    answer = generate_answer(context, query.question)
    return {'answer': answer.strip()}

if __name__ == '__main__':
    uvicorn.run('pdf_chatbot:app', host='0.0.0.0', port=8000, reload=True)



