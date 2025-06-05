# PDF Chatbot with RAG (Retrieval-Augmented Generation)

This is a local chatbot that can answer questions based on the content of a PDF file.  
It uses semantic search (`paraphrase-multilingual`) and a generative LLM (`mistral`) via Ollama.

## Features

- Loads any PDF document
- Splits it into chunks and embeds them using `paraphrase-multilingual`
- Finds the most relevant chunk for a user question
- Uses `mistral` to generate a conversational answer

## Requirements

- Python 3.8+
- Ollama installed and running
- Downloaded models:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull paraphrase-multilingual
  ollama pull mistral
  ```
- Python packages:
  ```bash
  pip install fastapi uvicorn pymupdf requests scikit-learn
  ```

## Files

- `pdf_chatbot.py` — FastAPI backend
- `pdf_chatbot_front.html` — simple chat interface
- `example.pdf` — any PDF file 

This project was tested using the [Hugging Face NLP Course PDF](https://figshare.com/articles/book/_b_Hands-On_NLP_with_Hugging_Face_A_Practical_Guide_for_University-Level_Education_in_Modern_Language_Processing_Technologies_b_/25764642?file=46153470). You can download the PDF manually.


## How to Run

1. Make sure Ollama is installed and models are downloaded:
   ```bash
   ollama serve
   ollama pull paraphrase-multilingual
   ollama pull mistral
   ```

2. Start the chatbot:
   ```bash
   python pdf_chatbot.py
   ```

3. Open `pdf_chatbot_front.html` in your browser.

4. Ask questions in natural language. The bot will search the PDF and generate an answer using Mistral.

## Notes

- Mistral is resource-intensive. On CPU-only systems, responses may take 10–20 seconds.
- You can use a smaller model like `phi3` for faster responses.
- All processing is done locally. No data is sent to any external server.

## Example Question

If using the Hugging Face NLP Course PDF, a sample question could be:

```
What is the difference between pretraining and fine-tuning in transformers?
```
## Multilingual Support
You can ask questions in different languages.
The search works across multiple languages using paraphrase-multilingual, and the answer will follow the language used in the prompt.

## Privacy and Deployment

This project uses local LLMs via Ollama.  
All generation and embedding is done offline.  
This setup is suitable for self-hosted deployments in organizations without relying on cloud APIs.

