# Investment Rules RAG

This Django project now has a small RAG ingestion pipeline in the `rag` app.

## What the pipeline does

1. Extracts selectable text from a PDF page by page.
2. Cleans repeated headers, footers, empty lines, and spacing noise.
3. Splits the text into overlapping chunks.
4. Creates one embedding vector for each chunk.
5. Stores the source document, chunk metadata, chunk text, and vectors in a local Chroma vector database.
6. Searches the stored chunks by comparing a customer question vector to the chunk vectors.

## Why chunking is done this way

The chunker starts from document structure instead of blindly cutting every N characters. It keeps page numbers, watches for likely headings such as Arabic legal sections and articles, and avoids splitting in the middle of normal paragraphs when possible.

The default chunk size is about 800 tokens with 120 tokens of overlap. The overlap helps when a customer asks a question whose answer crosses the boundary between two chunks.

Each chunk stores:

- source PDF name
- page start and page end
- section title, when detected
- chunk index
- token count
- original chunk text
- embedding vector

## Index the PDF

From the `rag_ai` directory:

Create a local `.env` file first:

```text
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_APP_TITLE=Investment Rules RAG
OPENROUTER_HTTP_REFERER=http://localhost:8000
RAG_EMBEDDING_PROVIDER=openrouter
OPENROUTER_EMBEDDING_MODEL=nvidia/nemotron-3-embed-1b:free
RAG_CHAT_MODEL=openrouter/free
RAG_OCR_MODEL=openrouter/free
RAG_OCR_PDF_ENGINE=cloudflare-ai
RAG_DEFAULT_TOP_K=5
RAG_REQUEST_TIMEOUT=180
RAG_EMBEDDING_TIMEOUT=120
RAG_OCR_TIMEOUT=180
RAG_SOURCE_MAX_CHARS=1200
RAG_MAX_OUTPUT_TOKENS=700
```

```powershell
python manage.py ingest_pdf "C:\Users\batoo\OneDrive\Desktop\Investment Rule.pdf"
```

You can index multiple PDFs into the same Chroma collection:

```powershell
python manage.py ingest_pdf "C:\Users\batoo\OneDrive\Desktop\Investment Rule.pdf" "C:\Users\batoo\OneDrive\Desktop\Investment Rule 2.pdf" "C:\Users\batoo\OneDrive\Desktop\Investment Rule 3.pdf"
```

If a PDF is scanned/image-only, the command falls back to OpenRouter PDF parsing with the free `cloudflare-ai` engine. If a PDF has selectable text but the extraction quality is poor, set `RAG_FORCE_OCR=true` in `.env` and re-index that PDF.

Optional settings:

```powershell
python manage.py ingest_pdf "C:\Users\batoo\OneDrive\Desktop\Investment Rule.pdf" --max-tokens 800 --overlap-tokens 120
```

The default Chroma persistence path is:

```text
rag_ai/data/chroma
```

By default, the command stores each embedding provider and model in a separate Chroma collection. For example, the default free OpenRouter embedding model uses:

```text
investment_rules_openrouter-nvidia-nemotron-3-embed-1b-free
```

## Search indexed chunks

```powershell
python manage.py search_rag "ما هي شروط الاستثمار؟" --top-k 5
```

This returns the most relevant Chroma chunks with page numbers. The next step is to pass those chunks to an LLM and instruct it to answer only from the retrieved source text.

## Ask a question

```powershell
python manage.py ask_rag "ما هي شروط الاستثمار؟" --top-k 5
```

This retrieves the top matching chunks from Chroma, sends only those chunks to the configured OpenRouter chat model, and returns an answer with page citations.

The default answer model is:

```text
openrouter/free
```

This uses OpenRouter's free model router. OCR uses OpenRouter's free `cloudflare-ai` PDF engine. Free models are useful for demos and low-volume testing, but they can have lower rate limits, changing availability, slower responses, and less predictable answer quality than paid models.

## Ask through the backend API

Start the Django server:

```powershell
python manage.py runserver
```

Send a POST request:

```text
POST http://127.0.0.1:8000/api/ask/
Content-Type: application/json
```

Body:

```json
{
  "question": "ما هي شروط الاستثمار؟"
}
```

The API uses these defaults automatically: `embedding_provider=openrouter`, `model=openrouter/free`, `top_k=5`, and `temperature=0.1`.

Example response:

```json
{
  "question": "ما هي شروط الاستثمار؟",
  "answer": "بناء على النصوص المسترجعة من القانون، فإن شروط الاستثمار تشمل تقديم الوثائق المطلوبة وارتباط المشروع بالقطاعات المشمولة بأحكام القانون، وذلك وفق ما ورد في المصادر المسترجعة [source 1, pages 35-36] [source 2, pages 38-39].",
  "model": "openrouter/free",
  "collection": "investment_rules_openrouter-nvidia-nemotron-3-embed-1b-free",
  "top_k": 5,
  "sources": [
    {
      "source_number": 1,
      "source": "Investment Rule 2.pdf",
      "source_path": "C:\\Users\\batoo\\OneDrive\\Desktop\\Investment Rule 2.pdf",
      "score": 0.82,
      "page_start": 12,
      "page_end": 13,
      "section_title": "المادة ...",
      "snippet": "Retrieved source excerpt...",
      "metadata": {}
    }
  ]
}
```

## Embeddings

The project currently uses `local-hash-v1` by default. It is dependency-free and useful for proving that extraction, chunking, storage, and retrieval all work.

For zero-cost testing, the default OpenRouter embedding model is `nvidia/nemotron-3-embed-1b:free`. For stronger production-quality semantic search, you can switch to a paid embedding model such as OpenRouter `openai/text-embedding-3-small`, direct OpenAI `text-embedding-3-small`, sentence-transformers, or another multilingual embedding model. The embedding layer is isolated in `rag/services/embeddings.py` so this can be changed without rewriting the PDF extraction, chunking, or storage code.
