# Investment Rules RAG

This Django project now has a small RAG ingestion pipeline in the `rag` app.

## What the pipeline does

1. Extracts selectable text from a PDF page by page.
2. Cleans repeated headers, footers, empty lines, and spacing noise.
3. Splits the text into legal-aware chunks, preferring article boundaries.
4. Creates one embedding vector for each chunk.
5. Stores the source document, chunk metadata, chunk text, and vectors in a local Chroma vector database.
6. Searches the stored chunks by comparing a customer question vector to the chunk vectors.

## Why chunking is done this way

The chunker uses a hybrid legal strategy. It first looks for real article headers such as `المادة (١)` and the reversed PDF-extraction form `:/ 1 المادة`. When enough article headers are found, each article becomes its own chunk with stable legal metadata. If the text does not have reliable article markers, the chunker falls back to paragraph/section chunks.

The default chunk size is about 800 tokens. Normal articles stay together whenever possible. Very long articles are split into smaller parts while keeping the same article metadata. Non-article fallback sections still use 120 tokens of overlap.

Each chunk stores:

- source PDF name
- page start and page end
- section title, when detected
- chunk type: article, article part, section, or text
- legal source type, such as law, decree, or decision, when detected
- document number and year, when detected
- chapter, when detected
- article number, when detected
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
OPENROUTER_EMBEDDING_MODEL=baai/bge-m3
RAG_CHAT_MODEL=openrouter/free
RAG_DEFAULT_TOP_K=5
RAG_REQUEST_TIMEOUT=180
RAG_EMBEDDING_TIMEOUT=120
RAG_SOURCE_MAX_CHARS=1200
RAG_MAX_OUTPUT_TOKENS=700
```

```powershell
python manage.py ingest_pdf --reset
```

By default, this indexes the three PDFs in the project `resources` folder:

- `resources/Investment Rule.pdf`
- `resources/Investment Rule 2.pdf`
- `resources/Investment Rule 3.pdf`

You can still pass explicit PDF paths if you want to index different files:

```powershell
python manage.py ingest_pdf "C:\Users\batoo\PycharmProjects\Investment-Rules-RAG\resources\Investment Rule.pdf" "C:\Users\batoo\PycharmProjects\Investment-Rules-RAG\resources\Investment Rule 2.pdf" "C:\Users\batoo\PycharmProjects\Investment-Rules-RAG\resources\Investment Rule 3.pdf" --reset
```

All three source PDFs are treated as selectable/searchable PDFs. OCR is no longer part of the ingestion flow.

Optional settings:

```powershell
python manage.py ingest_pdf --max-tokens 800 --overlap-tokens 120
```

The default Chroma persistence path is:

```text
rag_ai/data/chroma
```

By default, the command stores each embedding provider and model in a separate Chroma collection. For example, the default OpenRouter BAAI/bge-m3 embedding model uses:

```text
investment_rules_openrouter-baai-bge-m3
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

This retrieves the top matching chunks from Chroma, sends only those chunks to the configured OpenRouter chat model, and returns a plain answer. Source metadata is returned separately by the API.

The default answer model is:

```text
openrouter/free
```

This uses OpenRouter's free model router. Free models are useful for demos and low-volume testing, but they can have lower rate limits, changing availability, slower responses, and less predictable answer quality than paid models.

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
  "answer": "بناء على النصوص المسترجعة من القانون، فإن شروط الاستثمار تشمل تقديم الوثائق المطلوبة وارتباط المشروع بالقطاعات المشمولة بأحكام القانون.",
  "model": "openrouter/free",
  "collection": "investment_rules_openrouter-baai-bge-m3",
  "top_k": 5,
  "sources": [
    {
      "source_number": 1,
      "source": "Investment Rule 2.pdf",
      "source_path": "C:\\Users\\batoo\\PycharmProjects\\Investment-Rules-RAG\\resources\\Investment Rule 2.pdf",
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

The project can still use `local-hash-v1` if you explicitly request `--embedding-provider local`; it is dependency-free and useful only for proving that extraction, chunking, storage, and retrieval work.

The default OpenRouter embedding model is `baai/bge-m3`, which returns 1024-dimensional vectors and is strong for multilingual retrieval. The chat answer model remains `openrouter/free`. If you need zero-cost embeddings again, set `OPENROUTER_EMBEDDING_MODEL` back to a free embedding model and re-index into that model's collection.
