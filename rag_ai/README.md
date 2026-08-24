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
```

```powershell
python manage.py ingest_pdf "C:\Users\batoo\OneDrive\Desktop\Investment Rule.pdf" --embedding-provider openrouter
```

Optional settings:

```powershell
python manage.py ingest_pdf "C:\Users\batoo\OneDrive\Desktop\Investment Rule.pdf" --max-tokens 800 --overlap-tokens 120
```

The default Chroma persistence path is:

```text
rag_ai/data/chroma
```

By default, the command stores each embedding provider in a separate Chroma collection. For example, OpenRouter uses:

```text
investment_rules_openrouter
```

## Search indexed chunks

```powershell
python manage.py search_rag "ما هي شروط الاستثمار؟" --embedding-provider openrouter --top-k 5
```

This returns the most relevant Chroma chunks with page numbers. The next step is to pass those chunks to an LLM and instruct it to answer only from the retrieved source text.

## Embeddings

The project currently uses `local-hash-v1` by default. It is dependency-free and useful for proving that extraction, chunking, storage, and retrieval all work.

For production-quality semantic search, switch to a real embedding model such as OpenRouter `openai/text-embedding-3-small`, direct OpenAI `text-embedding-3-small`, sentence-transformers, or another multilingual embedding model. The embedding layer is isolated in `rag/services/embeddings.py` so this can be changed without rewriting the PDF extraction, chunking, or storage code.
