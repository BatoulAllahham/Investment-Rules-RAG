from __future__ import annotations

import re


def collection_name_for_provider(base_name: str, embedding_provider: str) -> str:
    provider_slug = re.sub(r"[^a-z0-9_-]+", "_", embedding_provider.strip().lower())
    provider_slug = provider_slug.strip("_-") or "default"
    collection_name = f"{base_name}_{provider_slug}"
    collection_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", collection_name).strip("_-")

    if len(collection_name) > 63:
        collection_name = collection_name[:63].rstrip("_-")
    if len(collection_name) < 3:
        collection_name = f"{collection_name}_rag"

    return collection_name
