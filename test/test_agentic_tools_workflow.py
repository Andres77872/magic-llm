import json
import os
import sys
from typing import Any, Dict, List, Union

import pytest
import requests

# add project root to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from magic_llm import MagicLLM
from magic_llm.model import ModelChat
from magic_llm.util import run_agentic

KEYS_FILE = os.getenv(
    "MAGIC_LLM_KEYS",
    "/home/andres/Documents/keys.json",
)


def _load_keys() -> Dict[str, Dict[str, str]]:
    if not os.path.exists(KEYS_FILE):
        pytest.skip(
            f"No keys file found at {KEYS_FILE}. "
            "Set MAGIC_LLM_KEYS env var or place keys.json in this directory.",
            allow_module_level=True,
        )
    with open(KEYS_FILE) as f:
        return json.load(f)


def _http_post_form(url: str, data: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    headers = {
        "accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    r = requests.post(url, data=data, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _http_post_json(url: str, body: Dict[str, Any], headers: Dict[str, str], timeout: int = 90) -> Dict[str, Any]:
    r = requests.post(url, json=body, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _png_to_jpg(url: str) -> str:
    u = url.strip()
    if u.endswith(".png"):
        u = u[:-4] + ".jpg"
    return u


def _select_provider_keys(keys: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    # Prefer OpenAI for function calling; fall back to Anthropic or DeepInfra
    for name in ("openai", "anthropic", "deepinfra"):
        if name in keys:
            return dict(keys[name])
    pytest.skip("No supported provider keys found in keys file.")


def _build_rewriter_tool(client: MagicLLM):
    def rewriter_querie(text: str, max_variants: int = 5) -> Dict[str, Any]:
        """Rewrite a search query into up to 5 variants and return JSON.

        Args:
            text: Original user query
            max_variants: Maximum number of variants to produce
        Returns:
            {"variants": [<string> ...]}
        """
        sys_prompt = (
            "You are a helpful assistant that rewrites the given query into up to 5 different,"
            " focused search variants."
            " Respond ONLY with a strict JSON array of strings."
            " Do not include any commentary."
        )
        chat = ModelChat(system=sys_prompt)
        chat.add_user_message(f"Rewrite this query into up to {max_variants} variants: {text}")
        # Ask the model for a normal content JSON response
        resp = client.llm.generate(chat, tool_choice="none")
        raw = resp.content or "[]"
        try:
            arr = json.loads(raw)
            if not isinstance(arr, list):
                arr = [str(raw)]
        except Exception:
            arr = [raw]
        # Trim and ensure uniqueness while preserving order
        seen = set()
        variants: List[str] = []
        for v in arr:
            s = str(v).strip()
            if s and s not in seen:
                variants.append(s)
                seen.add(s)
            if len(variants) >= max_variants:
                break
        return {"variants": variants}

    return rewriter_querie


def _search_papers_tool(query: Union[str, List[str]], limit: int = 10) -> Dict[str, Any]:
    """Search arXiv via ColPali endpoint for one or more queries.

    Behavior:
    - Deduplicates by page image across all queries
    - Groups results by (id, version) after deduplication to aid LLM prompting

    Returns a dict with:
      - data: list of unique documents (backward compatible with existing tools)
      - papers: {"<id>v<version>": {content: str, pages: {"<page>": {image: jpg_url}}}}
      - index: stable list of paper keys in insertion order
    """
    queries = [query] if isinstance(query, str) else list(query)
    seen_images = set()
    aggregated: List[Dict[str, Any]] = []

    for q in queries:
        payload = {"query": q, "limit": limit}
        res = _http_post_form("https://llm.arz.ai/rag/colpali/arxiv", payload)
        for doc in res.get("data", []):
            img = str(doc.get("page_image", "")).strip()
            if img and img not in seen_images:
                aggregated.append(doc)
                seen_images.add(img)

    # Build grouped structure by (id, version)
    papers: Dict[str, Dict[str, Any]] = {}
    index: List[str] = []

    for d in aggregated:
        arxiv_id = str(d.get("id", "")).strip()
        version = str(d.get("version", "")).strip()
        # compute paper key like 2308.11628v1
        key = f"{arxiv_id}v{version}" if arxiv_id and version else (arxiv_id or d.get("url", "")).strip()

        if key not in papers:
            title = str(d.get("title", "")).strip()
            authors = str(d.get("authors", "")).strip()
            date = str(d.get("date", "")).strip()
            url = str(d.get("url", "")).strip()
            abstract = str(d.get("abstract", "")).strip()
            # Concise, LLM-friendly content string
            content = (
                f"{title}\n"
                f"Authors: {authors}\n"
                f"Date: {date}\n"
                f"ArXiv: {key}\n"
                f"URL: {url}\n\n"
                f"Abstract: {abstract}"
            ).strip()

            papers[key] = {
                "content": content,
                "pages": {},  # filled below
            }
            index.append(key)

        page_num = d.get("page")
        try:
            page_key = str(int(page_num)) if page_num is not None else None
        except Exception:
            page_key = str(page_num) if page_num is not None else None

        if page_key is not None:
            jpg_url = _png_to_jpg(str(d.get("page_image", "")))
            papers[key]["pages"][page_key] = {"image": jpg_url}

    return {"data": aggregated, "papers": papers, "index": index}


def _reranker_tool(
        query: str,
        documents: List[Dict[str, Any]],
        jina_api_key: Union[str, None] = None,
        min_score: float = 0.85,
        top_k: int = 16,
) -> Dict[str, Any]:
    """Rerank images using Jina Reranker with JPG thumbnails.

    - Converts all page_image URLs to .jpg (optimized) before sending to Jina.
    - Keeps documents with score >= min_score, limited to top_k.
    - Returns selected metadata: page, page_image (jpg), abstract, title, date, authors.
    """
    if not jina_api_key:
        jina_api_key = 'jina_'
    if not jina_api_key:
        pytest.skip("JINA_API_KEY is required for reranker tool")

    images: List[str] = []
    for d in documents:
        page_img = str(d.get("page_image", ""))
        if page_img:
            images.append(_png_to_jpg(page_img))

    body = {
        "model": "jina-reranker-m0",
        "query": query,
        "documents": [{"image": u} for u in images],
        "return_documents": False,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {jina_api_key}",
    }
    resp = _http_post_json("https://api.jina.ai/v1/rerank", body, headers)
    results = resp.get("results", [])

    # Sort results by relevance score desc and filter
    results_sorted = sorted(results, key=lambda r: r.get("relevance_score", 0), reverse=True)
    selected: List[Dict[str, Any]] = []
    for r in results_sorted:
        score = float(r.get("relevance_score", 0))
        if score < float(min_score):
            continue
        idx = int(r.get("index", -1))
        if 0 <= idx < len(documents):
            d = documents[idx]
            selected.append({
                "id": d.get("id"),
                "version": d.get("version"),
                "page": d.get("page"),
                "page_image": _png_to_jpg(str(d.get("page_image", ""))),
                "abstract": d.get("abstract"),
                "title": d.get("title"),
                "date": d.get("date"),
                "authors": d.get("authors"),
                "url": d.get("url"),
                "score": score,
            })
        if len(selected) >= int(top_k):
            break

    return {"results": selected}


@pytest.mark.timeout(180)
def test_agentic_tools_end_to_end_search_and_rerank():
    keys = _load_keys()
    key = keys['openrouter']
    client = MagicLLM(model='openai/gpt-4.1', **key)

    # Define tools as callables
    rewriter_querie = _build_rewriter_tool(client)
    search_papers = _search_papers_tool
    reranker = _reranker_tool

    tools = [rewriter_querie, search_papers, reranker]

    system_prompt = (
        "You are a research assistant. Use tools to accomplish the task:"
        " 1) If the input is ambiguous, call rewriter_querie to generate up to 5 variants;"
        " 2) Call search_papers with either the original query or the variants array to fetch results;"
        " 3) Call reranker with a rerank query and the returned documents;"
        " Finally, summarize the most relevant papers."
    )

    user_input = (
        "Find relevant arXiv papers about 'ai on education'."
        " Then rerank the images for the query 'small language model data extraction'."
        " Provide a short summary at the end."
    )

    # Let the agent decide when to call tools
    resp = run_agentic(
        client=client,
        user_input=user_input,
        system_prompt=system_prompt,
        tools=tools,
        tool_choice="auto",
        max_iterations=6,
    )

    # Minimal sanity checks; we primarily want a real run + printed outputs
    assert resp is not None
    assert isinstance(resp.content, (str, type(None)))
    print("Model:", resp.model)
    print("Finish reason:", resp.finish_reason)
    print("Content:\n", resp.content)

    # --- Post-rerank image analysis demo ----------------------------------
    # Build a vision prompt by sending the top-16 reranked images with their metadata
    # Use the same client as above and the image-handling pattern from image tests
    search_query = "ai on education"
    rerank_query = search_query

    # Search and dedupe
    search_result = _search_papers_tool(search_query, limit=32)
    documents = list(search_result.get("data", []))

    # Rerank and select top 16 with metadata
    rerank_result = _reranker_tool(query=rerank_query, documents=documents, top_k=16, min_score=0.0)
    top_images = list(rerank_result.get("results", []))[:16]

    if top_images:
        vision_chat = ModelChat()
        # Initial instruction message
        vision_chat.add_user_message(
            "Analyze these reranked arXiv page images and summarize key insights.\n"
            "For each, consider the title, authors, and abstract provided alongside the image."
        )

        # Add each image with its own metadata as a separate message
        for idx, item in enumerate(top_images, start=1):
            meta = (
                f"[{idx}] {str(item.get('id') or '').strip()}v{str(item.get('version') or '').strip()}\n"
                f"Title: {str(item.get('title') or '').strip()}\n"
                f"Authors: {str(item.get('authors') or '').strip()}\n"
                f"URL: {str(item.get('url') or '').strip()}\n"
                f"Abstract: {str(item.get('abstract') or '').strip()}"
            )
            image_url = str(item.get("page_image", "")).strip()
            if image_url:
                vision_chat.add_user_message(meta, image=image_url)

        # Generate a vision-enabled response
        vision_resp = client.llm.generate(vision_chat)
        assert vision_resp.content, "Expected non-empty content from vision analysis"
        print("Vision analysis summary:\n", vision_resp.content)
