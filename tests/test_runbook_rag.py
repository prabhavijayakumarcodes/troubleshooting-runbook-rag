"""Offline tests: no API key or network needed.

Gemini is replaced by a deterministic bag-of-words embedding and a fake chat model, so these tests check
the plumbing (parsing, chunking, hybrid retrieval, chain wiring), not answer quality. Answer quality is
evaluated with the real models in the notebook.
"""

import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runbook_rag as rr  # noqa: E402


class BagOfWordsEmbeddings(Embeddings):
    """Hashed bag-of-words vectors: crude, but deterministic and good enough to exercise FAISS."""

    dim = 2048

    def _vec(self, text):
        v = np.zeros(self.dim)
        for tok in re.findall(r"[a-z0-9_]+", text.lower()):
            v[zlib.crc32(tok.encode()) % self.dim] += 1.0
        return (v / (np.linalg.norm(v) or 1.0)).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture(scope="module")
def chunks():
    return rr.section_chunks()


@pytest.fixture(scope="module")
def store(chunks):
    return rr.build_vector_store(chunks, BagOfWordsEmbeddings())


def test_every_error_code_has_exactly_one_section(chunks):
    codes = [c.metadata["error_code"] for c in chunks if c.metadata["error_code"]]
    assert sorted(codes) == sorted(rr.ERROR_CODES)


def test_page_headers_and_banners_removed(chunks):
    text = "\n".join(c.page_content for c in chunks)
    assert "Contoso Platform Engineering |" not in text
    assert not re.search(r"^Page \d+$", text, re.M)


def test_checklist_steps_are_not_mistaken_for_headings(chunks):
    sections = {c.metadata["section"] for c in chunks}
    assert "7. Standard first five minutes" in sections
    assert not any(s.startswith("2. Identify") for s in sections)


def test_tables_are_rebuilt_row_by_row(chunks):
    service_map = next(c for c in chunks if c.metadata["section"] == "3. Service map").page_content
    assert ("Service: search-service; Domain: Discovery; Owning team: Team Compass; "
            "Primary datastore: OpenSearch cluster") in service_map
    severity = next(c for c in chunks if c.metadata["section"] == "5. Severity matrix").page_content
    assert "Severity: Sev2; Definition: Major feature degraded" in severity
    assert "Ack target: 10 min" in severity


def test_sql_and_commands_keep_their_line_breaks(chunks):
    db = next(c for c in chunks if c.metadata["error_code"] == "DB-1205").page_content
    assert "\nFROM pg_stat_activity blocked\nJOIN pg_stat_activity blocking\n" in db


def test_every_chunk_has_citation_metadata(chunks):
    for c in chunks:
        assert c.metadata["source"] == "troubleshooting_runbook.pdf"
        assert 1 <= c.metadata["page_number"] <= 7


@pytest.mark.parametrize("text,expected", [
    ("Getting ERR-504 on payment-gateway", ["ERR-504"]),
    ("err 401 spikes", ["ERR-401"]),
    ("pod Last State: Terminated, Reason: OOMKilled, Exit Code: 137", ["MEM-OOM"]),
    ("write failed: No space left on device", ["DISK-FULL"]),
    ("remaining connection slots are reserved", ["DB-POOL"]),
    ("Break batch updates into chunks of 500 rows", []),
    ("What is the MTTR target?", []),
])
def test_detect_error_codes(text, expected):
    assert rr.detect_error_codes(text) == expected


def test_vector_store_contains_only_the_runbook(store, chunks):
    docs = list(store.docstore._dict.values())
    assert store.index.ntotal == len(chunks)
    assert {d.metadata["source"] for d in docs} == {"troubleshooting_runbook.pdf"}


def test_hybrid_retrieval_pins_the_error_code_section(store):
    hits = rr.RunbookRetriever(store, k=4).retrieve("ERR-504 on payment-gateway, what now?")
    assert hits[0].doc.metadata["error_code"] == "ERR-504"
    assert hits[0].reason == "error-code match"
    assert any(h.doc.metadata["section"] == "3. Service map" and h.reason == "service match" for h in hits)
    assert len(hits) == 4
    assert len({h.doc.metadata["chunk_id"] for h in hits}) == 4


def test_semantic_only_mode_has_no_pinned_hits(store):
    hits = rr.RunbookRetriever(store, k=3, hybrid=False).retrieve("ERR-504 on payment-gateway")
    assert len(hits) == 3 and all(h.reason == "semantic" for h in hits)
    assert hits == sorted(hits, key=lambda h: -h.score)


def test_evaluate_retrieval_runs(store):
    retriever = rr.RunbookRetriever(store, k=4)
    result = rr.evaluate_retrieval(lambda q: [h.doc for h in retriever.retrieve(q)])
    assert 0 <= result["mrr"] <= result["hit_rate"] <= 1
    assert len(result["rows"]) == len(rr.RETRIEVAL_EVAL)


def test_chain_passes_retrieved_context_to_the_llm(store):
    captured = {}

    class SpyLLM(FakeListChatModel):
        def _call(self, messages, *args, **kwargs):
            captured["prompt"] = "\n".join(m.content for m in messages)
            return super()._call(messages, *args, **kwargs)

    chain = rr.build_rag_chain(rr.RunbookRetriever(store, k=3), SpyLLM(responses=["stub answer [p.4]"]))
    out = chain.invoke({"question": "ERR-504 gateway timeout"})
    assert out["answer"] == "stub answer [p.4]"
    assert "[p.4 | ERR-504 - Gateway timeout]" in captured["prompt"]
    assert rr.NOT_FOUND_MSG in captured["prompt"]
    assert "p.4 | ERR-504 - Gateway timeout | error-code match" in rr.format_sources(out["hits"])
