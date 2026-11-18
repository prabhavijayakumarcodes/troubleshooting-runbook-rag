# Troubleshooting Runbook RAG Assistant

An on-call assistant that answers questions **only** from the *Contoso Platform Troubleshooting Runbook v3.2* (`data/troubleshooting_runbook.pdf`). It uses Retrieval-Augmented Generation (RAG) built with LangChain, Google Gemini and FAISS.

You can ask a question, paste an alert or paste a log line. The assistant replies in the runbook's own triage order: **error code → likely causes → diagnostics (in order) → fix → prevention → owner / escalation**. It cites a page (`[p.X]`) for every point and copies commands verbatim. If the runbook does not cover something, it says so instead of guessing.

```text
$ python ask.py "Pod event: Reason: OOMKilled, Exit Code: 137 on reporting-engine"
```

## What's inside

| File | Purpose |
|---|---|
| [`Runbook_RAG_Assistant.ipynb`](Runbook_RAG_Assistant.ipynb) | Step-by-step walkthrough: loading, cleaning, chunking experiment, embeddings, FAISS, hybrid retrieval, grounded prompt, demos and evaluation |
| [`runbook_rag.py`](runbook_rag.py) | Reusable pipeline used by the notebook, the CLI and the tests |
| [`ask.py`](ask.py) | Command-line assistant (one-shot or interactive) |
| [`tests/`](tests) | 18 offline unit tests (no API key needed) |
| `data/troubleshooting_runbook.pdf` | The only knowledge source |

## How it works

1. **Load and clean.** `PyPDFLoader` reads the pages. Repeated page headers and footers are removed, and PDF line wrapping is re-joined. SQL, `kubectl` commands and numbered steps keep their line breaks.
2. **Rebuild tables.** `PyPDFLoader` flattens the service map, the severity matrix and the quick reference to one cell per line, which loses which team owns which service. `pdfplumber` re-extracts them as `Header: value; ...` rows.
3. **Section-aware chunking.** The text is split by the runbook's own structure, so each error code (`ERR-401` … `CERT-EXP`) and each numbered or lettered section becomes one chunk. The metadata records `page_number`, `section`, `category` and `error_code`. `RecursiveCharacterTextSplitter` is used only as a fallback for sections longer than 1,600 characters.
4. **Embed and index.** Gemini embeddings (`models/gemini-embedding-001`) go into an L2-normalised FAISS index, so FAISS distance maps to cosine similarity. An assertion checks that the index contains only the runbook. The index is saved with a fingerprint of the PDF and the settings, and rebuilt automatically when either changes.
5. **Hybrid retrieval.**
   - **Error-code match:** if the question names a code, or contains one of the runbook's symptom signatures (`upstream connect error`, `Exit Code: 137`, `No space left on device`, …), that section is always returned first.
   - **Service match:** if a service is named, the service map is added.
   - **Semantic search** fills the remaining top-k slots.

   This avoids mixing up neighbouring codes, such as ERR-503 and ERR-504, which both mention payment-gateway but need different fixes.
6. **Grounded prompt.** The model answers from the context only and never invents commands, thresholds or team names. It replies with a fixed message when the runbook doesn't cover a question and flags partly covered answers. It treats retrieved text as data, not instructions. Temperature is 0.
7. **Evaluation.**
   - **Retrieval:** hit rate and MRR on 20 questions with known answers. The notebook compares page-based chunking at 300, 500, 800 and 1,200 characters, section-aware chunking, and section-aware chunking with hybrid retrieval.
   - **Answers:** checks that answers contain the expected facts and a page citation, that partly covered questions are flagged, and that out-of-scope questions are refused.

## How to run

```bash
git clone https://github.com/prabhavijayakumarcodes/troubleshooting-runbook-rag.git
cd troubleshooting-runbook-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You need a Google Gemini API key (free at <https://aistudio.google.com/apikey>). Set it as an environment variable, or paste it when prompted. Keys are never hard-coded.

```bash
export GOOGLE_API_KEY="your-key-here"
```

**Notebook**

```bash
jupyter notebook Runbook_RAG_Assistant.ipynb
```

Then choose **Run All**.

**CLI**

```bash
python ask.py "payment-gateway is throwing ERR-504, walk me through it"
python ask.py                      # interactive mode
python ask.py --semantic-only ...  # compare against pure semantic retrieval
```

**Tests** (offline: they use a local bag-of-words embedding and a fake LLM)

```bash
pytest -q tests
```

To use different models, set `RUNBOOK_EMBEDDING_MODEL` or `RUNBOOK_LLM_MODEL`; no code change is needed.

## Inconsistencies found in the runbook

The assistant answers from the runbook exactly as written, so these are worth fixing in the source document:

- Section 2 says to escalate *"according to the severity matrix in Section 3"*, but the matrix is in **Section 5**. Section 3 is the service map.
- ERR-504 points to *"Section 5 database errors"*, but Section 5 is the severity matrix. The database errors are on page 5 and have no section number.
- Some error codes are missing template fields:
  - ERR-503 has no *Likely causes* and no *Prevention*.
  - ERR-504 has no *Prevention*.
  - DISK-FULL and CERT-EXP have no *Likely causes*.

## Limitations

- The section and symptom patterns are tuned to this runbook's template. Runbooks in a different format need new patterns.
- For a larger collection of runbooks, the next steps are a reranker, metadata filters (by service or category) and running the evaluation in CI.

The runbook is synthetic demo data.
