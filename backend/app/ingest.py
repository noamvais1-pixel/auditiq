"""Build the on-disk index. Run with `python -m app.ingest`."""

from __future__ import annotations

import logging
import sys
from collections import Counter

from .config import settings
from .rag import Index, load_corpus


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("ingest")

    chunks = load_corpus(settings.corpus_dir)
    if not chunks:
        log.error("no documents found in %s", settings.corpus_dir)
        return 1

    index = Index().build(chunks)
    index.save(settings.index_path)

    per_doc = Counter(c.doc_id for c in chunks)
    lengths = sorted(len(c.text) for c in chunks)
    log.info("indexed %d chunks from %d documents", len(chunks), len(per_doc))
    log.info("embedding provider: %s (dim=%d)", index.provider.name, index.provider.dim)
    log.info(
        "chunk length: min=%d median=%d max=%d",
        lengths[0], lengths[len(lengths) // 2], lengths[-1],
    )
    for doc, n in per_doc.most_common():
        log.info("  %-24s %3d chunks", doc, n)
    log.info("written to %s", settings.index_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
