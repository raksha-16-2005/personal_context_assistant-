"""The message row schema, in one place.

Both corpora produce rows of this exact shape - `enron.parse_message` and
`gmail.fetch_messages` (which is a thin wrapper *around* `enron.parse_message`,
see gmail.py's own docstring) - so `scripts/build_corpus.py` and the
multi-tenant web app's per-user ingestion worker (webapp/app/ingestion) both
write Parquet with this schema rather than trusting pyarrow's type inference
to agree with itself twice. `date_utc` in particular has to be forced to
`timestamp("us", tz="UTC")` explicitly: inference from a column of Python
`datetime` objects is not guaranteed to pick the same tz-awareness every time,
and `Pipeline`/`chunktext` downstream both assume it.
"""
from __future__ import annotations

import pyarrow as pa

MESSAGE_SCHEMA = pa.schema([
    ("message_id", pa.string()),
    ("dedup_key", pa.string()),
    ("date_utc", pa.timestamp("us", tz="UTC")),
    ("sender", pa.string()),
    ("recipients", pa.string()),
    ("cc", pa.string()),
    ("subject", pa.string()),
    ("subject_norm", pa.string()),
    ("body", pa.string()),
    ("body_new", pa.string()),
    ("in_reply_to", pa.string()),
    ("references", pa.string()),
    ("has_list_unsubscribe", pa.bool_()),
    ("source_path", pa.string()),
    ("owner", pa.string()),
    ("folder", pa.string()),
])
