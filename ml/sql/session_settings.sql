-- Per-connection vector search settings, applied by Python
-- (ml.retrieval.vector.enable_iterative_scan) and by the Go API's connection
-- pool. Iterative scans keep filtered HNSW queries scanning until LIMIT is
-- met. ef_search 1000: measured on the relevance set's unfiltered queries
-- against an exact scan (2026-10-21), recall of the true top 10 rose from
-- 0.83 to 0.89 (two-tower) and 0.85 to 0.92 (E5) going from 200, with no
-- change in median latency. One statement per line.
SET hnsw.iterative_scan = relaxed_order;
SET hnsw.ef_search = 1000;
