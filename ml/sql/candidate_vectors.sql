-- Stored vectors of a candidate set under the given models, for exact
-- similarity features. Shared by Python and Go.
SELECT episode_id, model, embedding FROM episode_embeddings
WHERE episode_id = ANY(%(ids)s) AND model = ANY(%(models)s)
