package search

import "testing"

func TestIndexName(t *testing.T) {
	cases := map[string]string{
		"intfloat/multilingual-e5-small": "episode_embeddings_hnsw_intfloat_multilingual_e5_small",
		"two-tower:two-tower-llm-names-1.0.0-56e2c0708f7e-3b1c13c6": "episode_embeddings_hnsw_two_tower_two_tower_llm_names_1_0_0_56e",
	}
	for model, want := range cases {
		if got := IndexName(model); got != want {
			t.Errorf("IndexName(%q) = %q, want %q", model, got, want)
		}
	}
}
