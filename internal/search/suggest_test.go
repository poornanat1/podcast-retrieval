package search

import "testing"

func TestMatchesPrefixAndEscape(t *testing.T) {
	if !matchesPrefix("the history of rome", "hist") || !matchesPrefix("history hit", "hist") || matchesPrefix("prehistory", "hist") {
		t.Error("word-prefix matching")
	}
	if got := escapeLike(`50% _of\ it`); got != `50\% \_of\\ it` {
		t.Errorf("escapeLike = %q", got)
	}
}
