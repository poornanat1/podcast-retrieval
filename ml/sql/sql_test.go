package sql

import (
	"strings"
	"testing"
)

func TestRewrite(t *testing.T) {
	got := Rewrite("WHERE a LIKE %(lang)s || '%%' AND b = %(top_k)s AND c = %(lang)s")
	want := "WHERE a LIKE @lang || '%' AND b = @top_k AND c = @lang"
	if got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestSharedStatementsLoad(t *testing.T) {
	for _, name := range []string{"lexical_search", "vector_search", "candidate_evidence", "candidate_vectors"} {
		q, err := Named(name)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(q, "%(") || !strings.Contains(q, "@") {
			t.Errorf("%s: placeholders not rewritten", name)
		}
	}
}

func TestSessionStatements(t *testing.T) {
	got := SessionStatements()
	if len(got) != 2 || got[1] != "SET hnsw.ef_search = 1000" {
		t.Fatalf("got %q", got)
	}
}
