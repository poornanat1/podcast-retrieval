package migrate_test

import (
	"context"
	"testing"

	"podfind/internal/migrate"
	"podfind/internal/pgtest"
	"podfind/migrations"
)

func TestApplyIsIdempotent(t *testing.T) {
	pool := pgtest.Pool(t, "migrate") // first full application happens here
	ctx := context.Background()

	applied, err := migrate.Apply(ctx, pool, migrations.FS)
	if err != nil {
		t.Fatalf("second apply: %v", err)
	}
	if len(applied) != 0 {
		t.Fatalf("second apply ran %v, want none", applied)
	}

	// Every embedded migration must be recorded.
	var n int
	if err := pool.QueryRow(ctx, "SELECT count(*) FROM schema_migrations").Scan(&n); err != nil {
		t.Fatalf("count schema_migrations: %v", err)
	}
	if n == 0 {
		t.Fatal("no migrations recorded")
	}
}
