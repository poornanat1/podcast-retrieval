// Package pgtest provides per-package test databases with the full schema
// applied. Tests skip when Postgres is unreachable locally but fail in CI,
// so a misconfigured CI service can never silently skip coverage.
package pgtest

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"podfind/internal/migrate"
	"podfind/migrations"
)

const defaultAdminURL = "postgres://podfind:podfind@localhost:5432/podfind?sslmode=disable"

// Pool returns a pool connected to a dedicated database named
// podfind_test_<name>, freshly migrated. Each test package should pass its
// own name so packages can run in parallel without sharing state.
func Pool(t *testing.T, name string) *pgxpool.Pool {
	t.Helper()

	adminURL := os.Getenv("PODFIND_TEST_ADMIN_URL")
	if adminURL == "" {
		adminURL = defaultAdminURL
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	admin, err := pgxpool.New(ctx, adminURL)
	if err == nil {
		err = admin.Ping(ctx)
	}
	if err != nil {
		if os.Getenv("CI") != "" {
			t.Fatalf("postgres unreachable in CI at %s: %v", adminURL, err)
		}
		t.Skipf("postgres unreachable at %s: %v (start it with `docker compose up -d`)", adminURL, err)
	}
	defer admin.Close()

	dbName := "podfind_test_" + name
	var exists bool
	if err := admin.QueryRow(ctx,
		"SELECT EXISTS (SELECT 1 FROM pg_database WHERE datname = $1)", dbName,
	).Scan(&exists); err != nil {
		t.Fatalf("check test database: %v", err)
	}
	if !exists {
		if _, err := admin.Exec(ctx, "CREATE DATABASE "+dbName); err != nil {
			t.Fatalf("create test database: %v", err)
		}
	}

	cfg, err := pgxpool.ParseConfig(adminURL)
	if err != nil {
		t.Fatalf("parse admin url: %v", err)
	}
	cfg.ConnConfig.Database = dbName

	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		t.Fatalf("connect to %s: %v", dbName, err)
	}
	t.Cleanup(pool.Close)

	// Reset to a clean schema, then apply all migrations.
	if _, err := pool.Exec(ctx, "DROP SCHEMA public CASCADE; CREATE SCHEMA public"); err != nil {
		t.Fatalf("reset schema: %v", err)
	}
	if _, err := migrate.Apply(ctx, pool, migrations.FS); err != nil {
		t.Fatalf("apply migrations: %v", err)
	}

	return pool
}
