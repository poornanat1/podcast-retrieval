// Command migrate applies the embedded SQL migrations to DATABASE_URL.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"podfind/internal/migrate"
	"podfind/migrations"
)

const defaultDatabaseURL = "postgres://podfind:podfind@localhost:5432/podfind?sslmode=disable"

func main() {
	url := os.Getenv("DATABASE_URL")
	if url == "" {
		url = defaultDatabaseURL
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	pool, err := pgxpool.New(ctx, url)
	if err != nil {
		log.Fatalf("connect: %v", err)
	}
	defer pool.Close()

	applied, err := migrate.Apply(ctx, pool, migrations.FS)
	if err != nil {
		log.Fatalf("migrate: %v", err)
	}
	if len(applied) == 0 {
		fmt.Println("migrations: up to date")
		return
	}
	for _, name := range applied {
		fmt.Println("applied", name)
	}
}
