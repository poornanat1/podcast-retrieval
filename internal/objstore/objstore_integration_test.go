package objstore

import (
	"context"
	"errors"
	"os"
	"testing"
)

// Runs only against a configured store (OBJECT_STORE_ENDPOINT set), e.g.
// the compose objstore service.
func TestPutGetDelete(t *testing.T) {
	if os.Getenv("OBJECT_STORE_ENDPOINT") == "" {
		t.Skip("OBJECT_STORE_ENDPOINT not set")
	}
	store, err := NewFromEnv()
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	key := "test/objstore-" + t.Name()
	if err := store.Put(ctx, key, []byte("hello"), "text/plain"); err != nil {
		t.Fatal(err)
	}
	got, err := store.Get(ctx, key)
	if err != nil || string(got) != "hello" {
		t.Fatalf("get: %q, %v", got, err)
	}
	if err := store.Delete(ctx, key); err != nil {
		t.Fatal(err)
	}
	if err := store.Delete(ctx, key); err != nil {
		t.Fatalf("second delete: %v", err)
	}
	if _, err := store.Get(ctx, key); !errors.Is(err, ErrNotFound) {
		t.Fatalf("get after delete: %v", err)
	}
}
