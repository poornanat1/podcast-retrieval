// Package objstore stores raw artifacts (transcript files, later datasets
// and models) in S3-compatible object storage.
package objstore

import (
	"bytes"
	"context"
	"fmt"
	"os"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

// Store writes objects into one bucket.
type Store struct {
	client *minio.Client
	bucket string
}

// NewFromEnv builds a Store from OBJECT_STORE_* environment variables. It
// returns (nil, nil) when OBJECT_STORE_ENDPOINT is unset, letting callers
// treat raw-artifact retention as optional.
func NewFromEnv() (*Store, error) {
	endpoint := os.Getenv("OBJECT_STORE_ENDPOINT")
	if endpoint == "" {
		return nil, nil
	}
	bucket := os.Getenv("OBJECT_STORE_BUCKET")
	if bucket == "" {
		return nil, fmt.Errorf("OBJECT_STORE_BUCKET is required when OBJECT_STORE_ENDPOINT is set")
	}
	client, err := minio.New(endpoint, &minio.Options{
		Creds: credentials.NewStaticV4(
			os.Getenv("OBJECT_STORE_ACCESS_KEY"),
			os.Getenv("OBJECT_STORE_SECRET_KEY"), ""),
		Secure: os.Getenv("OBJECT_STORE_USE_SSL") == "true",
	})
	if err != nil {
		return nil, fmt.Errorf("object store client: %w", err)
	}
	return &Store{client: client, bucket: bucket}, nil
}

// Put writes data under key, overwriting any existing object.
func (s *Store) Put(ctx context.Context, key string, data []byte, contentType string) error {
	_, err := s.client.PutObject(ctx, s.bucket, key, bytes.NewReader(data), int64(len(data)),
		minio.PutObjectOptions{ContentType: contentType})
	if err != nil {
		return fmt.Errorf("put %s/%s: %w", s.bucket, key, err)
	}
	return nil
}
