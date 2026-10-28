// Package objstore stores raw artifacts (transcript files, retained voice
// recordings, datasets and models) in S3-compatible object storage. It
// speaks the S3 API only, so the same code runs against a local RustFS
// container, a Railway bucket, Cloudflare R2, or S3 itself.
package objstore

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/aws-sdk-go-v2/service/s3/types"
)

// Config is read from OBJECT_STORE_* environment variables:
//
//	OBJECT_STORE_ENDPOINT    host:port or URL of the S3 endpoint (unset disables the store)
//	OBJECT_STORE_BUCKET      bucket name
//	OBJECT_STORE_ACCESS_KEY, OBJECT_STORE_SECRET_KEY
//	OBJECT_STORE_REGION      signing region (default us-east-1; Railway buckets use auto)
//	OBJECT_STORE_USE_SSL     true forces https when the endpoint has no scheme
//	OBJECT_STORE_PATH_STYLE  true addresses buckets by path (default: true for
//	                         plain host:port endpoints, false for URLs)
type Config struct {
	Endpoint  string
	Bucket    string
	AccessKey string
	SecretKey string
	Region    string
	PathStyle bool
}

// ConfigFromEnv reads the OBJECT_STORE_* variables. ok is false when no
// endpoint is configured.
func ConfigFromEnv() (cfg Config, ok bool, err error) {
	raw := os.Getenv("OBJECT_STORE_ENDPOINT")
	if raw == "" {
		return cfg, false, nil
	}
	cfg = Config{
		Bucket:    os.Getenv("OBJECT_STORE_BUCKET"),
		AccessKey: os.Getenv("OBJECT_STORE_ACCESS_KEY"),
		SecretKey: os.Getenv("OBJECT_STORE_SECRET_KEY"),
		Region:    os.Getenv("OBJECT_STORE_REGION"),
	}
	if cfg.Bucket == "" {
		return cfg, false, errors.New("OBJECT_STORE_BUCKET is required when OBJECT_STORE_ENDPOINT is set")
	}
	cfg.Endpoint, cfg.PathStyle = EndpointURL(raw, os.Getenv("OBJECT_STORE_USE_SSL") == "true")
	if v := os.Getenv("OBJECT_STORE_PATH_STYLE"); v != "" {
		cfg.PathStyle = v == "true"
	}
	if cfg.Region == "" {
		cfg.Region = "us-east-1"
	}
	return cfg, true, nil
}

// EndpointURL turns a plain host:port into a URL (http, or https when ssl
// is set) and leaves URLs alone. Plain endpoints are local stores, which
// are addressed by path; URLs are managed buckets, which are addressed by
// virtual host.
func EndpointURL(raw string, ssl bool) (url string, pathStyle bool) {
	if strings.Contains(raw, "://") {
		return raw, false
	}
	if ssl {
		return "https://" + raw, true
	}
	return "http://" + raw, true
}

// Store writes objects into one bucket.
type Store struct {
	client *s3.Client
	bucket string
}

// New builds a Store for cfg.
func New(cfg Config) *Store {
	client := s3.New(s3.Options{
		BaseEndpoint: aws.String(cfg.Endpoint),
		Region:       cfg.Region,
		Credentials:  credentials.NewStaticCredentialsProvider(cfg.AccessKey, cfg.SecretKey, ""),
		UsePathStyle: cfg.PathStyle,
	})
	return &Store{client: client, bucket: cfg.Bucket}
}

// NewFromEnv builds a Store from OBJECT_STORE_* environment variables. It
// returns (nil, nil) when OBJECT_STORE_ENDPOINT is unset, letting callers
// treat raw-artifact retention as optional.
func NewFromEnv() (*Store, error) {
	cfg, ok, err := ConfigFromEnv()
	if err != nil || !ok {
		return nil, err
	}
	return New(cfg), nil
}

// Put writes data under key, overwriting any existing object.
func (s *Store) Put(ctx context.Context, key string, data []byte, contentType string) error {
	_, err := s.client.PutObject(ctx, &s3.PutObjectInput{
		Bucket:        aws.String(s.bucket),
		Key:           aws.String(key),
		Body:          bytes.NewReader(data),
		ContentLength: aws.Int64(int64(len(data))),
		ContentType:   aws.String(contentType),
	})
	if err != nil {
		return fmt.Errorf("put %s/%s: %w", s.bucket, key, err)
	}
	return nil
}

// Get reads the object under key. A missing object returns ErrNotFound.
func (s *Store) Get(ctx context.Context, key string) ([]byte, error) {
	out, err := s.client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(s.bucket), Key: aws.String(key),
	})
	if err != nil {
		var nsk *types.NoSuchKey
		if errors.As(err, &nsk) {
			return nil, fmt.Errorf("get %s/%s: %w", s.bucket, key, ErrNotFound)
		}
		return nil, fmt.Errorf("get %s/%s: %w", s.bucket, key, err)
	}
	defer out.Body.Close()
	var buf bytes.Buffer
	if _, err := buf.ReadFrom(out.Body); err != nil {
		return nil, fmt.Errorf("read %s/%s: %w", s.bucket, key, err)
	}
	return buf.Bytes(), nil
}

// ErrNotFound marks a missing object.
var ErrNotFound = errors.New("object not found")

// Delete removes key; a missing object is not an error.
func (s *Store) Delete(ctx context.Context, key string) error {
	_, err := s.client.DeleteObject(ctx, &s3.DeleteObjectInput{
		Bucket: aws.String(s.bucket), Key: aws.String(key),
	})
	if err != nil {
		return fmt.Errorf("delete %s/%s: %w", s.bucket, key, err)
	}
	return nil
}
