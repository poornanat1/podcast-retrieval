package rss

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

// ErrUnsafeURL marks a URL the crawler refuses to fetch: anything that is
// not http(s), or that resolves to an address inside our own network.
// Feed XML is publisher-controlled, so transcript links and redirects
// could otherwise point the worker at internal services.
var ErrUnsafeURL = errors.New("url is not a public http(s) address")

const maxRedirects = 5

// publicIP reports whether ip is globally routable. Loopback, private,
// link-local, multicast, carrier-grade NAT, and documentation ranges are
// all refused.
func publicIP(ip net.IP) bool {
	if ip == nil || ip.IsUnspecified() || ip.IsLoopback() || ip.IsPrivate() ||
		ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() || ip.IsMulticast() ||
		ip.IsInterfaceLocalMulticast() {
		return false
	}
	if v4 := ip.To4(); v4 != nil {
		switch {
		case v4[0] == 0: // 0.0.0.0/8
			return false
		case v4[0] == 100 && v4[1]&0xc0 == 64: // 100.64.0.0/10, carrier-grade NAT
			return false
		case v4[0] == 192 && v4[1] == 0 && v4[2] == 0: // 192.0.0.0/24
			return false
		case v4[0] == 198 && v4[1]&0xfe == 18: // 198.18.0.0/15, benchmarking
			return false
		case v4[0] >= 240: // reserved and broadcast
			return false
		}
		return true
	}
	// Unique local addresses (fc00::/7) are "private" to Go already; the
	// remaining ranges Go does not classify.
	if ip[0] == 0x20 && ip[1] == 0x01 && ip[2] == 0x0d && ip[3] == 0xb8 { // 2001:db8::/32
		return false
	}
	return true
}

// checkURL refuses non-http(s) URLs and literal non-public hosts before
// any connection is attempted.
func checkURL(raw string) error {
	u, err := url.Parse(raw)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrUnsafeURL, err)
	}
	if u.Scheme != "http" && u.Scheme != "https" || u.Hostname() == "" {
		return fmt.Errorf("%w: %s", ErrUnsafeURL, raw)
	}
	if ip := net.ParseIP(u.Hostname()); ip != nil && !publicIP(ip) {
		return fmt.Errorf("%w: %s", ErrUnsafeURL, raw)
	}
	if strings.EqualFold(u.Hostname(), "localhost") || strings.HasSuffix(strings.ToLower(u.Hostname()), ".localhost") {
		return fmt.Errorf("%w: %s", ErrUnsafeURL, raw)
	}
	return nil
}

// safeDial resolves addr and connects to the first public address it
// resolves to, so a name that resolves into our network (including through
// DNS rebinding after validation) can never be reached.
func safeDial(ctx context.Context, network, addr string) (net.Conn, error) {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return nil, err
	}
	ips, err := net.DefaultResolver.LookupIPAddr(ctx, host)
	if err != nil {
		return nil, err
	}
	dialer := &net.Dialer{Timeout: 10 * time.Second}
	var last error = fmt.Errorf("%w: %s", ErrUnsafeURL, host)
	for _, ip := range ips {
		if !publicIP(ip.IP) {
			continue
		}
		conn, err := dialer.DialContext(ctx, network, net.JoinHostPort(ip.IP.String(), port))
		if err == nil {
			return conn, nil
		}
		last = err
	}
	return nil, last
}

// NewSafeClient returns an HTTP client that only reaches public addresses,
// follows at most five redirects, and stays on http(s) across them.
func NewSafeClient(timeout time.Duration) *http.Client {
	transport := &http.Transport{
		Proxy:                 nil, // never route through an environment proxy
		DialContext:           safeDial,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          64,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: time.Second,
	}
	return &http.Client{
		Timeout:   timeout,
		Transport: transport,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= maxRedirects {
				return fmt.Errorf("%w: more than %d redirects", ErrUnsafeURL, maxRedirects)
			}
			return checkURL(req.URL.String())
		},
	}
}

var (
	safeClientOnce sync.Once
	safeClient     *http.Client
)

func defaultSafeClient() *http.Client {
	safeClientOnce.Do(func() { safeClient = NewSafeClient(30 * time.Second) })
	return safeClient
}
