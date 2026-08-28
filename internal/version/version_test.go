package version

import "testing"

func TestStringIsNonEmpty(t *testing.T) {
	if String() == "" {
		t.Fatal("version string must not be empty")
	}
}
