// Command ml-worker runs dataset, embedding, and index maintenance jobs.
package main

import (
	"fmt"

	"podfind/internal/version"
)

func main() {
	fmt.Printf("podfind ml-worker %s (scaffold)\n", version.String())
}
