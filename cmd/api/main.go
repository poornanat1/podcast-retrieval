// Command api serves the public PodFind HTTP API.
package main

import (
	"fmt"

	"podfind/internal/version"
)

func main() {
	fmt.Printf("podfind api %s (scaffold)\n", version.String())
}
