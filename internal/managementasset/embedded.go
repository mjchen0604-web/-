package managementasset

import _ "embed"

//go:embed management.html
var embeddedHTML []byte

//go:embed portal.html
var embeddedPortalHTML []byte

// EmbeddedHTML returns the built-in management control panel HTML.
func EmbeddedHTML() []byte {
	return embeddedHTML
}

// EmbeddedPortalHTML returns the built-in portal HTML.
func EmbeddedPortalHTML() []byte {
	return embeddedPortalHTML
}
