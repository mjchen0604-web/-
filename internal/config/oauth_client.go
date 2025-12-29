package config

import (
	"os"
	"strings"
)

// ResolveGeminiOAuthClient returns the configured Gemini OAuth client credentials.
func ResolveGeminiOAuthClient(cfg *Config) (string, string) {
	var client OAuthClient
	if cfg != nil {
		client = cfg.OAuth.Gemini
	}
	return resolveOAuthClient(client, "GEMINI_OAUTH_CLIENT_ID", "GEMINI_OAUTH_CLIENT_SECRET")
}

// ResolveAntigravityOAuthClient returns the configured Antigravity OAuth client credentials.
func ResolveAntigravityOAuthClient(cfg *Config) (string, string) {
	var client OAuthClient
	if cfg != nil {
		client = cfg.OAuth.Antigravity
	}
	return resolveOAuthClient(client, "ANTIGRAVITY_OAUTH_CLIENT_ID", "ANTIGRAVITY_OAUTH_CLIENT_SECRET")
}

func resolveOAuthClient(client OAuthClient, idEnv, secretEnv string) (string, string) {
	clientID := strings.TrimSpace(client.ClientID)
	clientSecret := strings.TrimSpace(client.ClientSecret)

	if clientID == "" {
		if value := lookupEnvValue(idEnv, strings.ToLower(idEnv)); value != "" {
			clientID = value
		}
	}
	if clientSecret == "" {
		if value := lookupEnvValue(secretEnv, strings.ToLower(secretEnv)); value != "" {
			clientSecret = value
		}
	}

	return clientID, clientSecret
}

func lookupEnvValue(keys ...string) string {
	for _, key := range keys {
		if value, ok := os.LookupEnv(key); ok {
			if trimmed := strings.TrimSpace(value); trimmed != "" {
				return trimmed
			}
		}
	}
	return ""
}
