package executor

import (
	"fmt"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/config"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

func applyCodexCommandSettings(payload []byte, settings config.CodexSettings) []byte {
	out := payload
	if settings.ReasoningEffort != "" {
		if updated, err := sjson.SetBytes(out, "reasoning.effort", settings.ReasoningEffort); err == nil {
			out = updated
		}
	}
	if settings.ReasoningSummary != "" {
		if updated, err := sjson.SetBytes(out, "reasoning.summary", settings.ReasoningSummary); err == nil {
			out = updated
		}
	}
	if settings.CompatibilityMode {
		out = applyCodexCompatibilityMode(out)
	}
	if settings.EnableWebSearch {
		tools := gjson.GetBytes(out, "tools")
		if !tools.Exists() || !tools.IsArray() || len(tools.Array()) == 0 {
			if updated, err := sjson.SetBytes(out, "tools", []any{map[string]any{"type": "web_search"}}); err == nil {
				out = updated
			}
		}
	}
	return out
}

func annotateCodexOriginalRequest(original []byte, settings config.CodexSettings) []byte {
	if len(original) == 0 || !gjson.ValidBytes(original) {
		return original
	}
	payload := map[string]any{
		"reasoning_compat": settings.ReasoningCompat,
	}
	if settings.Verbose {
		payload["verbose"] = true
	}
	if settings.VerboseObfuscation {
		payload["verbose_obfuscation"] = true
	}
	if settings.CompatibilityMode {
		payload["compatibility_mode"] = true
	}
	updated, err := sjson.SetBytes(original, "__codex_settings", payload)
	if err != nil {
		return original
	}
	return updated
}

func splitCodexReasoningSuffix(model string) (base, effort string, ok bool) {
	trimmed := strings.TrimSpace(model)
	if trimmed == "" {
		return "", "", false
	}
	normalized := strings.ToLower(trimmed)
	efforts := []string{"minimal", "low", "medium", "high", "xhigh"}

	if idx := strings.LastIndex(normalized, ":"); idx > 0 {
		maybe := strings.TrimSpace(normalized[idx+1:])
		for _, effort := range efforts {
			if maybe == effort {
				return strings.TrimSpace(trimmed[:idx]), effort, true
			}
		}
	}

	for _, effort := range efforts {
		for _, sep := range []string{"-", "_"} {
			suffix := sep + effort
			if strings.HasSuffix(normalized, suffix) {
				base = strings.TrimSpace(trimmed[:len(trimmed)-len(suffix)])
				if base == "" {
					return "", "", false
				}
				return base, effort, true
			}
		}
	}
	return "", "", false
}

func applyCodexCompatibilityMode(payload []byte) []byte {
	out := payload
	if updated, err := sjson.DeleteBytes(out, "instructions"); err == nil {
		out = updated
	}
	input := gjson.GetBytes(out, "input")
	if !input.Exists() || !input.IsArray() {
		return out
	}
	items := input.Array()
	for i := 0; i < len(items); i++ {
		role := items[i].Get("role")
		if !strings.EqualFold(role.String(), "system") {
			continue
		}
		path := fmt.Sprintf("input.%d.role", i)
		if updated, err := sjson.SetBytes(out, path, "user"); err == nil {
			out = updated
		}
	}
	return out
}
