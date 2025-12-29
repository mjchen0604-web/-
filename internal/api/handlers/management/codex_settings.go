package management

import (
	"encoding/json"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/config"
)

func (h *Handler) GetCodexSettings(c *gin.Context) {
	if h == nil || h.cfg == nil {
		c.JSON(http.StatusOK, gin.H{"codex-settings": config.NormalizeCodexSettings(config.CodexSettings{})})
		return
	}
	settings := config.NormalizeCodexSettings(h.cfg.CodexSettings)
	c.JSON(http.StatusOK, gin.H{"codex-settings": settings})
}

func (h *Handler) PutCodexSettings(c *gin.Context) {
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid body"})
		return
	}
	settings, err := decodeCodexSettings(body)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid body"})
		return
	}
	if h.cfg == nil {
		h.cfg = &config.Config{}
	}
	h.cfg.CodexSettings = config.NormalizeCodexSettings(settings)
	h.persist(c)
}

func decodeCodexSettings(body []byte) (config.CodexSettings, error) {
	var wrapper map[string]json.RawMessage
	if err := json.Unmarshal(body, &wrapper); err != nil {
		return config.CodexSettings{}, err
	}
	if raw, ok := wrapper["codex-settings"]; ok {
		var settings config.CodexSettings
		if err := json.Unmarshal(raw, &settings); err != nil {
			return config.CodexSettings{}, err
		}
		return settings, nil
	}
	var settings config.CodexSettings
	if err := json.Unmarshal(body, &settings); err != nil {
		return config.CodexSettings{}, err
	}
	return settings, nil
}
