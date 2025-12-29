// Package openai provides response translation functionality for Codex to OpenAI API compatibility.
// This package handles the conversion of Codex API responses into OpenAI Chat Completions-compatible
// JSON format, transforming streaming events and non-streaming responses into the format
// expected by OpenAI API clients. It supports both streaming and non-streaming modes,
// handling text content, tool calls, reasoning content, and usage metadata appropriately.
package chat_completions

import (
	"bytes"
	"context"
	"strings"
	"time"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

var (
	dataTag = []byte("data:")
)

func normalizeCodexReasoningCompat(raw string) string {
	value := strings.ToLower(strings.TrimSpace(raw))
	switch value {
	case "legacy", "o3", "think-tags", "current":
		return value
	default:
		return "think-tags"
	}
}

func resolveCodexReasoningCompat(originalRequestRawJSON []byte) string {
	if v := gjson.GetBytes(originalRequestRawJSON, "__codex_settings.reasoning_compat"); v.Exists() {
		return normalizeCodexReasoningCompat(v.String())
	}
	if v := gjson.GetBytes(originalRequestRawJSON, "codex_settings.reasoning_compat"); v.Exists() {
		return normalizeCodexReasoningCompat(v.String())
	}
	return normalizeCodexReasoningCompat("")
}

func joinReasoningText(summary, full string) string {
	summary = strings.TrimSpace(summary)
	full = strings.TrimSpace(full)
	if summary == "" {
		return full
	}
	if full == "" {
		return summary
	}
	return summary + "\n\n" + full
}

func applyCodexReasoningCompatToMessage(message map[string]any, summary, full, compat string) map[string]any {
	compat = normalizeCodexReasoningCompat(compat)
	switch compat {
	case "o3":
		if combined := joinReasoningText(summary, full); combined != "" {
			message["reasoning"] = map[string]any{"content": []map[string]any{{"type": "text", "text": combined}}}
		}
		return message
	case "legacy", "current":
		if strings.TrimSpace(summary) != "" {
			message["reasoning_summary"] = summary
		}
		if strings.TrimSpace(full) != "" {
			message["reasoning"] = full
		}
		return message
	default:
		combined := joinReasoningText(summary, full)
		if combined == "" {
			return message
		}
		thinkBlock := "<think>" + combined + "</think>"
		if content, ok := message["content"].(string); ok {
			message["content"] = thinkBlock + content
			return message
		}
		if message["content"] == nil {
			message["content"] = thinkBlock
		}
		return message
	}
}

// ConvertCliToOpenAIParams holds parameters for response conversion.
type ConvertCliToOpenAIParams struct {
	ResponseID        string
	CreatedAt         int64
	Model             string
	FunctionCallIndex int
	ReasoningCompat   string
	ThinkOpen         bool
	ThinkClosed       bool
	SawAnySummary     bool
	PendingSummary    bool
}

// ConvertCodexResponseToOpenAI translates a single chunk of a streaming response from the
// Codex API format to the OpenAI Chat Completions streaming format.
// It processes various Codex event types and transforms them into OpenAI-compatible JSON responses.
// The function handles text content, tool calls, reasoning content, and usage metadata, outputting
// responses that match the OpenAI API format. It supports incremental updates for streaming responses.
//
// Parameters:
//   - ctx: The context for the request, used for cancellation and timeout handling
//   - modelName: The name of the model being used for the response
//   - rawJSON: The raw JSON response from the Codex API
//   - param: A pointer to a parameter object for maintaining state between calls
//
// Returns:
//   - []string: A slice of strings, each containing an OpenAI-compatible JSON response
func ConvertCodexResponseToOpenAI(_ context.Context, modelName string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, param *any) []string {
	if *param == nil {
		*param = &ConvertCliToOpenAIParams{
			Model:             modelName,
			CreatedAt:         0,
			ResponseID:        "",
			FunctionCallIndex: -1,
		}
	}

	if !bytes.HasPrefix(rawJSON, dataTag) {
		return []string{}
	}
	rawJSON = bytes.TrimSpace(rawJSON[5:])

	// Initialize the OpenAI SSE template.
	template := `{"id":"","object":"chat.completion.chunk","created":12345,"model":"model","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":null,"tool_calls":null},"finish_reason":null,"native_finish_reason":null}]}`

	rootResult := gjson.ParseBytes(rawJSON)

	typeResult := rootResult.Get("type")
	dataType := typeResult.String()
	if dataType == "response.created" {
		(*param).(*ConvertCliToOpenAIParams).ResponseID = rootResult.Get("response.id").String()
		(*param).(*ConvertCliToOpenAIParams).CreatedAt = rootResult.Get("response.created_at").Int()
		(*param).(*ConvertCliToOpenAIParams).Model = rootResult.Get("response.model").String()
		return []string{}
	}
	params := (*param).(*ConvertCliToOpenAIParams)
	if params.ReasoningCompat == "" {
		params.ReasoningCompat = resolveCodexReasoningCompat(originalRequestRawJSON)
	}
	compat := params.ReasoningCompat

	// Extract and set the model version.
	if modelResult := gjson.GetBytes(rawJSON, "model"); modelResult.Exists() {
		template, _ = sjson.Set(template, "model", modelResult.String())
	}

	template, _ = sjson.Set(template, "created", (*param).(*ConvertCliToOpenAIParams).CreatedAt)

	// Extract and set the response ID.
	template, _ = sjson.Set(template, "id", (*param).(*ConvertCliToOpenAIParams).ResponseID)

	// Extract and set usage metadata (token counts).
	if usageResult := gjson.GetBytes(rawJSON, "response.usage"); usageResult.Exists() {
		if outputTokensResult := usageResult.Get("output_tokens"); outputTokensResult.Exists() {
			template, _ = sjson.Set(template, "usage.completion_tokens", outputTokensResult.Int())
		}
		if totalTokensResult := usageResult.Get("total_tokens"); totalTokensResult.Exists() {
			template, _ = sjson.Set(template, "usage.total_tokens", totalTokensResult.Int())
		}
		if inputTokensResult := usageResult.Get("input_tokens"); inputTokensResult.Exists() {
			template, _ = sjson.Set(template, "usage.prompt_tokens", inputTokensResult.Int())
		}
		if reasoningTokensResult := usageResult.Get("output_tokens_details.reasoning_tokens"); reasoningTokensResult.Exists() {
			template, _ = sjson.Set(template, "usage.completion_tokens_details.reasoning_tokens", reasoningTokensResult.Int())
		}
	}

	makeContentChunk := func(text string) string {
		chunk := template
		chunk, _ = sjson.Set(chunk, "choices.0.delta.role", "assistant")
		chunk, _ = sjson.Set(chunk, "choices.0.delta.content", text)
		return chunk
	}
	makeReasoningSummaryChunk := func(text string) string {
		chunk := template
		chunk, _ = sjson.Set(chunk, "choices.0.delta.role", "assistant")
		chunk, _ = sjson.Set(chunk, "choices.0.delta.reasoning_summary", text)
		chunk, _ = sjson.Set(chunk, "choices.0.delta.reasoning", text)
		return chunk
	}
	makeReasoningChunk := func(text string) string {
		chunk := template
		chunk, _ = sjson.Set(chunk, "choices.0.delta.role", "assistant")
		chunk, _ = sjson.Set(chunk, "choices.0.delta.reasoning", text)
		return chunk
	}
	makeO3ReasoningChunk := func(text string) string {
		chunk := template
		chunk, _ = sjson.Set(chunk, "choices.0.delta.role", "assistant")
		chunk, _ = sjson.Set(chunk, "choices.0.delta.reasoning.content.0.type", "text")
		chunk, _ = sjson.Set(chunk, "choices.0.delta.reasoning.content.0.text", text)
		return chunk
	}

	switch dataType {
	case "response.reasoning_summary_part.added":
		if compat == "think-tags" || compat == "o3" {
			if params.SawAnySummary {
				params.PendingSummary = true
			} else {
				params.SawAnySummary = true
			}
		}
		return []string{}
	case "response.reasoning_summary_text.delta", "response.reasoning_text.delta":
		delta := rootResult.Get("delta").String()
		switch compat {
		case "o3":
			chunks := make([]string, 0, 2)
			if dataType == "response.reasoning_summary_text.delta" && params.PendingSummary {
				chunks = append(chunks, makeO3ReasoningChunk("\n"))
				params.PendingSummary = false
			}
			chunks = append(chunks, makeO3ReasoningChunk(delta))
			return chunks
		case "legacy", "current":
			if dataType == "response.reasoning_summary_text.delta" {
				return []string{makeReasoningSummaryChunk(delta)}
			}
			return []string{makeReasoningChunk(delta)}
		default:
			chunks := make([]string, 0, 3)
			if !params.ThinkOpen && !params.ThinkClosed {
				chunks = append(chunks, makeContentChunk("<think>"))
				params.ThinkOpen = true
			}
			if dataType == "response.reasoning_summary_text.delta" && params.PendingSummary {
				chunks = append(chunks, makeContentChunk("\n"))
				params.PendingSummary = false
			}
			chunks = append(chunks, makeContentChunk(delta))
			return chunks
		}
	case "response.reasoning_summary_text.done":
		return []string{}
	case "response.output_text.delta":
		if deltaResult := rootResult.Get("delta"); deltaResult.Exists() {
			return []string{makeContentChunk(deltaResult.String())}
		}
		return []string{}
	case "response.completed":
		finishReason := "stop"
		if params.FunctionCallIndex != -1 {
			finishReason = "tool_calls"
		}
		template, _ = sjson.Set(template, "choices.0.finish_reason", finishReason)
		template, _ = sjson.Set(template, "choices.0.native_finish_reason", finishReason)
		if compat == "think-tags" && params.ThinkOpen && !params.ThinkClosed {
			params.ThinkOpen = false
			params.ThinkClosed = true
			return []string{makeContentChunk("</think>"), template}
		}
		return []string{template}
	case "response.output_item.done":
		functionCallItemTemplate := `{"index":0,"id":"","type":"function","function":{"name":"","arguments":""}}`
		itemResult := rootResult.Get("item")
		if itemResult.Exists() {
			if itemResult.Get("type").String() != "function_call" {
				return []string{}
			}

			// set the index
			(*param).(*ConvertCliToOpenAIParams).FunctionCallIndex++
			functionCallItemTemplate, _ = sjson.Set(functionCallItemTemplate, "index", (*param).(*ConvertCliToOpenAIParams).FunctionCallIndex)

			template, _ = sjson.SetRaw(template, "choices.0.delta.tool_calls", `[]`)
			functionCallItemTemplate, _ = sjson.Set(functionCallItemTemplate, "id", itemResult.Get("call_id").String())

			// Restore original tool name if it was shortened
			name := itemResult.Get("name").String()
			// Build reverse map on demand from original request tools
			rev := buildReverseMapFromOriginalOpenAI(originalRequestRawJSON)
			if orig, ok := rev[name]; ok {
				name = orig
			}
			functionCallItemTemplate, _ = sjson.Set(functionCallItemTemplate, "function.name", name)

			functionCallItemTemplate, _ = sjson.Set(functionCallItemTemplate, "function.arguments", itemResult.Get("arguments").String())
			template, _ = sjson.Set(template, "choices.0.delta.role", "assistant")
			template, _ = sjson.SetRaw(template, "choices.0.delta.tool_calls.-1", functionCallItemTemplate)
		}

	default:
		return []string{}
	}

	return []string{template}
}

// ConvertCodexResponseToOpenAINonStream converts a non-streaming Codex response to a non-streaming OpenAI response.
// This function processes the complete Codex response and transforms it into a single OpenAI-compatible
// JSON response. It handles message content, tool calls, reasoning content, and usage metadata, combining all
// the information into a single response that matches the OpenAI API format.
//
// Parameters:
//   - ctx: The context for the request, used for cancellation and timeout handling
//   - modelName: The name of the model being used for the response (unused in current implementation)
//   - rawJSON: The raw JSON response from the Codex API
//   - param: A pointer to a parameter object for the conversion (unused in current implementation)
//
// Returns:
//   - string: An OpenAI-compatible JSON response containing all message content and metadata
func ConvertCodexResponseToOpenAINonStream(_ context.Context, _ string, originalRequestRawJSON, requestRawJSON, rawJSON []byte, _ *any) string {
	rootResult := gjson.ParseBytes(rawJSON)
	// Verify this is a response.completed event
	if rootResult.Get("type").String() != "response.completed" {
		return ""
	}
	compat := resolveCodexReasoningCompat(originalRequestRawJSON)

	unixTimestamp := time.Now().Unix()

	responseResult := rootResult.Get("response")

	template := `{"id":"","object":"chat.completion","created":123456,"model":"model","choices":[{"index":0,"message":{"role":"assistant","content":null,"reasoning_content":null,"tool_calls":null},"finish_reason":null,"native_finish_reason":null}]}`

	// Extract and set the model version.
	if modelResult := responseResult.Get("model"); modelResult.Exists() {
		template, _ = sjson.Set(template, "model", modelResult.String())
	}

	// Extract and set the creation timestamp.
	if createdAtResult := responseResult.Get("created_at"); createdAtResult.Exists() {
		template, _ = sjson.Set(template, "created", createdAtResult.Int())
	} else {
		template, _ = sjson.Set(template, "created", unixTimestamp)
	}

	// Extract and set the response ID.
	if idResult := responseResult.Get("id"); idResult.Exists() {
		template, _ = sjson.Set(template, "id", idResult.String())
	}

	// Extract and set usage metadata (token counts).
	if usageResult := responseResult.Get("usage"); usageResult.Exists() {
		if outputTokensResult := usageResult.Get("output_tokens"); outputTokensResult.Exists() {
			template, _ = sjson.Set(template, "usage.completion_tokens", outputTokensResult.Int())
		}
		if totalTokensResult := usageResult.Get("total_tokens"); totalTokensResult.Exists() {
			template, _ = sjson.Set(template, "usage.total_tokens", totalTokensResult.Int())
		}
		if inputTokensResult := usageResult.Get("input_tokens"); inputTokensResult.Exists() {
			template, _ = sjson.Set(template, "usage.prompt_tokens", inputTokensResult.Int())
		}
		if reasoningTokensResult := usageResult.Get("output_tokens_details.reasoning_tokens"); reasoningTokensResult.Exists() {
			template, _ = sjson.Set(template, "usage.completion_tokens_details.reasoning_tokens", reasoningTokensResult.Int())
		}
	}

	// Process the output array for content and function calls
	outputResult := responseResult.Get("output")
	if outputResult.IsArray() {
		outputArray := outputResult.Array()
		var contentText string
		var reasoningSummaryText string
		var reasoningFullText string
		var toolCalls []string

		for _, outputItem := range outputArray {
			outputType := outputItem.Get("type").String()

			switch outputType {
			case "reasoning":
				// Extract reasoning content from summary
				if summaryResult := outputItem.Get("summary"); summaryResult.IsArray() {
					summaryArray := summaryResult.Array()
					for _, summaryItem := range summaryArray {
						if summaryItem.Get("type").String() == "summary_text" {
							if text := strings.TrimSpace(summaryItem.Get("text").String()); text != "" {
								if reasoningSummaryText != "" {
									reasoningSummaryText += "\n\n"
								}
								reasoningSummaryText += text
							}
						}
					}
				}
				if contentResult := outputItem.Get("content"); contentResult.IsArray() {
					contentArray := contentResult.Array()
					for _, contentItem := range contentArray {
						if text := strings.TrimSpace(contentItem.Get("text").String()); text != "" {
							if reasoningFullText != "" {
								reasoningFullText += "\n\n"
							}
							reasoningFullText += text
						}
					}
				}
				if text := strings.TrimSpace(outputItem.Get("text").String()); text != "" {
					if reasoningFullText != "" {
						reasoningFullText += "\n\n"
					}
					reasoningFullText += text
				}
			case "message":
				// Extract message content
				if contentResult := outputItem.Get("content"); contentResult.IsArray() {
					contentArray := contentResult.Array()
					for _, contentItem := range contentArray {
						if contentItem.Get("type").String() == "output_text" {
							contentText = contentItem.Get("text").String()
							break
						}
					}
				}
			case "function_call":
				// Handle function call content
				functionCallTemplate := `{"id": "","type": "function","function": {"name": "","arguments": ""}}`

				if callIdResult := outputItem.Get("call_id"); callIdResult.Exists() {
					functionCallTemplate, _ = sjson.Set(functionCallTemplate, "id", callIdResult.String())
				}

				if nameResult := outputItem.Get("name"); nameResult.Exists() {
					n := nameResult.String()
					rev := buildReverseMapFromOriginalOpenAI(originalRequestRawJSON)
					if orig, ok := rev[n]; ok {
						n = orig
					}
					functionCallTemplate, _ = sjson.Set(functionCallTemplate, "function.name", n)
				}

				if argsResult := outputItem.Get("arguments"); argsResult.Exists() {
					functionCallTemplate, _ = sjson.Set(functionCallTemplate, "function.arguments", argsResult.String())
				}

				toolCalls = append(toolCalls, functionCallTemplate)
			}
		}

		message := map[string]any{"role": "assistant"}
		if contentText != "" {
			message["content"] = contentText
		}
		if len(toolCalls) > 0 {
			calls := make([]any, 0, len(toolCalls))
			for _, toolCall := range toolCalls {
				if toolCall == "" {
					continue
				}
				if parsed := gjson.Parse(toolCall); parsed.Exists() && parsed.IsObject() {
					calls = append(calls, parsed.Value())
				}
			}
			if len(calls) > 0 {
				message["tool_calls"] = calls
			}
		}
		message = applyCodexReasoningCompatToMessage(message, reasoningSummaryText, reasoningFullText, compat)
		if updated, err := sjson.Set(template, "choices.0.message", message); err == nil {
			template = updated
		}
	}

	// Extract and set the finish reason based on status
	if statusResult := responseResult.Get("status"); statusResult.Exists() {
		status := statusResult.String()
		if status == "completed" {
			template, _ = sjson.Set(template, "choices.0.finish_reason", "stop")
			template, _ = sjson.Set(template, "choices.0.native_finish_reason", "stop")
		}
	}

	return template
}

// buildReverseMapFromOriginalOpenAI builds a map of shortened tool name -> original tool name
// from the original OpenAI-style request JSON using the same shortening logic.
func buildReverseMapFromOriginalOpenAI(original []byte) map[string]string {
	tools := gjson.GetBytes(original, "tools")
	rev := map[string]string{}
	if tools.IsArray() && len(tools.Array()) > 0 {
		var names []string
		arr := tools.Array()
		for i := 0; i < len(arr); i++ {
			t := arr[i]
			if t.Get("type").String() != "function" {
				continue
			}
			fn := t.Get("function")
			if !fn.Exists() {
				continue
			}
			if v := fn.Get("name"); v.Exists() {
				names = append(names, v.String())
			}
		}
		if len(names) > 0 {
			m := buildShortNameMap(names)
			for orig, short := range m {
				rev[short] = orig
			}
		}
	}
	return rev
}
