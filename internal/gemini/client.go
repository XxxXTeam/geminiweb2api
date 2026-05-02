package gemini

import (
	"bufio"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"main/internal/config"
	"main/internal/httpclient"
	"main/internal/logging"
	metricspkg "main/internal/metrics"
	"main/internal/support"
	"main/internal/token"
)

var (
	depGetConfig     func() config.Config
	depGetHTTPClient func() *http.Client
	depGetLogger     func() *logging.Logger
	depMetrics       *metricspkg.Metrics
	depTokens        *token.Manager
)

func Initialize(
	getConfig func() config.Config,
	getHTTPClient func() *http.Client,
	getLogger func() *logging.Logger,
	metrics *metricspkg.Metrics,
	tokens *token.Manager,
) {
	depGetConfig = getConfig
	depGetHTTPClient = getHTTPClient
	depGetLogger = getLogger
	depMetrics = metrics
	depTokens = tokens
}

type GeminiSession struct {
	mu             sync.RWMutex
	ConversationID string
	ResponseID     string
	ChoiceID       string
	TokenIndex     int
}

type GeminiSessionSnapshot struct {
	ConversationID string
	ResponseID     string
	ChoiceID       string
	TokenIndex     int
}

func (s *GeminiSession) Snapshot() GeminiSessionSnapshot {
	if s == nil {
		return GeminiSessionSnapshot{}
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	return GeminiSessionSnapshot{
		ConversationID: s.ConversationID,
		ResponseID:     s.ResponseID,
		ChoiceID:       s.ChoiceID,
		TokenIndex:     s.TokenIndex,
	}
}

func (s *GeminiSession) SetConversationID(conversationID string) {
	if s == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.ConversationID = conversationID
}

type ChatCompletionRequest struct {
	Model               string         `json:"model"`
	Messages            []Message      `json:"messages"`
	Stream              bool           `json:"stream"`
	Tools               []Tool         `json:"tools,omitempty"`
	ToolChoice          any            `json:"tool_choice,omitempty"`
	Temperature         float64        `json:"temperature,omitempty"`
	MaxTokens           int            `json:"max_tokens,omitempty"`
	MaxCompletionTokens int            `json:"max_completion_tokens,omitempty"`
	ConversationID      string         `json:"conversation_id,omitempty"`
	N                   int            `json:"n,omitempty"`
	Stop                interface{}    `json:"stop,omitempty"`
	TopP                float64        `json:"top_p,omitempty"`
	PresencePenalty     float64        `json:"presence_penalty,omitempty"`
	FrequencyPenalty    float64        `json:"frequency_penalty,omitempty"`
	ResponseFormat      map[string]any `json:"response_format,omitempty"`
	User                string         `json:"user,omitempty"`
	StreamOptions       *StreamOptions `json:"stream_options,omitempty"`
}

type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

type Function struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}

type Message struct {
	Role             string      `json:"role"`
	Content          interface{} `json:"content"`
	ReasoningContent string      `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall  `json:"tool_calls,omitempty"`
	ToolCallID       string      `json:"tool_call_id,omitempty"`
}

type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type ParsedMessage struct {
	Text   string
	Images []ImageData
}

type ImageData struct {
	MimeType string
	Base64   string
	URL      string
}

type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type ChatCompletionResponse struct {
	ID             string   `json:"id"`
	Object         string   `json:"object"`
	Created        int64    `json:"created"`
	Model          string   `json:"model"`
	Choices        []Choice `json:"choices"`
	Usage          Usage    `json:"usage,omitempty"`
	ConversationID string   `json:"conversation_id,omitempty"`
}

type Choice struct {
	Index        int      `json:"index"`
	Message      *Message `json:"message,omitempty"`
	Delta        *Delta   `json:"delta,omitempty"`
	FinishReason *string  `json:"finish_reason,omitempty"`
}

type Delta struct {
	Role             string     `json:"role,omitempty"`
	Content          string     `json:"content,omitempty"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type ModelsResponse struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type ResponsesRequest struct {
	Model  string      `json:"model"`
	Input  interface{} `json:"input"`
	Stream bool        `json:"stream,omitempty"`
}

type ResponsesResponse struct {
	ID        string `json:"id"`
	Object    string `json:"object"`
	CreatedAt int64  `json:"created_at"`
	Model     string `json:"model"`
	Output    []struct {
		Type    string `json:"type"`
		Role    string `json:"role"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	} `json:"output"`
}

type ErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code,omitempty"`
	} `json:"error"`
}

type OpenAIError struct {
	Status  int
	Type    string
	Code    string
	Message string
}

type AccountContext struct {
	ID      string
	Email   string
	Cookies string
	Proxy   string
	Token   string
	BLToken string
	FSID    string
	ReqID   string
}

var accountHTTPClients sync.Map

func httpClientForAccount(accountCtx AccountContext) *http.Client {
	proxyValue := strings.TrimSpace(accountCtx.Proxy)
	if proxyValue == "" {
		return depGetHTTPClient()
	}
	if client, ok := accountHTTPClients.Load(proxyValue); ok {
		return client.(*http.Client)
	}
	client, _, _ := httpclient.NewWithProxy(depGetConfig(), proxyValue, depGetLogger())
	actual, _ := accountHTTPClients.LoadOrStore(proxyValue, client)
	return actual.(*http.Client)
}

var errorCodeMap = map[int]string{
	0:    "success",
	1:    "invalid_request",
	2:    "rate_limit_exceeded",
	3:    "content_filtered",
	4:    "authentication_error",
	5:    "server_error",
	6:    "timeout",
	7:    "model_overloaded",
	8:    "context_length_exceeded",
	1013: "temporary_stream_error",
	1037: "usage_limit_exceeded",
	1050: "model_inconsistent",
	1052: "model_header_invalid",
	1060: "ip_temporarily_blocked",
	1016: "unauthenticated",
}

const (
	geminiInnerReqLen          = 69
	geminiInnerReqLenThinking  = 80
	geminiStreamingFlagIndex   = 7
	geminiDefaultMetadataSlots = 10
	geminiWebLanguage          = "zh-CN"
	headerModelJSPB            = "x-goog-ext-525001261-jspb"
	headerRequestUUIDJSPB      = "x-goog-ext-525005358-jspb"

	idxFeatureMode   = 49
	idxThinkingLevel = 79

	thinkingLevelStandard  = 1
	thinkingLevelExtended  = 2
	thinkingLevelDeepThink = 3

	featureModeDeepThink = 20
	featureModeVideo     = 11
	featureModeImage     = 14
)

type webModelSpec struct {
	HexID    string
	Capacity int
}

type experimentalRequestConfig struct {
	FeatureMode   int
	ThinkingLevel int
	Ef            int
	Xpc           string
	Lo            *bool
}

var modelSpecMap = map[string]webModelSpec{
	"gemini-3-flash":               {"fbb127bbb056c959", 1},
	"gemini-3":                     {"fbb127bbb056c959", 1},
	"gemini-flash":                 {"fbb127bbb056c959", 1},
	"gemini-3-flash-thinking":      {"5bf011840784117a", 1},
	"gemini-3-flash-plus":          {"56fdd199312815e2", 4},
	"gemini-3-flash-thinking-plus": {"e051ce1aa80aa576", 4},
	"gemini-3-flash-advanced":      {"56fdd199312815e2", 2},
	"gemini-3-pro":                 {"9d8ca3786ebdfbea", 1},
	"gemini-pro":                   {"9d8ca3786ebdfbea", 1},
	"gemini-2.5-pro":               {"9d8ca3786ebdfbea", 1},
	"gemini-3-pro-deep-think":      {"e6fa609c3fa255c0", 4},
	"gemini-3-pro-image":           {"e6fa609c3fa255c0", 4},
	"gemini-3-pro-video":           {"e6fa609c3fa255c0", 4},
	"gemini-3-pro-plus":            {"e6fa609c3fa255c0", 4},
	"gemini-3-pro-advanced":        {"e6fa609c3fa255c0", 2},
	"gemini-3.1":                   {"e6fa609c3fa255c0", 2},
	"gemini-3.1-pro":               {"e6fa609c3fa255c0", 2},
	"gemini-2.5-flash":             {"e6fa609c3fa255c0", 1},
	"gemini-2.5-flash-preview":     {"e6fa609c3fa255c0", 1},
	"gemini-2-flash":               {"203e6bb81620bcfe", 1},
	"gemini-2.0-flash":             {"203e6bb81620bcfe", 1},
}

func defaultGeminiMetadata() []interface{} {
	m := make([]interface{}, geminiDefaultMetadataSlots)
	m[0] = ""
	m[1] = ""
	m[2] = ""
	m[9] = ""
	return m
}

func sessionToGeminiMetadata(snapshot GeminiSessionSnapshot) []interface{} {
	m := defaultGeminiMetadata()
	if snapshot.ConversationID != "" {
		m[0] = snapshot.ConversationID
	}
	if snapshot.ResponseID != "" {
		m[1] = snapshot.ResponseID
	}
	if snapshot.ChoiceID != "" {
		m[2] = snapshot.ChoiceID
	}
	return m
}

func buildModelHeaderJSPB(spec webModelSpec, thinkingLevel int, uuidVal string) string {
	if thinkingLevel > 0 {
		return fmt.Sprintf(`[1,null,null,null,"%s",null,null,0,[4],null,null,%d,null,null,%d,null,"%s"]`,
			spec.HexID, thinkingLevel, thinkingLevel, uuidVal)
	}
	return fmt.Sprintf(`[1,null,null,null,"%s",null,null,0,[4],null,null,%d]`, spec.HexID, spec.Capacity)
}

func noteGeminiResponseErrors(body string, sessionKey string, mode string) {
	code, msg := parseGeminiErrorCode(body)
	if code != 0 {
		depGetLogger().Warn("%s响应含 Gemini 错误码 %d: %s", mode, code, msg)
		if code == 1052 || code == 1016 {
			depTokens.MarkSessionTokenBad(sessionKey)
		}
	}
	if hasErr, errStr := checkGeminiError(body); hasErr && code == 0 {
		depGetLogger().Warn("%s响应: %s", mode, errStr)
	}
}

func extractMessageContent(msg Message) string {
	return extractMultimodalContent(msg).Text
}

func extractMultimodalContent(msg Message) ParsedMessage {
	switch v := msg.Content.(type) {
	case string:
		return ParsedMessage{Text: v}
	case []interface{}:
		var parsed ParsedMessage
		var textParts []string
		for _, part := range v {
			p, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			if text, ok := p["text"].(string); ok {
				textParts = append(textParts, text)
			}
			imageURL, ok := extractImageURLPart(p)
			if !ok || imageURL == "" {
				continue
			}
			image := ImageData{URL: imageURL}
			if mimeType, data, ok := parseDataURI(imageURL); ok {
				image.MimeType = mimeType
				image.Base64 = data
			}
			parsed.Images = append(parsed.Images, image)
		}
		parsed.Text = strings.Join(textParts, "\n")
		return parsed
	default:
		if v != nil {
			return ParsedMessage{Text: fmt.Sprintf("%v", v)}
		}
		return ParsedMessage{}
	}
}

func extractImageURLPart(part map[string]interface{}) (string, bool) {
	imageURL, ok := part["image_url"]
	if !ok {
		return "", false
	}
	switch v := imageURL.(type) {
	case string:
		return v, true
	case map[string]interface{}:
		urlValue, ok := v["url"].(string)
		return urlValue, ok
	default:
		return "", false
	}
}

func parseDataURI(uri string) (mimeType string, data string, ok bool) {
	const marker = ";base64,"
	if !strings.HasPrefix(uri, "data:") {
		return "", "", false
	}
	idx := strings.Index(uri, marker)
	if idx < 0 {
		return "", "", false
	}
	mimeType = uri[len("data:"):idx]
	data = uri[idx+len(marker):]
	return mimeType, data, mimeType != "" && data != ""
}

func downloadImageAsBase64(imageURL string, httpClient *http.Client) (ImageData, error) {
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	resp, err := httpClient.Get(imageURL)
	if err != nil {
		return ImageData{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return ImageData{}, fmt.Errorf("download image returned status %d", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return ImageData{}, err
	}
	mimeType := resp.Header.Get("Content-Type")
	if idx := strings.Index(mimeType, ";"); idx >= 0 {
		mimeType = strings.TrimSpace(mimeType[:idx])
	}
	if mimeType == "" {
		mimeType = http.DetectContentType(body)
	}
	return ImageData{
		MimeType: mimeType,
		Base64:   base64.StdEncoding.EncodeToString(body),
		URL:      imageURL,
	}, nil
}

func buildToolsPrompt(tools []Tool) string {
	if len(tools) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("\n[TOOLS]\nYou have access to the following tools. To use a tool, respond with ONLY a JSON object in this exact format (no markdown, no code blocks):\n")
	sb.WriteString("{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n\n")
	sb.WriteString("Available tools:\n")

	for _, tool := range tools {
		if tool.Type != "function" {
			continue
		}
		sb.WriteString(fmt.Sprintf("- %s", tool.Function.Name))
		if tool.Function.Description != "" {
			sb.WriteString(fmt.Sprintf(": %s", tool.Function.Description))
		}
		if tool.Function.Parameters != nil {
			if props, ok := tool.Function.Parameters["properties"].(map[string]interface{}); ok {
				var params []string
				for k := range props {
					params = append(params, k)
				}
				sb.WriteString(fmt.Sprintf(" (params: %s)", strings.Join(params, ", ")))
			}
		}
		sb.WriteString("\n")
	}
	sb.WriteString("[/TOOLS]\n")
	return sb.String()
}

func BuildPrompt(req ChatCompletionRequest) string {
	prompt, _ := BuildPromptWithMedia(req)
	return prompt
}

func BuildPromptWithMedia(req ChatCompletionRequest) (string, []ImageData) {
	var prompt strings.Builder
	var images []ImageData
	toolsPrompt := buildToolsPrompt(req.Tools)
	if toolsPrompt != "" {
		prompt.WriteString(toolsPrompt)
		prompt.WriteString("\n---\n\n")
	}
	for _, msg := range req.Messages {
		parsed := extractMultimodalContent(msg)
		if len(parsed.Images) > 0 {
			images = append(images, parsed.Images...)
		}
		content := parsed.Text
		switch msg.Role {
		case "system":
			prompt.WriteString(fmt.Sprintf("[System Instruction]\n%s\n[/System Instruction]\n\n", content))
		case "user":
			prompt.WriteString(fmt.Sprintf("User: %s\n\n", content))
		case "assistant":
			if len(msg.ToolCalls) > 0 {
				for _, tc := range msg.ToolCalls {
					prompt.WriteString(fmt.Sprintf("Assistant (tool_call): %s(%s)\n\n", tc.Function.Name, tc.Function.Arguments))
				}
			} else {
				prompt.WriteString(fmt.Sprintf("Assistant: %s\n\n", content))
			}
		case "tool":
			prompt.WriteString(fmt.Sprintf("Tool Result [%s]: %s\n\n", msg.ToolCallID, content))
		}
	}
	return prompt.String(), images
}

func isDeepThinkAlias(modelName string) bool {
	n := strings.ToLower(strings.TrimSpace(modelName))
	return n == "gemini-3-pro-deep-think"
}

// isThinkingModel 判断模型是否需要设置 thinking 协议字段
func isThinkingModel(modelName string) bool {
	n := strings.ToLower(strings.TrimSpace(modelName))
	switch n {
	case "gemini-3-flash-thinking",
		"gemini-3-flash-thinking-plus":
		return true
	default:
		return false
	}
}

// getThinkingLevel 返回模型对应的 thinking level 和 feature mode
// 返回 (thinkingLevel, featureMode, needsThinkingFields)
func getThinkingLevel(modelName string) (int, int, bool) {
	n := strings.ToLower(strings.TrimSpace(modelName))
	switch n {
	case "gemini-3-pro-deep-think":
		return thinkingLevelDeepThink, featureModeDeepThink, true
	case "gemini-3-flash-thinking":
		return thinkingLevelStandard, 0, true
	case "gemini-3-flash-thinking-plus":
		return thinkingLevelExtended, 0, true
	default:
		return 0, 0, false
	}
}

func getExperimentalFeatureMode(modelName string) (int, bool) {
	switch strings.ToLower(strings.TrimSpace(modelName)) {
	case "gemini-3-pro-image":
		return featureModeImage, true
	case "gemini-3-pro-video":
		return featureModeVideo, true
	default:
		return 0, false
	}
}

func getExperimentalRequestConfig(modelName string) (experimentalRequestConfig, bool) {
	switch strings.ToLower(strings.TrimSpace(modelName)) {
	case "gemini-3-pro-image":
		lo := false
		return experimentalRequestConfig{
			FeatureMode:   featureModeImage,
			ThinkingLevel: 5,
			Ef:            featureModeImage,
			Xpc:           "MODE_CATEGORY_FAST",
			Lo:            &lo,
		}, true
	case "gemini-3-pro-video":
		lo := false
		return experimentalRequestConfig{
			FeatureMode:   featureModeVideo,
			ThinkingLevel: 5,
			Ef:            featureModeVideo,
			Xpc:           "MODE_CATEGORY_FAST",
			Lo:            &lo,
		}, true
	default:
		return experimentalRequestConfig{}, false
	}
}

func parseToolCalls(content string, tools []Tool) (string, []ToolCall) {
	if len(tools) == 0 {
		return content, nil
	}

	allowed := make(map[string]struct{}, len(tools))
	for _, t := range tools {
		allowed[t.Function.Name] = struct{}{}
	}

	clean := content
	toolCalls := make([]ToolCall, 0)
	seen := make(map[string]struct{})

	addCall := func(name, args, rawBlock string) {
		if _, ok := allowed[name]; !ok {
			return
		}
		key := name + "\n" + args
		if _, ok := seen[key]; ok {
			return
		}
		seen[key] = struct{}{}
		toolCalls = append(toolCalls, ToolCall{
			ID:   fmt.Sprintf("call_%s_%d", support.GenerateRandomHex(8), len(toolCalls)),
			Type: "function",
			Function: FunctionCall{
				Name:      name,
				Arguments: args,
			},
		})
		if rawBlock != "" {
			clean = strings.Replace(clean, rawBlock, "", 1)
		}
	}

	// 1) 优先解析 markdown fenced tool_call 块
	fenceRe := regexp.MustCompile("(?is)```tool_call\\s*(\\{[\\s\\S]*?\\})\\s*```")
	for _, m := range fenceRe.FindAllStringSubmatch(content, -1) {
		if len(m) < 2 {
			continue
		}
		name, args, ok := parseOneToolCallJSON(strings.TrimSpace(m[1]))
		if ok {
			addCall(name, args, m[0])
		}
	}

	// 2) 再扫描正文里可能出现的 JSON 对象
	for _, raw := range extractJSONObjectCandidates(content) {
		name, args, ok := parseOneToolCallJSON(raw)
		if ok {
			addCall(name, args, raw)
		}
	}

	return strings.TrimSpace(clean), toolCalls
}

func parseOneToolCallJSON(raw string) (name string, args string, ok bool) {
	var tc struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal([]byte(raw), &tc); err != nil {
		return "", "", false
	}
	if strings.TrimSpace(tc.Name) == "" {
		return "", "", false
	}
	argsNorm, ok := normalizeToolArguments(tc.Arguments)
	if !ok {
		return "", "", false
	}
	return tc.Name, argsNorm, true
}

func normalizeToolArguments(raw json.RawMessage) (string, bool) {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return "{}", true
	}

	// arguments 可能被模型输出为 JSON 字符串，需要二次反序列化
	if strings.HasPrefix(trimmed, `"`) {
		var inner string
		if err := json.Unmarshal([]byte(trimmed), &inner); err != nil {
			return "", false
		}
		trimmed = strings.TrimSpace(inner)
	}

	if trimmed == "" {
		return "{}", true
	}

	var obj interface{}
	if err := json.Unmarshal([]byte(trimmed), &obj); err != nil {
		return "", false
	}
	canon, err := json.Marshal(obj)
	if err != nil {
		return "", false
	}
	return string(canon), true
}

func extractJSONObjectCandidates(s string) []string {
	result := make([]string, 0)
	b := []byte(s)
	for i := 0; i < len(b); i++ {
		if b[i] != '{' {
			continue
		}
		depth := 0
		inString := false
		escaped := false
		for j := i; j < len(b); j++ {
			ch := b[j]
			if inString {
				if escaped {
					escaped = false
					continue
				}
				if ch == '\\' {
					escaped = true
					continue
				}
				if ch == '"' {
					inString = false
				}
				continue
			}

			if ch == '"' {
				inString = true
				continue
			}
			if ch == '{' {
				depth++
			}
			if ch == '}' {
				depth--
				if depth == 0 {
					candidate := strings.TrimSpace(string(b[i : j+1]))
					if strings.Contains(candidate, `"name"`) && strings.Contains(candidate, `"arguments"`) {
						result = append(result, candidate)
					}
					i = j
					break
				}
			}
		}
	}
	return result
}

func buildGeminiRequest(prompt string, images []ImageData, session *GeminiSession, modelName string, accountCtx AccountContext) (*http.Request, error) {

	uuidVal := strings.ToUpper(support.GenerateUUIDv4())
	spec := modelSpecMap["gemini-3-flash"]
	if s, ok := modelSpecMap[modelName]; ok {
		spec = s
		depGetLogger().Debug("正在使用模型: %s -> %s (capacity=%d)", modelName, spec.HexID, spec.Capacity)
	}

	sessionSnapshot := session.Snapshot()
	meta := defaultGeminiMetadata()
	if sessionSnapshot.ConversationID != "" {
		meta = sessionToGeminiMetadata(sessionSnapshot)
		depGetLogger().Debug("正在使用现有会话: c=%s, r=%s, rc=%s", sessionSnapshot.ConversationID, sessionSnapshot.ResponseID, sessionSnapshot.ChoiceID)
	} else {
		depGetLogger().Debug("正在开始新对话")
	}

	currentToken := accountCtx.Token
	if currentToken == "" {
		currentToken = depTokens.GetToken()
	}

	messageContent := []interface{}{prompt, 0, nil, nil, nil, nil, 0}

	// 根据是否为 thinking 模型决定数组长度
	thinkingLevel, featureMode, needsThinking := getThinkingLevel(modelName)
	experimentalCfg, hasExperimentalCfg := getExperimentalRequestConfig(modelName)
	reqLen := geminiInnerReqLen
	if needsThinking {
		reqLen = geminiInnerReqLenThinking
	}
	inner := make([]interface{}, reqLen)
	inner[0] = messageContent
	inner[1] = []interface{}{geminiWebLanguage}
	inner[2] = meta
	inner[6] = []interface{}{1}
	inner[geminiStreamingFlagIndex] = 1
	inner[10] = 1
	inner[11] = 0
	inner[17] = []interface{}{[]interface{}{0}}
	inner[18] = 0
	inner[27] = 1
	inner[30] = []interface{}{4}
	inner[41] = []interface{}{1}
	inner[53] = 0
	inner[59] = uuidVal
	inner[61] = []interface{}{}
	inner[68] = 2

	// 设置 Deep Think / Thinking 协议字段
	if needsThinking {
		if featureMode != 0 {
			inner[idxFeatureMode] = featureMode
		}
		inner[idxThinkingLevel] = thinkingLevel
		depGetLogger().Debug("已设置 thinking 协议字段: level=%d, featureMode=%d, reqLen=%d", thinkingLevel, featureMode, reqLen)
	} else if hasExperimentalCfg {
		inner[idxFeatureMode] = experimentalCfg.FeatureMode
		inner[idxThinkingLevel] = experimentalCfg.ThinkingLevel
		depGetLogger().Debug("已设置实验工具字段: featureMode=%d, ef=%d, xpc=%s", experimentalCfg.FeatureMode, experimentalCfg.Ef, experimentalCfg.Xpc)
	}

	innerJSON, err := json.Marshal(inner)
	if err != nil {
		return nil, fmt.Errorf("marshal f.req inner: %w", err)
	}
	freqData := fmt.Sprintf(`[null,%q]`, string(innerJSON))
	data := url.Values{}
	data.Set("at", currentToken)
	data.Set("f.req", freqData)
	endpoints := httpclient.CurrentGeminiEndpoints(depGetConfig())
	requestURL, err := buildGeminiRequestURL(endpoints.URL, accountCtx)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", requestURL, strings.NewReader(data.Encode()))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Accept", "*/*")
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36")
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded;charset=UTF-8")
	req.Header.Set("accept-language", "zh-CN")
	if accountCtx.Cookies != "" {
		req.Header.Set("Cookie", accountCtx.Cookies)
	} else if cfg := depGetConfig(); cfg.Cookies != "" {
		req.Header.Set("Cookie", cfg.Cookies)
	}
	req.Header.Set("cache-control", "no-cache")
	req.Header.Set("origin", endpoints.Origin)
	req.Header.Set("pragma", "no-cache")
	req.Header.Set("priority", "u=1, i")
	req.Header.Set("referer", endpoints.Referer)
	req.Header.Set("sec-ch-ua", `"Chromium";v="146", "Not-A.Brand";v="24", "Google Chrome";v="146"`)
	req.Header.Set("sec-ch-ua-arch", `"x86"`)
	req.Header.Set("sec-ch-ua-bitness", `"64"`)
	req.Header.Set("sec-ch-ua-form-factors", `"Desktop"`)
	req.Header.Set("sec-ch-ua-full-version", `"146.0.7680.179"`)
	req.Header.Set("sec-ch-ua-full-version-list", `"Chromium";v="146.0.7680.179", "Not-A.Brand";v="24.0.0.0", "Google Chrome";v="146.0.7680.179"`)
	req.Header.Set("sec-ch-ua-mobile", "?0")
	req.Header.Set("sec-ch-ua-model", `""`)
	req.Header.Set("sec-ch-ua-platform", `"Windows"`)
	req.Header.Set("sec-ch-ua-platform-version", `"19.0.0"`)
	req.Header.Set("sec-ch-ua-wow64", "?0")
	req.Header.Set("sec-fetch-dest", "empty")
	req.Header.Set("sec-fetch-mode", "cors")
	req.Header.Set("sec-fetch-site", "same-origin")
	req.Header.Set(headerModelJSPB, buildModelHeaderJSPB(spec, thinkingLevel, uuidVal))
	req.Header.Set(headerRequestUUIDJSPB, fmt.Sprintf(`["%s",1]`, uuidVal))
	req.Header.Set("x-goog-ext-73010989-jspb", "[0]")
	req.Header.Set("x-goog-ext-73010990-jspb", "[0]")
	req.Header.Set("x-same-domain", "1")
	return req, nil
}

func buildGeminiRequestURL(rawURL string, accountCtx AccountContext) (string, error) {
	parsedURL, err := url.Parse(rawURL)
	if err != nil {
		return "", err
	}

	query := parsedURL.Query()
	if query.Get("hl") == "" {
		query.Set("hl", "zh-CN")
	}
	if query.Get("rt") == "" {
		query.Set("rt", "c")
	}
	if query.Get("bl") == "" {
		if blToken := firstNonEmpty(accountCtx.BLToken, depTokens.GetBLToken()); blToken != "" {
			query.Set("bl", blToken)
		}
	}
	if query.Get("f.sid") == "" {
		if fsid := firstNonEmpty(accountCtx.FSID, depTokens.GetFSID()); fsid != "" {
			query.Set("f.sid", fsid)
		}
	}
	query.Set("_reqid", firstNonEmpty(accountCtx.ReqID, depTokens.NextReqID()))
	parsedURL.RawQuery = query.Encode()
	return parsedURL.String(), nil
}

func HandleStreamResponse(w http.ResponseWriter, prompt string, images []ImageData, model string, session *GeminiSession, tools []Tool, sessionKey string, snlm0eToken string, streamOptions *StreamOptions, writeError func(http.ResponseWriter, int, string), writeMappedError func(http.ResponseWriter, OpenAIError)) {
	start := time.Now()
	const maxRetries = 3
	var bodyStr, content, lastErr string
	var lastMappedErr *OpenAIError
	var accountID string

	for attempt := 1; attempt <= maxRetries; attempt++ {
		if attempt > 1 {
			depGetLogger().Info("流式请求正在进行第 %d/%d 次重试", attempt, maxRetries)
			time.Sleep(time.Duration(attempt*500) * time.Millisecond)
		}

		selected, err := depTokens.SelectAccountForSession(sessionKey, attempt > 1)
		if err != nil {
			lastErr = err.Error()
			mapped := OpenAIError{Status: http.StatusBadGateway, Type: "api_error", Code: "no_healthy_accounts", Message: err.Error()}
			lastMappedErr = &mapped
			continue
		}
		accountID = selected.ID
		accountCtx := AccountContext{
			ID:      selected.ID,
			Email:   selected.Email,
			Cookies: selected.Cookies,
			Proxy:   selected.Proxy,
			Token:   firstNonEmpty(selected.Token, snlm0eToken),
			BLToken: selected.BLToken,
			FSID:    selected.FSID,
			ReqID:   selected.ReqID,
		}

		req, err := buildGeminiRequest(prompt, images, session, model, accountCtx)
		if err != nil {
			depGetLogger().Error("构建 Gemini 请求失败: %v", err)
			depTokens.MarkAccountFailure(accountID, err.Error())
			lastErr = err.Error()
			mapped := OpenAIError{Status: http.StatusBadRequest, Type: "invalid_request_error", Code: "request_build_failed", Message: err.Error()}
			lastMappedErr = &mapped
			continue
		}

		depGetLogger().Debug("正在发送请求到 Gemini API...")
		resp, err := httpClientForAccount(accountCtx).Do(req)
		if err != nil {
			if httpclient.IsConnectionError(err) {
				depGetLogger().Warn("通过代理连接出错 (尝试 %d/%d): %v", attempt, maxRetries, err)
			} else {
				depGetLogger().Error("Gemini API 请求失败: %v", err)
			}
			if !isTransientNetworkError(err) {
				depTokens.MarkAccountFailure(accountID, err.Error())
			}
			lastErr = err.Error()
			mapped := OpenAIError{Status: http.StatusBadGateway, Type: "api_error", Code: "upstream_connection_error", Message: err.Error()}
			lastMappedErr = &mapped
			continue
		}

		depGetLogger().Debug("Gemini API 响应状态码: %d", resp.StatusCode)
		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			bodyStr = string(body)
			depGetLogger().Error("Gemini API 返回错误状态码 %d: %s", resp.StatusCode, bodyStr)
			if isHTMLErrorResponse(bodyStr) {
				depGetLogger().Warn("检测到 HTML 错误响应，已标记会话令牌失效")
				depTokens.MarkSessionTokenBad(sessionKey)
			}
			mapped := mapGeminiError(resp.StatusCode, bodyStr)
			depTokens.MarkAccountFailure(accountID, mapped.Message)
			lastErr = mapped.Message
			lastMappedErr = &mapped
			continue
		}

		streamedBody, streamedContent, err := streamGeminiResponse(w, resp, model, session, tools, streamOptions, accountCtx)
		if err != nil {
			depTokens.MarkAccountFailure(accountID, err.Error())
			lastErr = err.Error()
			mapped := OpenAIError{Status: http.StatusBadGateway, Type: "api_error", Code: "stream_read_error", Message: err.Error()}
			lastMappedErr = &mapped
			continue
		}
		bodyStr = streamedBody
		noteGeminiResponseErrors(bodyStr, sessionKey, "流式")

		if isHTMLErrorResponse(bodyStr) {
			depGetLogger().Warn("响应体中检测到 HTML 错误，已标记会话令牌失效")
			depTokens.MarkSessionTokenBad(sessionKey)
			depTokens.MarkAccountFailure(accountID, "Request failed due to token issue")
			lastErr = "Request failed due to token issue"
			mapped := OpenAIError{Status: http.StatusUnauthorized, Type: "authentication_error", Code: "token_invalid", Message: lastErr}
			lastMappedErr = &mapped
			continue
		}

		content = streamedContent

		if content == "" {
			if code, msg := parseGeminiErrorCode(bodyStr); code != 0 {
				depGetLogger().Error("流式响应无正文，错误码 %d: %s", code, msg)
				mapped := mapGeminiError(http.StatusBadGateway, bodyStr)
				depTokens.MarkAccountFailure(accountID, mapped.Message)
				lastErr = mapped.Message
				lastMappedErr = &mapped
				continue
			}
			if isEmptyAcknowledgmentResponse(bodyStr) {
				depGetLogger().Error("流式响应收到空的确认包 - 令牌可能已失效或过期")
				depTokens.MarkSessionTokenBad(sessionKey)
				depTokens.MarkAccountFailure(accountID, "Gemini returned empty response - token issue")
				lastErr = "Gemini returned empty response - token issue"
				mapped := OpenAIError{Status: http.StatusUnauthorized, Type: "authentication_error", Code: "empty_acknowledgment", Message: lastErr}
				lastMappedErr = &mapped
				continue
			}
		}

		depTokens.MarkAccountSuccess(accountID)
		lastErr = ""
		break
	}

	if lastErr != "" {
		depGetLogger().Error("所有 %d 次重试均失败，最后一次错误: %s", maxRetries, lastErr)
		depMetrics.AddRequest(false, len(prompt)/4, 0)
		if lastMappedErr != nil {
			writeMappedError(w, *lastMappedErr)
			return
		}
		writeError(w, http.StatusBadGateway, lastErr)
		return
	}

	inputTokens := len(prompt) / 4
	outputTokens := len(content) / 4
	depMetrics.AddRequest(true, inputTokens, outputTokens)
	depGetLogger().Info("流式响应完成，耗时 %.3fms", float64(time.Since(start).Microseconds())/1000)
}

func sendStreamUsageChunk(w http.ResponseWriter, flusher http.Flusher, model string, usage Usage) {
	chunk := ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []Choice{},
		Usage:   usage,
	}
	jsonData, _ := json.Marshal(chunk)
	w.Write([]byte(fmt.Sprintf("data: %s\n\n", jsonData)))
	flusher.Flush()
}

func inferStreamUsage(prompt string, content string) Usage {
	inputTokens := len(prompt) / 4
	outputTokens := len(content) / 4
	return Usage{
		PromptTokens:     inputTokens,
		CompletionTokens: outputTokens,
		TotalTokens:      inputTokens + outputTokens,
	}
}

func sendStreamChunk(w http.ResponseWriter, flusher http.Flusher, model string, content string, role string, isFinish bool) {
	chunk := ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []Choice{{Index: 0, Delta: &Delta{}}},
	}
	if role != "" {
		chunk.Choices[0].Delta.Role = role
	}
	if content != "" {
		chunk.Choices[0].Delta.Content = content
	}
	if isFinish {
		finishReason := "stop"
		chunk.Choices[0].FinishReason = &finishReason
	}
	jsonData, _ := json.Marshal(chunk)
	w.Write([]byte(fmt.Sprintf("data: %s\n\n", jsonData)))
	flusher.Flush()
}

func sendStreamChunkWithConversation(w http.ResponseWriter, flusher http.Flusher, model string, content string, role string, isFinish bool, conversationID string) {
	chunk := ChatCompletionResponse{
		ID:             fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:         "chat.completion.chunk",
		Created:        time.Now().Unix(),
		Model:          model,
		ConversationID: conversationID,
		Choices:        []Choice{{Index: 0, Delta: &Delta{}}},
	}
	if role != "" {
		chunk.Choices[0].Delta.Role = role
	}
	if content != "" {
		chunk.Choices[0].Delta.Content = content
	}
	if isFinish {
		finishReason := "stop"
		chunk.Choices[0].FinishReason = &finishReason
	}
	jsonData, _ := json.Marshal(chunk)
	w.Write([]byte(fmt.Sprintf("data: %s\n\n", jsonData)))
	flusher.Flush()
}

func sendStreamChunkWithTools(w http.ResponseWriter, flusher http.Flusher, model string, content string, toolCalls []ToolCall) {
	chunk := ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []Choice{{Index: 0, Delta: &Delta{Content: content, ToolCalls: toolCalls}}},
	}
	jsonData, _ := json.Marshal(chunk)
	w.Write([]byte(fmt.Sprintf("data: %s\n\n", jsonData)))
	flusher.Flush()
}

func sendStreamChunkFinish(w http.ResponseWriter, flusher http.Flusher, model string, finishReason string) {
	chunk := ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []Choice{{Index: 0, Delta: &Delta{}, FinishReason: &finishReason}},
	}
	jsonData, _ := json.Marshal(chunk)
	w.Write([]byte(fmt.Sprintf("data: %s\n\n", jsonData)))
	flusher.Flush()
}

func sendStreamReasoningChunk(w http.ResponseWriter, flusher http.Flusher, model string, reasoningContent string) {
	chunk := ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []Choice{{Index: 0, Delta: &Delta{ReasoningContent: reasoningContent}}},
	}
	jsonData, _ := json.Marshal(chunk)
	w.Write([]byte(fmt.Sprintf("data: %s\n\n", jsonData)))
	flusher.Flush()
}

func pollDeepThinkResult(session *GeminiSession, modelName string, accountCtx AccountContext) (string, string, error) {
	snapshot := session.Snapshot()
	if snapshot.ConversationID == "" {
		return "", "", fmt.Errorf("no conversation ID for deep think polling")
	}

	endpoints := httpclient.CurrentGeminiEndpoints(depGetConfig())
	baseURL := strings.Replace(endpoints.URL, "/assistant.lamda.BardFrontendService/StreamGenerate", "/batchexecute", 1)
	if baseURL == endpoints.URL {
		baseURL = strings.Replace(endpoints.URL, "StreamGenerate", "batchexecute", 1)
	}
	parsedURL, err := url.Parse(baseURL)
	if err != nil {
		return "", "", fmt.Errorf("parse batchexecute URL: %w", err)
	}

	currentToken := accountCtx.Token
	if currentToken == "" {
		currentToken = depTokens.GetToken()
	}

	convID := snapshot.ConversationID
	freqPayload := fmt.Sprintf(`[\"%s\",10,null,1,[0],[4],null,1]`, convID)
	freqData := fmt.Sprintf(`[[["hNvQHb","%s",null,"generic"]]]`, freqPayload)

	query := parsedURL.Query()
	query.Set("rpcids", "hNvQHb")
	query.Set("source-path", fmt.Sprintf("/app/%s", strings.TrimPrefix(convID, "c_")))
	query.Set("hl", "en-GB")
	query.Set("rt", "c")
	if blToken := firstNonEmpty(accountCtx.BLToken, depTokens.GetBLToken()); blToken != "" {
		query.Set("bl", blToken)
	}
	if fsid := firstNonEmpty(accountCtx.FSID, depTokens.GetFSID()); fsid != "" {
		query.Set("f.sid", fsid)
	}
	query.Set("_reqid", firstNonEmpty(accountCtx.ReqID, depTokens.NextReqID()))
	parsedURL.RawQuery = query.Encode()

	maxPolls := 30
	interval := 3
	var lastBody string

	for i := 0; i < maxPolls; i++ {
		time.Sleep(time.Duration(interval) * time.Second)
		interval += 2
		if interval > 15 {
			interval = 15
		}

		depGetLogger().Debug("Deep Think 轮询 %d/%d, convID=%s", i+1, maxPolls, convID)

		postData := url.Values{}
		postData.Set("f.req", freqData)
		postData.Set("at", currentToken)
		req, err := http.NewRequest("POST", parsedURL.String(), strings.NewReader(postData.Encode()))
		if err != nil {
			return "", "", fmt.Errorf("create poll request: %w", err)
		}

		req.Header.Set("Accept", "*/*")
		req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36")
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded;charset=UTF-8")
		req.Header.Set("accept-language", "zh-CN")
		if accountCtx.Cookies != "" {
			req.Header.Set("Cookie", accountCtx.Cookies)
		} else if cfg := depGetConfig(); cfg.Cookies != "" {
			req.Header.Set("Cookie", cfg.Cookies)
		}
		req.Header.Set("cache-control", "no-cache")
		req.Header.Set("origin", endpoints.Origin)
		req.Header.Set("pragma", "no-cache")
		req.Header.Set("priority", "u=1, i")
		req.Header.Set("referer", endpoints.Referer)
		req.Header.Set("sec-ch-ua", `"Chromium";v="146", "Not-A.Brand";v="24", "Google Chrome";v="146"`)
		req.Header.Set("sec-ch-ua-arch", `"x86"`)
		req.Header.Set("sec-ch-ua-bitness", `"64"`)
		req.Header.Set("sec-ch-ua-form-factors", `"Desktop"`)
		req.Header.Set("sec-ch-ua-full-version", `"146.0.7680.179"`)
		req.Header.Set("sec-ch-ua-full-version-list", `"Chromium";v="146.0.7680.179", "Not-A.Brand";v="24.0.0.0", "Google Chrome";v="146.0.7680.179"`)
		req.Header.Set("sec-ch-ua-mobile", "?0")
		req.Header.Set("sec-ch-ua-model", `""`)
		req.Header.Set("sec-ch-ua-platform", `"Windows"`)
		req.Header.Set("sec-ch-ua-platform-version", `"19.0.0"`)
		req.Header.Set("sec-ch-ua-wow64", "?0")
		req.Header.Set("sec-fetch-dest", "empty")
		req.Header.Set("sec-fetch-mode", "cors")
		req.Header.Set("sec-fetch-site", "same-origin")
		req.Header.Set(headerRequestUUIDJSPB, fmt.Sprintf(`["%s",1]`, strings.ToUpper(support.GenerateUUIDv4())))
		req.Header.Set("x-goog-ext-73010989-jspb", "[0]")
		req.Header.Set("x-goog-ext-73010990-jspb", "[0]")
		req.Header.Set("x-same-domain", "1")

		resp, err := httpClientForAccount(accountCtx).Do(req)
		if err != nil {
			depGetLogger().Warn("Deep Think 轮询请求失败: %v", err)
			continue
		}

		body, err := readResponseBody(resp, "Deep Think 轮询")
		if err != nil {
			resp.Body.Close()
			continue
		}
		lastBody = string(body)
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			depGetLogger().Warn("Deep Think 轮询返回异常状态码: %d, body preview: %.200s", resp.StatusCode, lastBody)
			continue
		}

		result := extractFinalContentWithThinking(lastBody)
		content := filterContent(result.Content)

		if content != "" && !isDeepThinkPlaceholder(content) && !strings.Contains(content, "I'm on it") {
			reasoning := result.ReasoningContent
			depGetLogger().Info("Deep Think 轮询成功, 内容长度=%d, 推理长度=%d, 轮询次数=%d", len(content), len(reasoning), i+1)
			return content, reasoning, nil
		}

		depGetLogger().Debug("Deep Think 轮询 %d: 内容仍为占位符或为空", i+1)
	}

	return "", "", fmt.Errorf("deep think polling timed out after %d attempts, last body preview: %.200s", maxPolls, lastBody)
}

func HandleNonStreamResponse(w http.ResponseWriter, prompt string, images []ImageData, model string, session *GeminiSession, tools []Tool, sessionKey string, snlm0eToken string, writeError func(http.ResponseWriter, int, string), writeMappedError func(http.ResponseWriter, OpenAIError), writeJSON func(http.ResponseWriter, int, interface{})) {
	start := time.Now()
	const maxRetries = 3
	var bodyStr, content, lastErr string
	var reasoningContent string
	var lastMappedErr *OpenAIError
	var accountID string

	for attempt := 1; attempt <= maxRetries; attempt++ {
		if attempt > 1 {
			depGetLogger().Info("非流式请求正在进行第 %d/%d 次重试", attempt, maxRetries)
			time.Sleep(time.Duration(attempt*500) * time.Millisecond)
		}

		selected, err := depTokens.SelectAccountForSession(sessionKey, attempt > 1)
		if err != nil {
			lastErr = err.Error()
			mapped := OpenAIError{Status: http.StatusBadGateway, Type: "api_error", Code: "no_healthy_accounts", Message: err.Error()}
			lastMappedErr = &mapped
			continue
		}
		accountID = selected.ID
		accountCtx := AccountContext{
			ID:      selected.ID,
			Email:   selected.Email,
			Cookies: selected.Cookies,
			Proxy:   selected.Proxy,
			Token:   firstNonEmpty(selected.Token, snlm0eToken),
			BLToken: selected.BLToken,
			FSID:    selected.FSID,
			ReqID:   selected.ReqID,
		}

		req, err := buildGeminiRequest(prompt, images, session, model, accountCtx)
		if err != nil {
			depGetLogger().Error("构建 Gemini 请求失败: %v", err)
			depTokens.MarkAccountFailure(accountID, err.Error())
			lastErr = err.Error()
			mapped := OpenAIError{Status: http.StatusBadRequest, Type: "invalid_request_error", Code: "request_build_failed", Message: err.Error()}
			lastMappedErr = &mapped
			continue
		}

		depGetLogger().Debug("正在发送请求到 Gemini API...")
		resp, err := httpClientForAccount(accountCtx).Do(req)
		if err != nil {
			if httpclient.IsConnectionError(err) {
				depGetLogger().Warn("通过代理连接出错 (尝试 %d/%d): %v", attempt, maxRetries, err)
			} else {
				depGetLogger().Error("Gemini API 请求失败: %v", err)
			}
			depTokens.MarkAccountFailure(accountID, err.Error())
			lastErr = err.Error()
			mapped := OpenAIError{Status: http.StatusBadGateway, Type: "api_error", Code: "upstream_connection_error", Message: err.Error()}
			lastMappedErr = &mapped
			continue
		}

		body, err := readResponseBody(resp, "非流式")
		if err != nil {
			if !isTransientNetworkError(err) {
				depTokens.MarkAccountFailure(accountID, err.Error())
			}
			lastErr = err.Error()
			mapped := OpenAIError{Status: http.StatusBadGateway, Type: "api_error", Code: "response_read_error", Message: err.Error()}
			lastMappedErr = &mapped
			continue
		}
		depGetLogger().Debug("Gemini API 响应状态码: %d", resp.StatusCode)
		depGetLogger().Debug("响应体大小: %d 字节", len(body))
		bodyStr = string(body)
		noteGeminiResponseErrors(bodyStr, sessionKey, "非流式")

		if resp.StatusCode != http.StatusOK {
			depGetLogger().Error("Gemini API 返回错误状态码 %d: %s", resp.StatusCode, bodyStr)
			if isHTMLErrorResponse(bodyStr) {
				depGetLogger().Warn("检测到 HTML 错误响应，已标记会话令牌失效")
				depTokens.MarkSessionTokenBad(sessionKey)
			}
			mapped := mapGeminiError(resp.StatusCode, bodyStr)
			depTokens.MarkAccountFailure(accountID, mapped.Message)
			lastErr = mapped.Message
			lastMappedErr = &mapped
			continue
		}

		if isHTMLErrorResponse(bodyStr) {
			depGetLogger().Warn("响应体中检测到 HTML 错误，已标记会话令牌失效")
			depTokens.MarkSessionTokenBad(sessionKey)
			depTokens.MarkAccountFailure(accountID, "Request failed due to token issue")
			lastErr = "Request failed due to token issue"
			mapped := OpenAIError{Status: http.StatusUnauthorized, Type: "authentication_error", Code: "token_invalid", Message: lastErr}
			lastMappedErr = &mapped
			continue
		}

		result := extractFinalContentWithThinking(bodyStr)
		content = filterContent(result.Content)
		reasoningContent = result.ReasoningContent

		if isDeepThinkPlaceholder(result.Content) && isDeepThinkAlias(model) {
			updateSessionFromResponse(session, bodyStr)
			polledContent, polledReasoning, pollErr := pollDeepThinkResult(session, model, accountCtx)
			if pollErr != nil {
				depGetLogger().Warn("Deep Think 轮询失败: %v", pollErr)
			} else {
				content = polledContent
				reasoningContent = polledReasoning
			}
		}

		if content == "" {
			if code, msg := parseGeminiErrorCode(bodyStr); code != 0 {
				depGetLogger().Error("非流式响应无正文，错误码 %d: %s", code, msg)
				mapped := mapGeminiError(http.StatusBadGateway, bodyStr)
				depTokens.MarkAccountFailure(accountID, mapped.Message)
				lastErr = mapped.Message
				lastMappedErr = &mapped
				continue
			}
			depGetLogger().Warn("从响应中提取的内容为空，响应体预览: %.500s", bodyStr)
			if isEmptyAcknowledgmentResponse(bodyStr) {
				depGetLogger().Error("收到空的确认响应 - 令牌可能已失效或过期")
				depTokens.MarkSessionTokenBad(sessionKey)
				depTokens.MarkAccountFailure(accountID, "Gemini returned empty response - token issue")
				lastErr = "Gemini returned empty response - token issue"
				mapped := OpenAIError{Status: http.StatusUnauthorized, Type: "authentication_error", Code: "empty_acknowledgment", Message: lastErr}
				lastMappedErr = &mapped
				continue
			}
		}

		depTokens.MarkAccountSuccess(accountID)
		lastErr = ""
		break
	}

	if lastErr != "" {
		depGetLogger().Error("所有 %d 次重试均失败，最后一次错误: %s", maxRetries, lastErr)
		depMetrics.AddRequest(false, len(prompt)/4, 0)
		if lastMappedErr != nil {
			writeMappedError(w, *lastMappedErr)
			return
		}
		writeError(w, http.StatusBadGateway, lastErr)
		return
	}

	updateSessionFromResponse(session, bodyStr)
	cleanContent, toolCalls := parseToolCalls(content, tools)
	cleanContent = filterContent(cleanContent)

	inputTokens := len(prompt) / 4
	outputTokens := len(content) / 4
	depMetrics.AddRequest(true, inputTokens, outputTokens)

	depGetLogger().Info("非流式响应完成，耗时 %.3fms，内容长度: %d",
		float64(time.Since(start).Microseconds())/1000, len(content))

	finishReason := "stop"
	if len(toolCalls) > 0 {
		finishReason = "tool_calls"
	}

	sessionSnapshot := session.Snapshot()

	response := ChatCompletionResponse{
		ID:             fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:         "chat.completion",
		Created:        time.Now().Unix(),
		Model:          model,
		ConversationID: sessionSnapshot.ConversationID,
		Choices: []Choice{{
			Index: 0,
			Message: &Message{
				Role:             "assistant",
				Content:          cleanContent,
				ReasoningContent: reasoningContent,
				ToolCalls:        toolCalls,
			},
			FinishReason: &finishReason,
		}},
		Usage: Usage{
			PromptTokens:     inputTokens,
			CompletionTokens: outputTokens,
			TotalTokens:      inputTokens + outputTokens,
		},
	}

	writeJSON(w, http.StatusOK, response)
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func isTransientNetworkError(err error) bool {
	if err == nil {
		return false
	}
	lower := strings.ToLower(err.Error())
	return strings.Contains(lower, "context deadline exceeded") ||
		strings.Contains(lower, "client.timeout") ||
		strings.Contains(lower, "timeout awaiting response headers") ||
		strings.Contains(lower, "i/o timeout") ||
		strings.Contains(lower, "unexpected eof")
}

func updateSessionFromResponse(session *GeminiSession, body string) {
	if session == nil {
		return
	}

	snapshot := session.Snapshot()
	convRe := regexp.MustCompile(`\\?"c_([a-f0-9]+)\\?"`)
	if matches := convRe.FindStringSubmatch(body); len(matches) > 1 {
		snapshot.ConversationID = "c_" + matches[1]
	}

	respRe := regexp.MustCompile(`\\?"r_([a-f0-9]+)\\?"`)
	if matches := respRe.FindStringSubmatch(body); len(matches) > 1 {
		snapshot.ResponseID = "r_" + matches[1]
	}

	choiceRe := regexp.MustCompile(`\\?"rc_([a-f0-9]+)\\?"`)
	if matches := choiceRe.FindStringSubmatch(body); len(matches) > 1 {
		snapshot.ChoiceID = "rc_" + matches[1]
	}

	session.mu.Lock()
	session.ConversationID = snapshot.ConversationID
	session.ResponseID = snapshot.ResponseID
	session.ChoiceID = snapshot.ChoiceID
	session.TokenIndex = snapshot.TokenIndex
	session.mu.Unlock()

	if snapshot.ConversationID != "" {
		depGetLogger().Debug("会话已更新: c=%s, r=%s, rc=%s", snapshot.ConversationID, snapshot.ResponseID, snapshot.ChoiceID)
	}
}

func extractRCText(items []interface{}) (string, bool) {
	if len(items) < 2 {
		return "", false
	}

	id, ok := items[0].(string)
	if !ok || !strings.HasPrefix(id, "rc_") {
		return "", false
	}

	textItems, ok := items[1].([]interface{})
	if !ok || len(textItems) == 0 {
		return "", false
	}

	text, ok := textItems[0].(string)
	if !ok {
		return "", false
	}

	return strings.TrimSpace(unescapeContent(text)), text != ""
}

func assembleContentFragments(contents []string) string {
	assembled := ""
	for _, raw := range contents {
		part := strings.TrimSpace(unescapeContent(raw))
		if part == "" {
			continue
		}
		if assembled == "" {
			assembled = part
			continue
		}
		if strings.Contains(assembled, part) {
			continue
		}
		if strings.Contains(part, assembled) {
			assembled = part
			continue
		}

		if overlap := suffixPrefixOverlap(assembled, part); overlap > 0 {
			assembled += part[overlap:]
			continue
		}
		if overlap := suffixPrefixOverlap(part, assembled); overlap > 0 {
			assembled = part + assembled[overlap:]
			continue
		}

		assembled += part
	}
	return assembled
}

func suffixPrefixOverlap(left string, right string) int {
	limit := len(left)
	if len(right) < limit {
		limit = len(right)
	}
	for size := limit; size >= 8; size-- {
		if left[len(left)-size:] == right[:size] {
			return size
		}
	}
	return 0
}

func extractQuotedString(s string) (string, int) {
	if len(s) == 0 {
		return "", 0
	}
	var result strings.Builder
	i := 0
	for i < len(s) {
		if s[i] == '"' {
			return result.String(), i
		} else if s[i] == '\\' && i+1 < len(s) {
			result.WriteByte(s[i])
			result.WriteByte(s[i+1])
			i += 2
		} else {
			result.WriteByte(s[i])
			i++
		}
	}
	return result.String(), i
}

func unescapeContent(s string) string {
	if s == "" {
		return ""
	}

	current := s
	for i := 0; i < 3; i++ {
		decoded, err := strconv.Unquote(`"` + strings.ReplaceAll(current, `"`, `\"`) + `"`)
		if err != nil || decoded == current {
			break
		}
		current = decoded
	}

	replacer := strings.NewReplacer(
		`\\n`, "\n",
		`\n`, "\n",
		`\\r`, "\r",
		`\r`, "\r",
		`\\t`, "\t",
		`\t`, "\t",
		`\\\"`, `"`,
		`\"`, `"`,
		`\\\\`, `\`,
	)
	return replacer.Replace(current)
}

func readResponseBody(resp *http.Response, mode string) ([]byte, error) {
	body, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err == nil {
		return body, nil
	}

	if isRetryableBodyReadError(err) && len(body) > 0 {
		depGetLogger().Warn("%s响应读取不完整，已使用部分响应继续处理: %v (已读 %d 字节)", mode, err, len(body))
		return body, nil
	}

	depGetLogger().Error("读取%s响应体失败: %v", mode, err)
	return nil, err
}

func isRetryableBodyReadError(err error) bool {
	return errors.Is(err, io.ErrUnexpectedEOF) || strings.Contains(strings.ToLower(err.Error()), "unexpected eof")
}

func parseGeminiErrorCode(body string) (int, string) {
	errorRe := regexp.MustCompile(`"errorCode"\s*:\s*(\d+)`)
	if matches := errorRe.FindStringSubmatch(body); len(matches) > 1 {
		code := 0
		fmt.Sscanf(matches[1], "%d", &code)
		if msg, ok := errorCodeMap[code]; ok {
			return code, msg
		}
		return code, "unknown_error"
	}
	return 0, ""
}

func filterContent(content string) string {
	patterns := []string{
		`温馨提示：如要解锁所有应用的完整功能，请开启 \[Gemini 应用活动记录\]\([^)]+\)\s*。?\s*`,
		`温馨提示：如要解锁所有应用的完整功能，请开启 Gemini 应用活动记录[^。]*。?\s*`,
		`温馨提示[：:][^\n]*Gemini[^\n]*活动记录[^\n]*\n?`,
		`我正在处理.*Deep Think[^\n]*\n?`,
		`正在生成回答[^\n]*\n?`,
		`稍后.*查看[^\n]*\n?`,
		`Responses with Deep Think[^\n]*\n?`,
		`check back in a bit[^\n]*\n?`,
		`http://googleusercontent\.com/agentic_processing_chip/\d+[^\n]*\n?`,
	}
	result := content
	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		result = re.ReplaceAllString(result, "")
	}
	return strings.TrimSpace(result)
}

func isDeepThinkPlaceholder(body string) bool {
	return strings.Contains(body, "agentic_processing_chip") ||
		strings.Contains(body, "Deep Think") ||
		strings.Contains(body, "正在生成回答")
}

type contentResult struct {
	Content          string
	ReasoningContent string
}

func extractFinalContentWithThinking(body string) contentResult {
	if result := extractContentFromWrbFramesV2(body); result.Content != "" || result.ReasoningContent != "" {
		return result
	}
	return extractFinalContentFallback(body)
}

func extractFinalContentFallback(body string) contentResult {
	var contents []string
	patterns := []struct {
		startPattern string
		arrPattern   string
		escaped      bool
	}{
		{`"rc_`, `",["`, false},
		{`\"rc_`, `\",[\"`, true},
	}

	for _, p := range patterns {
		idx := 0
		for {
			start := strings.Index(body[idx:], p.startPattern)
			if start == -1 {
				break
			}
			start += idx
			arrStart := strings.Index(body[start:], p.arrPattern)
			if arrStart == -1 {
				idx = start + len(p.startPattern)
				continue
			}
			if p.escaped {
				arrStart += start + len(p.arrPattern)
				endPos := strings.Index(body[arrStart:], `"]`)
				if endPos == -1 {
					idx = arrStart
					continue
				}
				content := body[arrStart : arrStart+endPos]
				if content != "" {
					contents = append(contents, content)
				}
				idx = arrStart + endPos + 2
			} else {
				arrStart += start + len(p.arrPattern)
				content, endPos := extractQuotedString(body[arrStart:])
				if content != "" {
					contents = append(contents, content)
				}
				idx = arrStart + endPos + 1
			}
		}
	}

	jsonArrayRe := regexp.MustCompile(`\[\s*"rc_[a-f0-9]+"\s*,\s*\[\s*"([^"\\]*(?:\\.[^"\\]*)*)"\s*\]`)
	matches := jsonArrayRe.FindAllStringSubmatch(body, -1)
	for _, match := range matches {
		if len(match) > 1 && match[1] != "" {
			contents = append(contents, match[1])
		}
	}

	return contentResult{Content: assembleContentFragments(contents)}
}

func extractContentFromWrbFramesV2(body string) contentResult {
	lines := strings.Split(body, "\n")
	var best contentResult

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || !strings.HasPrefix(line, "[[") {
			continue
		}

		var frames []interface{}
		if err := json.Unmarshal([]byte(line), &frames); err != nil {
			continue
		}

		for _, frame := range frames {
			frameItems, ok := frame.([]interface{})
			if !ok || len(frameItems) < 3 {
				continue
			}

			eventName, _ := frameItems[0].(string)
			if eventName != "wrb.fr" {
				continue
			}

			payload, _ := frameItems[2].(string)
			if payload == "" {
				continue
			}

			candidate := extractContentFromPayloadV2(payload)
			if len(candidate.Content) > len(best.Content) || (best.Content != "" && candidate.ReasoningContent != "" && best.ReasoningContent == "") {
				best = candidate
			}
		}
	}

	return best
}

func extractContentFromPayloadV2(payload string) contentResult {
	var data interface{}
	if err := json.Unmarshal([]byte(payload), &data); err != nil {
		return contentResult{}
	}

	var result contentResult
	visitRCNodesV2(data, &result)
	result.Content = strings.TrimSpace(result.Content)
	return result
}

func visitRCNodesV2(node interface{}, result *contentResult) {
	switch value := node.(type) {
	case []interface{}:
		if text, ok := extractRCText(value); ok && len(text) > len(result.Content) {
			result.Content = text
		}
		if thinking := extractThinkingFromRCNode(value); thinking != "" && result.ReasoningContent == "" {
			result.ReasoningContent = thinking
		}
		for _, item := range value {
			visitRCNodesV2(item, result)
		}
	}
}

func extractThinkingFromRCNode(items []interface{}) string {
	for i := len(items) - 1; i >= 3; i-- {
		arr, ok := items[i].([]interface{})
		if !ok || len(arr) == 0 {
			continue
		}
		text := extractThinkingFromIndex(arr)
		if isLikelyThinkingContent(text) {
			return text
		}
	}
	return ""
}

func isLikelyThinkingContent(text string) bool {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return false
	}
	if strings.HasPrefix(trimmed, "rc_") || strings.Contains(trimmed, "e6fa609c3fa255c0") {
		return false
	}
	markers := []string{
		"**Step",
		"Step ",
		"Thinking",
		"thinking",
		"思考",
		"推理",
		"分析",
	}
	for _, marker := range markers {
		if strings.Contains(trimmed, marker) {
			return true
		}
	}
	return false
}

func extractThinkingFromIndex(arr []interface{}) string {
	var sb strings.Builder
	for _, item := range arr {
		switch v := item.(type) {
		case string:
			trimmed := strings.TrimSpace(v)
			if trimmed != "" {
				sb.WriteString(trimmed)
			}
		case []interface{}:
			if len(v) > 0 {
				if s, ok := v[0].(string); ok {
					trimmed := strings.TrimSpace(s)
					if trimmed != "" {
						sb.WriteString(trimmed)
					}
				}
			}
		}
	}
	return strings.TrimSpace(sb.String())
}

func isEmptyAcknowledgmentResponse(body string) bool {
	hasResponseID := strings.Contains(body, `"r_`) || strings.Contains(body, `\"r_`)
	hasChoiceContent := strings.Contains(body, `"rc_`) || strings.Contains(body, `\"rc_`)
	hasNullConversation := strings.Contains(body, "[null,\"r_") || strings.Contains(body, "[null,\\\"r_")
	if hasResponseID && !hasChoiceContent && hasNullConversation {
		return true
	}
	return false
}

func isHTMLErrorResponse(body string) bool {
	htmlIndicators := []string{
		"<html",
		"<div id=\"infoDiv\"",
		"background-color:#eee",
		"我们的系统检测到",
		"异常流量",
		"自动程序发出的",
		"人机识别",
		"google.com/policies/terms",
		"服务条款",
		"display:none",
		"style.display='block'",
		"<!DOCTYPE html>",
		"<head>",
		"captcha",
		"recaptcha",
		"blocked",
		"access denied",
		"rate limit",
	}
	lower := strings.ToLower(body)
	for _, indicator := range htmlIndicators {
		if strings.Contains(lower, strings.ToLower(indicator)) {
			return true
		}
	}
	return false
}

func mapGeminiError(statusCode int, body string) OpenAIError {
	if isHTMLErrorResponse(body) {
		return OpenAIError{Status: http.StatusBadGateway, Type: "invalid_request_error", Code: "upstream_html_error", Message: "Gemini returned login, consent, or protection page"}
	}
	if code, msg := parseGeminiErrorCode(body); code != 0 {
		switch code {
		case 2, 7, 1037:
			return OpenAIError{Status: http.StatusTooManyRequests, Type: "rate_limit_error", Code: msg, Message: fmt.Sprintf("Gemini error %d: %s", code, msg)}
		case 4, 1016:
			return OpenAIError{Status: http.StatusUnauthorized, Type: "authentication_error", Code: msg, Message: fmt.Sprintf("Gemini error %d: %s", code, msg)}
		case 8:
			return OpenAIError{Status: http.StatusBadRequest, Type: "invalid_request_error", Code: msg, Message: fmt.Sprintf("Gemini error %d: %s", code, msg)}
		default:
			return OpenAIError{Status: http.StatusBadGateway, Type: "api_error", Code: msg, Message: fmt.Sprintf("Gemini error %d: %s", code, msg)}
		}
	}
	if statusCode == http.StatusUnauthorized {
		return OpenAIError{Status: http.StatusUnauthorized, Type: "authentication_error", Code: "unauthorized", Message: "Gemini unauthorized"}
	}
	if statusCode == http.StatusForbidden {
		return OpenAIError{Status: http.StatusForbidden, Type: "permission_error", Code: "forbidden", Message: "Gemini forbidden"}
	}
	if statusCode == http.StatusTooManyRequests {
		return OpenAIError{Status: http.StatusTooManyRequests, Type: "rate_limit_error", Code: "rate_limit_exceeded", Message: "Gemini rate limited the request"}
	}
	return OpenAIError{Status: http.StatusBadGateway, Type: "api_error", Code: "bad_gateway", Message: fmt.Sprintf("Gemini API error: %d", statusCode)}
}

func streamGeminiResponse(w http.ResponseWriter, resp *http.Response, model string, session *GeminiSession, tools []Tool, streamOptions *StreamOptions, accountCtx AccountContext) (string, string, error) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	flusher, ok := w.(http.Flusher)
	if !ok {
		return "", "", fmt.Errorf("streaming not supported")
	}
	body, err := readResponseBody(resp, "流式")
	if err != nil {
		return "", "", err
	}
	bodyStr := string(body)
	updateSessionFromResponse(session, bodyStr)
	sessionSnapshot := session.Snapshot()
	sendStreamChunkWithConversation(w, flusher, model, "", "assistant", false, sessionSnapshot.ConversationID)

	result := extractFinalContentWithThinking(bodyStr)
	content := filterContent(result.Content)
	reasoningContent := result.ReasoningContent

	if isDeepThinkPlaceholder(result.Content) && isDeepThinkAlias(model) {
		polledContent, polledReasoning, pollErr := pollDeepThinkResult(session, model, accountCtx)
		if pollErr != nil {
			depGetLogger().Warn("Deep Think 流式轮询失败: %v", pollErr)
		} else {
			content = polledContent
			reasoningContent = polledReasoning
		}
	}

	if reasoningContent != "" {
		sendStreamReasoningChunk(w, flusher, model, reasoningContent)
	}

	if content != "" {
		cleanContent, toolCalls := parseToolCalls(content, tools)
		cleanContent = filterContent(cleanContent)
		for _, part := range chunkText(cleanContent, 48) {
			if len(toolCalls) > 0 && part == cleanContent {
				sendStreamChunkWithTools(w, flusher, model, part, toolCalls)
			} else {
				sendStreamChunk(w, flusher, model, part, "", false)
			}
		}
	}
	_, toolCalls := parseToolCalls(content, tools)
	if len(toolCalls) > 0 {
		sendStreamChunkFinish(w, flusher, model, "tool_calls")
	} else {
		sendStreamChunk(w, flusher, model, "", "", true)
	}
	if streamOptions != nil && streamOptions.IncludeUsage {
		sendStreamUsageChunk(w, flusher, model, inferStreamUsage("", content))
	}
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	flusher.Flush()
	return bodyStr, content, nil
}

func chunkText(content string, size int) []string {
	if size <= 0 || len(content) <= size {
		return []string{content}
	}
	chunks := make([]string, 0, (len(content)/size)+1)
	reader := bufio.NewReader(strings.NewReader(content))
	for {
		buf := make([]rune, 0, size)
		for len(buf) < size {
			r, _, err := reader.ReadRune()
			if err != nil {
				break
			}
			buf = append(buf, r)
		}
		if len(buf) == 0 {
			break
		}
		chunks = append(chunks, string(buf))
	}
	return chunks
}

func checkGeminiError(body string) (bool, string) {
	code, msg := parseGeminiErrorCode(body)
	if code != 0 {
		return true, fmt.Sprintf("Gemini error code %d: %s", code, msg)
	}
	if strings.Contains(body, `"error"`) {
		errorMsgRe := regexp.MustCompile(`"error"\s*:\s*\{\s*"message"\s*:\s*"([^"]+)"`)
		if matches := errorMsgRe.FindStringSubmatch(body); len(matches) > 1 {
			return true, matches[1]
		}
	}
	return false, ""
}
