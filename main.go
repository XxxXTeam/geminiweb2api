package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	rand2 "math/rand"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const (
	GeminiURL      = "https://gemini.arpt.io/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate"
	GeminiHomeURL  = "https://gemini.google.com/"
	DefaultBLToken = "b4c8a140d16df3a02e732943458fa040"
)

var modelIDMap = map[string]string{
	"gemini-3-flash":   "1640bdc9f7ef4826",
	"gemini-2.5-flash": "e6fa609c3fa255c0",
	"gemini-2.5-pro":   "9d8ca3786ebdfbea",
	"gemini-2-flash":   "203e6bb81620bcfe",
	"gemini-flash":     "1640bdc9f7ef4826",
	"gemini-pro":       "9d8ca3786ebdfbea",
}

type Config struct {
	APIKey   string   `json:"api_key"`
	Token    string   `json:"token"`
	Cookies  string   `json:"cookies"`
	Tokens   []string `json:"tokens"`
	Proxy    string   `json:"proxy"`
	Port     int      `json:"port"`
	LogFile  string   `json:"log_file"`
	LogLevel string   `json:"log_level"`
	Note     []string `json:"note"`
}

type TokenInfo struct {
	SNlM0e    string // at token
	BLToken   string
	FetchedAt time.Time
	mutex     sync.RWMutex
}

var tokenInfo = &TokenInfo{
	BLToken: DefaultBLToken,
}

type TokenManager struct {
	sessionTokens map[string]*AnonToken
	mutex         sync.RWMutex
}

type AnonToken struct {
	SNlM0e    string
	FetchedAt time.Time
	IsBad     bool
}

var tokenManager = &TokenManager{
	sessionTokens: make(map[string]*AnonToken),
}

func (tm *TokenManager) Init() {
}

func (tm *TokenManager) FetchAnonymousToken() (string, error) {
	req, err := http.NewRequest("GET", GeminiHomeURL, nil)
	if err != nil {
		return "", fmt.Errorf("create request failed: %v", err)
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8")
	req.Header.Set("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
	randomIP := generateRandomIP()
	req.Header.Set("X-Forwarded-For", randomIP)
	req.Header.Set("X-Real-IP", randomIP)

	resp, err := httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("unexpected status: %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read body failed: %v", err)
	}

	patterns := []string{
		`"SNlM0e":"([^"]+)"`,
		`SNlM0e\\x22:\\x22([^\\]+)\\x22`,
		`WIZ_global_data[^}]*"SNlM0e":"([^"]+)"`,
	}

	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		matches := re.FindSubmatch(body)
		if len(matches) > 1 {
			snlm0e := string(matches[1])
			logger.Debug("Fetched anonymous SNlM0e (length=%d)", len(snlm0e))
			return snlm0e, nil
		}
	}

	return "", fmt.Errorf("SNlM0e not found in anonymous page")
}
func (tm *TokenManager) GetTokenForSession(sessionKey string, isNewSession bool) (string, int) {
	tm.mutex.Lock()
	defer tm.mutex.Unlock()

	if token, exists := tm.sessionTokens[sessionKey]; exists && !isNewSession && !token.IsBad {
		if time.Since(token.FetchedAt) < 25*time.Minute {
			return token.SNlM0e, 0
		}
	}
	snlm0e, err := tm.FetchAnonymousToken()
	if err != nil {
		if token, exists := tm.sessionTokens[sessionKey]; exists {
			return token.SNlM0e, 0
		}
		return "", 0
	}

	tm.sessionTokens[sessionKey] = &AnonToken{
		SNlM0e:    snlm0e,
		FetchedAt: time.Now(),
		IsBad:     false,
	}
	logger.Debug("Assigned new anonymous token to session %s", sessionKey)
	return snlm0e, 0
}

func (tm *TokenManager) MarkTokenBad(idx int) (string, int) {
	return "", 0
}

func (tm *TokenManager) MarkSessionTokenBad(sessionKey string) {
	tm.mutex.Lock()
	defer tm.mutex.Unlock()

	if token, exists := tm.sessionTokens[sessionKey]; exists {
		token.IsBad = true
		logger.Warn("Session %s token marked as bad", sessionKey)
	}
}

var config Config
var configMutex sync.RWMutex
var httpClient *http.Client
var requestID uint64

type GeminiSession struct {
	ConversationID string // c_xxx
	ResponseID     string // r_xxx
	ChoiceID       string // rc_xxx
	TokenIndex     int
}

var sessions = make(map[string]*GeminiSession)

type Metrics struct {
	TotalRequests   uint64    `json:"total_requests"`
	SuccessRequests uint64    `json:"success_requests"`
	FailedRequests  uint64    `json:"failed_requests"`
	InputTokens     uint64    `json:"input_tokens"`
	OutputTokens    uint64    `json:"output_tokens"`
	StartTime       time.Time `json:"-"`
	RecentRequests  []int64   `json:"-"`
}

var metrics = &Metrics{
	StartTime:      time.Now(),
	RecentRequests: make([]int64, 0),
}

func (m *Metrics) AddRequest(success bool, inputTokens, outputTokens int) {
	atomic.AddUint64(&m.TotalRequests, 1)
	if success {
		atomic.AddUint64(&m.SuccessRequests, 1)
	} else {
		atomic.AddUint64(&m.FailedRequests, 1)
	}
	atomic.AddUint64(&m.InputTokens, uint64(inputTokens))
	atomic.AddUint64(&m.OutputTokens, uint64(outputTokens))
	m.RecentRequests = append(m.RecentRequests, time.Now().Unix())
}

func (m *Metrics) GetRPM() float64 {
	now := time.Now().Unix()
	oneMinuteAgo := now - 60
	count := 0
	var recent []int64
	for _, t := range m.RecentRequests {
		if t >= oneMinuteAgo {
			count++
			recent = append(recent, t)
		}
	}
	m.RecentRequests = recent
	return float64(count)
}

const (
	LogLevelDebug = "debug"
	LogLevelInfo  = "info"
	LogLevelWarn  = "warn"
	LogLevelError = "error"
)

type Logger struct {
	infoLog  *log.Logger
	warnLog  *log.Logger
	errorLog *log.Logger
	debugLog *log.Logger
	level    string
}

var logger *Logger

func newLogger(level string, file *os.File) *Logger {
	flags := log.Ldate | log.Ltime | log.Lmicroseconds
	l := &Logger{
		infoLog:  log.New(file, "[INFO]  ", flags),
		warnLog:  log.New(file, "[WARN]  ", flags),
		errorLog: log.New(file, "[ERROR] ", flags),
		debugLog: log.New(file, "[DEBUG] ", flags),
		level:    level,
	}
	return l
}

func (l *Logger) Debug(format string, v ...interface{}) {
	if l.level == LogLevelDebug {
		l.debugLog.Printf(format, v...)
	}
}

func (l *Logger) Info(format string, v ...interface{}) {
	if l.level == LogLevelDebug || l.level == LogLevelInfo {
		l.infoLog.Printf(format, v...)
	}
}

func (l *Logger) Warn(format string, v ...interface{}) {
	if l.level != LogLevelError {
		l.warnLog.Printf(format, v...)
	}
}

func (l *Logger) Error(format string, v ...interface{}) {
	l.errorLog.Printf(format, v...)
}

func getRequestID() uint64 {
	return atomic.AddUint64(&requestID, 1)
}

type loggingResponseWriter struct {
	http.ResponseWriter
	statusCode int
	size       int
}

func (lrw *loggingResponseWriter) WriteHeader(code int) {
	lrw.statusCode = code
	lrw.ResponseWriter.WriteHeader(code)
}

func (lrw *loggingResponseWriter) Write(b []byte) (int, error) {
	size, err := lrw.ResponseWriter.Write(b)
	lrw.size += size
	return size, err
}

func (lrw *loggingResponseWriter) Flush() {
	if flusher, ok := lrw.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func loggingMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		reqID := getRequestID()

		lrw := &loggingResponseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		logger.Info("[#%d] --> %s %s %s", reqID, r.Method, r.URL.Path, r.RemoteAddr)

		next(lrw, r)

		duration := time.Since(start)
		logger.Info("[#%d] <-- %d %s %d bytes %.3fms",
			reqID, lrw.statusCode, http.StatusText(lrw.statusCode), lrw.size, float64(duration.Microseconds())/1000)
	}
}

type ChatCompletionRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Stream      bool      `json:"stream"`
	Tools       []Tool    `json:"tools,omitempty"`
	ToolChoice  any       `json:"tool_choice,omitempty"`
	Temperature float64   `json:"temperature,omitempty"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
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
	Role       string      `json:"role"`
	Content    interface{} `json:"content"`
	ToolCalls  []ToolCall  `json:"tool_calls,omitempty"`
	ToolCallID string      `json:"tool_call_id,omitempty"`
}

type ContentPart struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
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
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

type Choice struct {
	Index        int      `json:"index"`
	Message      *Message `json:"message,omitempty"`
	Delta        *Delta   `json:"delta,omitempty"`
	FinishReason *string  `json:"finish_reason"`
}

type Delta struct {
	Role      string     `json:"role,omitempty"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
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

type ErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

func loadConfig() error {
	configMutex.Lock()
	defer configMutex.Unlock()

	file, err := os.Open("config.json")
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&config); err != nil {
		return err
	}

	if config.Port == 0 {
		config.Port = 8080
	}
	if config.LogLevel == "" {
		config.LogLevel = LogLevelInfo
	}
	return nil
}

func reloadConfig() error {
	if err := loadConfig(); err != nil {
		return err
	}
	configMutex.RLock()
	defer configMutex.RUnlock()
	if config.Proxy != "" {
		initHTTPClient()
	}
	logger.Info("Config reloaded successfully")
	return nil
}

func startConfigWatcher() {
	go func() {
		var lastModTime time.Time
		for {
			time.Sleep(5 * time.Second)
			info, err := os.Stat("config.json")
			if err != nil {
				continue
			}
			modTime := info.ModTime()
			if !lastModTime.IsZero() && modTime.After(lastModTime) {
				if err := reloadConfig(); err != nil {
					logger.Error("Failed to reload config: %v", err)
				}
			}
			lastModTime = modTime
		}
	}()
}

func initLogger() error {
	var output *os.File
	var err error

	if config.LogFile != "" {
		output, err = os.OpenFile(config.LogFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
		if err != nil {
			return err
		}
		logger = newLogger(config.LogLevel, io.MultiWriter(os.Stdout, output).(*os.File))
	} else {
		logger = newLogger(config.LogLevel, os.Stdout)
	}
	return nil
}

func initHTTPClient() {
	transport := &http.Transport{}

	if config.Proxy != "" {
		proxyURL, err := url.Parse(config.Proxy)
		if err == nil {
			transport.Proxy = http.ProxyURL(proxyURL)
			logger.Info("Using proxy: %s", config.Proxy)
		} else {
			logger.Warn("Invalid proxy URL: %s, error: %v", config.Proxy, err)
		}
	}

	httpClient = &http.Client{
		Transport: transport,
		Timeout:   120 * time.Second,
	}
	logger.Info("HTTP client initialized")
}
func fetchToken() error {
	if config.Cookies == "" {
		logger.Warn("No cookies configured, using config token only")
		return nil
	}

	req, err := http.NewRequest("GET", GeminiHomeURL, nil)
	if err != nil {
		return fmt.Errorf("create request failed: %v", err)
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")
	req.Header.Set("Cookie", config.Cookies)
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8")
	req.Header.Set("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
	randomIP := generateRandomIP()
	req.Header.Set("X-Forwarded-For", randomIP)
	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status: %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read body failed: %v", err)
	}
	patterns := []string{
		`"SNlM0e":"([^"]+)"`,
		`SNlM0e\\x22:\\x22([^\\]+)\\x22`,
		`WIZ_global_data[^}]*"SNlM0e":"([^"]+)"`,
	}

	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		matches := re.FindSubmatch(body)
		if len(matches) > 1 {
			tokenInfo.mutex.Lock()
			tokenInfo.SNlM0e = string(matches[1])
			tokenInfo.FetchedAt = time.Now()
			tokenInfo.mutex.Unlock()
			logger.Info("Token fetched successfully (length=%d)", len(tokenInfo.SNlM0e))
			return nil
		}
	}
	if config.Token != "" {
		tokenInfo.mutex.Lock()
		tokenInfo.SNlM0e = config.Token
		tokenInfo.FetchedAt = time.Now()
		tokenInfo.mutex.Unlock()
		logger.Info("Using configured token")
		return nil
	}

	return fmt.Errorf("SNlM0e token not found in page")
}
func getToken() string {
	tokenInfo.mutex.RLock()
	defer tokenInfo.mutex.RUnlock()
	if tokenInfo.SNlM0e != "" {
		return tokenInfo.SNlM0e
	}
	return config.Token
}
func refreshTokenIfNeeded() {
	tokenInfo.mutex.RLock()
	needRefresh := tokenInfo.SNlM0e == "" || time.Since(tokenInfo.FetchedAt) > 30*time.Minute
	tokenInfo.mutex.RUnlock()

	if needRefresh && config.Cookies != "" {
		if err := fetchToken(); err != nil {
			logger.Warn("Failed to refresh token: %v", err)
		}
	}
}

func startTokenRefresher() {
	if config.Cookies != "" {
		if err := fetchToken(); err != nil {
			logger.Warn("Initial token fetch failed: %v", err)
		}
	}
	go func() {
		ticker := time.NewTicker(25 * time.Minute)
		defer ticker.Stop()
		for range ticker.C {
			refreshTokenIfNeeded()
		}
	}()
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, message string) {
	resp := ErrorResponse{}
	resp.Error.Message = message
	resp.Error.Type = "invalid_request_error"
	writeJSON(w, status, resp)
}

func handleTelemetry(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	configMutex.RLock()
	note := config.Note
	configMutex.RUnlock()

	uptime := time.Since(metrics.StartTime).Seconds()
	response := map[string]interface{}{
		"status":           "running",
		"uptime_seconds":   int64(uptime),
		"total_requests":   atomic.LoadUint64(&metrics.TotalRequests),
		"success_requests": atomic.LoadUint64(&metrics.SuccessRequests),
		"failed_requests":  atomic.LoadUint64(&metrics.FailedRequests),
		"rpm":              metrics.GetRPM(),
		"input_tokens":     atomic.LoadUint64(&metrics.InputTokens),
		"output_tokens":    atomic.LoadUint64(&metrics.OutputTokens),
		"note":             note,
	}
	writeJSON(w, http.StatusOK, response)
}

func init() {
	rand2.Seed(time.Now().UnixNano())
}

func main() {
	if err := loadConfig(); err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}
	logger = newLogger(config.LogLevel, os.Stdout)

	initHTTPClient()
	tokenManager.Init()
	startTokenRefresher()
	startConfigWatcher()

	mux := http.NewServeMux()
	mux.HandleFunc("/", handleTelemetry)
	mux.HandleFunc("/v1/models", loggingMiddleware(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			logger.Warn("Invalid method %s for /v1/models", r.Method)
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		now := time.Now().Unix()
		models := ModelsResponse{
			Object: "list",
			Data: []Model{
				{ID: "gemini-3-flash", Object: "model", Created: now, OwnedBy: "google"},
				{ID: "gemini-2.5-flash", Object: "model", Created: now, OwnedBy: "google"},
				{ID: "gemini-2.5-pro", Object: "model", Created: now, OwnedBy: "google"},
				{ID: "gemini-2-flash", Object: "model", Created: now, OwnedBy: "google"},
			},
		}
		writeJSON(w, http.StatusOK, models)
	}))
	mux.HandleFunc("/v1/chat/completions", loggingMiddleware(handleChatCompletions))

	addr := fmt.Sprintf(":%d", config.Port)
	logger.Info("Server listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, mux))
}

func handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		logger.Warn("Invalid method %s for /v1/chat/completions", r.Method)
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	auth := r.Header.Get("Authorization")
	if auth == "" {
		logger.Warn("Missing authorization header from %s", r.RemoteAddr)
		writeError(w, http.StatusUnauthorized, "missing authorization header")
		return
	}
	auth = strings.TrimPrefix(auth, "Bearer ")
	if auth != config.APIKey {
		logger.Warn("Invalid API key attempt from %s", r.RemoteAddr)
		writeError(w, http.StatusUnauthorized, "invalid api key")
		return
	}

	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		logger.Error("Failed to decode request body: %v", err)
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	logger.Info("Chat request: model=%s, messages=%d, stream=%v", req.Model, len(req.Messages), req.Stream)
	logger.Debug("Request messages: %+v", req.Messages)

	sessionKey := r.Header.Get("X-Session-ID")
	if sessionKey == "" {
		sessionKey = fmt.Sprintf("default-%s", r.RemoteAddr)
	}

	session, exists := sessions[sessionKey]
	isNewSession := !exists
	if !exists {
		session = &GeminiSession{}
		sessions[sessionKey] = session
		logger.Debug("Created new session: %s", sessionKey)
	} else {
		logger.Debug("Using existing session: %s (c=%s)", sessionKey, session.ConversationID)
	}
	snlm0eToken, _ := tokenManager.GetTokenForSession(sessionKey, isNewSession)

	prompt := buildPrompt(req)

	if req.Stream {
		logger.Debug("Starting stream response")
		handleStreamResponse(w, prompt, req.Model, session, req.Tools, sessionKey, snlm0eToken)
	} else {
		logger.Debug("Starting non-stream response")
		handleNonStreamResponse(w, prompt, req.Model, session, req.Tools, sessionKey, snlm0eToken)
	}
}
func extractMessageContent(msg Message) string {
	switch v := msg.Content.(type) {
	case string:
		return v
	case []interface{}:
		var parts []string
		for _, part := range v {
			if p, ok := part.(map[string]interface{}); ok {
				if text, ok := p["text"].(string); ok {
					parts = append(parts, text)
				}
			}
		}
		return strings.Join(parts, "\n")
	default:
		if v != nil {
			return fmt.Sprintf("%v", v)
		}
		return ""
	}
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

func buildPrompt(req ChatCompletionRequest) string {
	var prompt strings.Builder
	toolsPrompt := buildToolsPrompt(req.Tools)
	if toolsPrompt != "" {
		prompt.WriteString(toolsPrompt)
		prompt.WriteString("\n---\n\n")
	}
	for _, msg := range req.Messages {
		content := extractMessageContent(msg)
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

	return prompt.String()
}

func generateUUIDv4() string {
	b := make([]byte, 16)
	rand.Read(b)
	b[6] = (b[6] & 0x0f) | 0x40 // version 4
	b[8] = (b[8] & 0x3f) | 0x80 // variant
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

func generateRandomIP() string {
	ips := []string{
		fmt.Sprintf("%d.%d.%d.%d", 1+rand2.Intn(126), rand2.Intn(256), rand2.Intn(256), 1+rand2.Intn(254)),
		fmt.Sprintf("%d.%d.%d.%d", 128+rand2.Intn(63), rand2.Intn(256), rand2.Intn(256), 1+rand2.Intn(254)),
		fmt.Sprintf("%d.%d.%d.%d", 192+rand2.Intn(31), rand2.Intn(256), rand2.Intn(256), 1+rand2.Intn(254)),
	}
	return ips[rand2.Intn(len(ips))]
}

func generateRandomHex(length int) string {
	b := make([]byte, length/2)
	rand.Read(b)
	return hex.EncodeToString(b)
}
func parseToolCalls(content string, tools []Tool) (string, []ToolCall) {
	if len(tools) == 0 {
		return content, nil
	}

	var toolCalls []ToolCall
	cleanContent := content
	re1 := regexp.MustCompile(`(?s)\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}`)
	matches1 := re1.FindAllStringSubmatch(content, -1)
	for i, match := range matches1 {
		name := match[1]
		args := match[2]
		for _, t := range tools {
			if t.Function.Name == name {
				toolCalls = append(toolCalls, ToolCall{
					ID:   fmt.Sprintf("call_%s_%d", generateRandomHex(8), i),
					Type: "function",
					Function: FunctionCall{
						Name:      name,
						Arguments: args,
					},
				})
				cleanContent = strings.Replace(cleanContent, match[0], "", 1)
				break
			}
		}
	}

	if len(toolCalls) > 0 {
		return strings.TrimSpace(cleanContent), toolCalls
	}
	re2 := regexp.MustCompile("(?s)```tool_call\\s*\\n?(\\{.*?\\})\\s*```")
	matches2 := re2.FindAllStringSubmatch(content, -1)
	for i, match := range matches2 {
		var tc struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}

		jsonStr := match[1]
		if err := json.Unmarshal([]byte(jsonStr), &tc); err != nil {
			logger.Debug("Failed to parse tool call: %v", err)
			continue
		}
		toolExists := false
		for _, t := range tools {
			if t.Function.Name == tc.Name {
				toolExists = true
				break
			}
		}
		if !toolExists {
			continue
		}

		toolCall := ToolCall{
			ID:   fmt.Sprintf("call_%s_%d", generateRandomHex(8), i),
			Type: "function",
			Function: FunctionCall{
				Name:      tc.Name,
				Arguments: string(tc.Arguments),
			},
		}
		toolCalls = append(toolCalls, toolCall)
		cleanContent = strings.Replace(cleanContent, match[0], "", 1)
	}

	return strings.TrimSpace(cleanContent), toolCalls
}

func buildGeminiRequest(prompt string, session *GeminiSession, modelName string, snlm0eToken string) (*http.Request, error) {
	uuid := generateUUIDv4()
	modelID := modelIDMap["gemini-3-flash"]
	if id, ok := modelIDMap[modelName]; ok {
		modelID = id
		logger.Debug("Using model: %s -> %s", modelName, modelID)
	}

	var contextArray []interface{}
	if session != nil && session.ConversationID != "" {
		contextArray = []interface{}{session.ConversationID, session.ResponseID, session.ChoiceID, nil, nil, nil, nil, nil, nil, ""}
		logger.Debug("Using existing session: c=%s, r=%s, rc=%s", session.ConversationID, session.ResponseID, session.ChoiceID)
	} else {
		contextArray = []interface{}{nil, nil, nil, nil, nil, nil, nil, nil, nil, ""}
		logger.Debug("Starting new conversation")
	}
	currentToken := snlm0eToken
	if currentToken == "" {
		currentToken = getToken()
	}
	if currentToken == "" {
		logger.Warn("No token available, request may fail")
	}

	innerArray := []interface{}{
		[]interface{}{prompt, 0, nil, nil, nil, nil, 0},
		[]interface{}{"zh-CN"},
		contextArray,
		currentToken,
		modelID,
		nil,
		[]interface{}{0},
		1, nil, nil, 9, 0, nil, nil, nil, nil, nil,
		[]interface{}{[]interface{}{1}},
		0, nil, nil, nil, nil, nil, nil, nil, nil, 1, nil, nil,
		[]interface{}{4},
		nil, nil, nil, nil, nil, nil, nil, nil, nil, nil,
		[]interface{}{2},
		nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, 0, nil, nil, nil, nil, nil,
		uuid,
		nil,
		[]interface{}{},
	}

	innerJSON, _ := json.Marshal(innerArray)
	freqData := fmt.Sprintf(`[null,%q]`, string(innerJSON))
	data := url.Values{}
	data.Set("f.req", freqData)
	req, err := http.NewRequest("POST", GeminiURL, strings.NewReader(data.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded;charset=UTF-8")
	req.Header.Set("accept-language", "zh-CN")
	if config.Cookies != "" {
		req.Header.Set("Cookie", config.Cookies)
	}
	req.Header.Set("cache-control", "no-cache")
	req.Header.Set("origin", "https://gemini.google.com")
	req.Header.Set("pragma", "no-cache")
	req.Header.Set("priority", "u=1, i")
	req.Header.Set("referer", "https://gemini.google.com/")
	req.Header.Set("sec-ch-ua", `"Not;A=Brand";v="24", "Chromium";v="128"`)
	req.Header.Set("sec-ch-ua-arch", `"x86"`)
	req.Header.Set("sec-ch-ua-bitness", `"64"`)
	req.Header.Set("sec-ch-ua-form-factors", `"Desktop"`)
	req.Header.Set("sec-ch-ua-full-version", `"128.0.6568.0"`)
	req.Header.Set("sec-ch-ua-full-version-list", `"Not;A=Brand";v="24.0.0.0", "Chromium";v="128.0.6568.0"`)
	req.Header.Set("sec-ch-ua-mobile", "?0")
	req.Header.Set("sec-ch-ua-model", `""`)
	req.Header.Set("sec-ch-ua-platform", `"Linux"`)
	req.Header.Set("sec-ch-ua-platform-version", `"6.14.0"`)
	req.Header.Set("sec-ch-ua-wow64", "?0")
	req.Header.Set("sec-fetch-dest", "empty")
	req.Header.Set("sec-fetch-mode", "cors")
	req.Header.Set("sec-fetch-site", "same-origin")
	randomIP := generateRandomIP()
	req.Header.Set("X-Forwarded-For", randomIP)
	req.Header.Set("X-Real-IP", randomIP)
	logger.Debug("Using random XFF IP: %s", randomIP)
	return req, nil
}

func escapeJSON(s string) string {
	s = strings.ReplaceAll(s, "\\", "\\\\")
	s = strings.ReplaceAll(s, "\"", "\\\"")
	s = strings.ReplaceAll(s, "\n", "\\n")
	s = strings.ReplaceAll(s, "\r", "\\r")
	s = strings.ReplaceAll(s, "\t", "\\t")
	return s
}

func handleStreamResponse(w http.ResponseWriter, prompt string, model string, session *GeminiSession, tools []Tool, sessionKey string, snlm0eToken string) {
	start := time.Now()

	req, err := buildGeminiRequest(prompt, session, model, snlm0eToken)
	if err != nil {
		logger.Error("Failed to build Gemini request: %v", err)
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	logger.Debug("Sending request to Gemini API...")
	resp, err := httpClient.Do(req)
	if err != nil {
		logger.Error("Gemini API request failed: %v", err)
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer resp.Body.Close()
	logger.Debug("Gemini API response status: %d", resp.StatusCode)
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		bodyStr := string(body)
		logger.Error("Gemini API returned status %d: %s", resp.StatusCode, bodyStr)
		if isHTMLErrorResponse(bodyStr) {
			logger.Warn("HTML error detected, marking session token as bad")
			tokenManager.MarkSessionTokenBad(sessionKey)
		}

		writeError(w, http.StatusBadGateway, fmt.Sprintf("Gemini API error: %d", resp.StatusCode))
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported")
		return
	}
	sendStreamChunk(w, flusher, model, "", "assistant", false)
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		logger.Error("Failed to read stream response: %v", err)
		sendStreamChunk(w, flusher, model, "", "", true)
		w.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
		return
	}

	logger.Debug("Stream response body size: %d bytes", len(body))
	bodyStr := string(body)
	if isHTMLErrorResponse(bodyStr) {
		logger.Warn("HTML error detected in response body, marking session token as bad")
		tokenManager.MarkSessionTokenBad(sessionKey)
		sendStreamChunk(w, flusher, model, "Request failed due to token issue, please retry.", "", false)
		sendStreamChunk(w, flusher, model, "", "", true)
		w.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
		metrics.AddRequest(false, len(prompt)/4, 0)
		return
	}

	content := extractFinalContent(bodyStr)
	content = filterContent(content)

	if content == "" && isEmptyAcknowledgmentResponse(bodyStr) {
		logger.Error("Received empty acknowledgment response in stream - token may be invalid or expired")
		tokenManager.MarkSessionTokenBad(sessionKey)
		sendStreamChunk(w, flusher, model, "Error: Gemini returned empty response - token issue, please retry.", "", false)
		sendStreamChunk(w, flusher, model, "", "", true)
		w.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
		metrics.AddRequest(false, len(prompt)/4, 0)
		return
	}

	if content != "" {
		logger.Debug("Extracted stream content (len=%d): %.100s", len(content), content)
		cleanContent, toolCalls := parseToolCalls(content, tools)
		cleanContent = filterContent(cleanContent)
		if len(toolCalls) > 0 {
			sendStreamChunkWithTools(w, flusher, model, cleanContent, toolCalls)
		} else {
			sendStreamChunk(w, flusher, model, content, "", false)
		}
	}

	updateSessionFromResponse(session, bodyStr)

	inputTokens := len(prompt) / 4
	outputTokens := len(content) / 4
	metrics.AddRequest(true, inputTokens, outputTokens)
	_, toolCalls := parseToolCalls(content, tools)
	if len(toolCalls) > 0 {
		sendStreamChunkFinish(w, flusher, model, "tool_calls")
	} else {
		sendStreamChunk(w, flusher, model, "", "", true)
	}
	w.Write([]byte("data: [DONE]\n\n"))
	flusher.Flush()
	logger.Info("Stream response completed in %.3fms", float64(time.Since(start).Microseconds())/1000)
}

func sendStreamChunk(w http.ResponseWriter, flusher http.Flusher, model string, content string, role string, isFinish bool) {
	chunk := ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []Choice{
			{
				Index: 0,
				Delta: &Delta{},
			},
		},
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
		Choices: []Choice{
			{
				Index: 0,
				Delta: &Delta{
					Content:   content,
					ToolCalls: toolCalls,
				},
			},
		},
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
		Choices: []Choice{
			{
				Index:        0,
				Delta:        &Delta{},
				FinishReason: &finishReason,
			},
		},
	}
	jsonData, _ := json.Marshal(chunk)
	w.Write([]byte(fmt.Sprintf("data: %s\n\n", jsonData)))
	flusher.Flush()
}

func handleNonStreamResponse(w http.ResponseWriter, prompt string, model string, session *GeminiSession, tools []Tool, sessionKey string, snlm0eToken string) {
	start := time.Now()

	req, err := buildGeminiRequest(prompt, session, model, snlm0eToken)
	if err != nil {
		logger.Error("Failed to build Gemini request: %v", err)
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	logger.Debug("Sending request to Gemini API...")
	resp, err := httpClient.Do(req)
	if err != nil {
		logger.Error("Gemini API request failed: %v", err)
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	defer resp.Body.Close()
	logger.Debug("Gemini API response status: %d", resp.StatusCode)

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		logger.Error("Failed to read response body: %v", err)
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	logger.Debug("Response body size: %d bytes", len(body))
	bodyStr := string(body)

	if resp.StatusCode != http.StatusOK {
		logger.Error("Gemini API returned status %d: %s", resp.StatusCode, bodyStr)
		if isHTMLErrorResponse(bodyStr) {
			logger.Warn("HTML error detected, marking session token as bad")
			tokenManager.MarkSessionTokenBad(sessionKey)
		}

		writeError(w, http.StatusBadGateway, fmt.Sprintf("Gemini API error: %d", resp.StatusCode))
		return
	}
	if isHTMLErrorResponse(bodyStr) {
		logger.Warn("HTML error detected in response body, marking session token as bad")
		tokenManager.MarkSessionTokenBad(sessionKey)
		metrics.AddRequest(false, len(prompt)/4, 0)
		writeError(w, http.StatusBadGateway, "Request failed due to token issue, please retry")
		return
	}

	content := extractFinalContent(bodyStr)
	content = filterContent(content)

	if content == "" {
		logger.Warn("Empty content extracted from response, body preview: %.500s", bodyStr)
		if isEmptyAcknowledgmentResponse(bodyStr) {
			logger.Error("Received empty acknowledgment response - token may be invalid or expired")
			tokenManager.MarkSessionTokenBad(sessionKey)
			metrics.AddRequest(false, len(prompt)/4, 0)
			writeError(w, http.StatusBadGateway, "Gemini returned empty response - token issue, please retry")
			return
		}
	}
	updateSessionFromResponse(session, bodyStr)
	cleanContent, toolCalls := parseToolCalls(content, tools)
	cleanContent = filterContent(cleanContent)

	inputTokens := len(prompt) / 4
	outputTokens := len(content) / 4
	metrics.AddRequest(true, inputTokens, outputTokens)

	logger.Info("Non-stream response completed in %.3fms, content length: %d",
		float64(time.Since(start).Microseconds())/1000, len(content))
	finishReason := "stop"
	if len(toolCalls) > 0 {
		finishReason = "tool_calls"
	}

	response := ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []Choice{
			{
				Index: 0,
				Message: &Message{
					Role:      "assistant",
					Content:   cleanContent,
					ToolCalls: toolCalls,
				},
				FinishReason: &finishReason,
			},
		},
		Usage: Usage{
			PromptTokens:     inputTokens,
			CompletionTokens: outputTokens,
			TotalTokens:      inputTokens + outputTokens,
		},
	}

	writeJSON(w, http.StatusOK, response)
}

func extractContent(line string) string {
	return extractFinalContent(line)
}

func updateSessionFromResponse(session *GeminiSession, body string) {
	if session == nil {
		return
	}

	convRe := regexp.MustCompile(`"c_([a-f0-9]+)"`)
	if matches := convRe.FindStringSubmatch(body); len(matches) > 1 {
		session.ConversationID = "c_" + matches[1]
	}

	respRe := regexp.MustCompile(`"r_([a-f0-9]+)"`)
	if matches := respRe.FindStringSubmatch(body); len(matches) > 1 {
		session.ResponseID = "r_" + matches[1]
	}

	choiceRe := regexp.MustCompile(`"rc_([a-f0-9]+)"`)
	if matches := choiceRe.FindStringSubmatch(body); len(matches) > 1 {
		session.ChoiceID = "rc_" + matches[1]
	}

	if session.ConversationID != "" {
		logger.Debug("Updated session: c=%s, r=%s, rc=%s", session.ConversationID, session.ResponseID, session.ChoiceID)
	}
}

func extractFinalContent(body string) string {
	var contents []string
	idx := 0
	for {
		start := strings.Index(body[idx:], `"rc_`)
		if start == -1 {
			break
		}
		start += idx
		arrStart := strings.Index(body[start:], `",["`)
		if arrStart == -1 {
			idx = start + 4
			continue
		}
		arrStart += start + 4
		content, endPos := extractQuotedString(body[arrStart:])
		if content != "" {
			contents = append(contents, content)
		}
		idx = arrStart + endPos + 1
	}
	idx = 0
	for {
		start := strings.Index(body[idx:], `\"rc_`)
		if start == -1 {
			break
		}
		start += idx

		arrStart := strings.Index(body[start:], `\",[\"`)
		if arrStart == -1 {
			idx = start + 4
			continue
		}
		arrStart += start + 5
		endPos := strings.Index(body[arrStart:], `\"]`)
		if endPos == -1 {
			idx = arrStart
			continue
		}
		content := body[arrStart : arrStart+endPos]
		if content != "" {
			contents = append(contents, content)
		}
		idx = arrStart + endPos + 2
	}
	longest := ""
	for _, c := range contents {
		if len(c) > len(longest) {
			longest = c
		}
	}
	return unescapeContent(longest)
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
	s = strings.ReplaceAll(s, "\\\\n", "\n")
	s = strings.ReplaceAll(s, "\\\\t", "\t")
	s = strings.ReplaceAll(s, "\\\\\"", "\"")
	s = strings.ReplaceAll(s, "\\\\'", "'")
	s = strings.ReplaceAll(s, "\\\\\\\\", "\\")
	s = strings.ReplaceAll(s, "\\n", "\n")
	s = strings.ReplaceAll(s, "\\t", "\t")
	s = strings.ReplaceAll(s, "\\\"", "\"")
	s = strings.ReplaceAll(s, "\\'", "'")
	s = strings.ReplaceAll(s, "\\\\", "\\")
	return s
}

func filterContent(content string) string {

	patterns := []string{
		`温馨提示：如要解锁所有应用的完整功能，请开启 \[Gemini 应用活动记录\]\([^)]+\)\s*。?\s*`,
		`温馨提示：如要解锁所有应用的完整功能，请开启 Gemini 应用活动记录[^。]*。?\s*`,
		`温馨提示[：:][^\n]*Gemini[^\n]*活动记录[^\n]*\n?`,
	}

	result := content
	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		result = re.ReplaceAllString(result, "")
	}

	return strings.TrimSpace(result)
}

func isEmptyAcknowledgmentResponse(body string) bool {
	hasResponseID := strings.Contains(body, `"r_`) || strings.Contains(body, `\"r_`)
	hasChoiceContent := strings.Contains(body, `"rc_`) || strings.Contains(body, `\"rc_`)
	hasNullConversation := strings.Contains(body, `[null,"r_`) || strings.Contains(body, `[null,\"r_`)
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
	}

	for _, indicator := range htmlIndicators {
		if strings.Contains(body, indicator) {
			return true
		}
	}

	return false
}
