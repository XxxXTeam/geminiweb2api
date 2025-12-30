package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
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
	GeminiURL = "https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate"
)

type Config struct {
	APIKey   string   `json:"api_key"`
	Token    string   `json:"token"`
	Proxy    string   `json:"proxy"`
	Port     int      `json:"port"`
	LogFile  string   `json:"log_file"`
	LogLevel string   `json:"log_level"`
	Note     []string `json:"note"`
}

var config Config
var configMutex sync.RWMutex
var httpClient *http.Client
var requestID uint64

type GeminiSession struct {
	ConversationID string // c_xxx
	ResponseID     string // r_xxx
	ChoiceID       string // rc_xxx
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
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
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
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
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

func main() {
	if err := loadConfig(); err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}
	logger = newLogger(config.LogLevel, os.Stdout)

	initHTTPClient()
	startConfigWatcher()

	mux := http.NewServeMux()
	mux.HandleFunc("/", handleTelemetry)
	mux.HandleFunc("/v1/models", loggingMiddleware(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			logger.Warn("Invalid method %s for /v1/models", r.Method)
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		models := ModelsResponse{
			Object: "list",
			Data: []Model{
				{
					ID:      "gemini-3-flash",
					Object:  "model",
					Created: time.Now().Unix(),
					OwnedBy: "google",
				},
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
	if !exists {
		session = &GeminiSession{}
		sessions[sessionKey] = session
		logger.Debug("Created new session: %s", sessionKey)
	} else {
		logger.Debug("Using existing session: %s (c=%s)", sessionKey, session.ConversationID)
	}

	var prompt strings.Builder
	for _, msg := range req.Messages {
		switch msg.Role {
		case "system":
			prompt.WriteString(fmt.Sprintf("System: %s\n", msg.Content))
		case "user":
			prompt.WriteString(fmt.Sprintf("User: %s\n", msg.Content))
		case "assistant":
			prompt.WriteString(fmt.Sprintf("Assistant: %s\n", msg.Content))
		}
	}

	if req.Stream {
		logger.Debug("Starting stream response")
		handleStreamResponse(w, prompt.String(), req.Model, session)
	} else {
		logger.Debug("Starting non-stream response")
		handleNonStreamResponse(w, prompt.String(), req.Model, session)
	}
}

func buildGeminiRequest(prompt string, session *GeminiSession) (*http.Request, error) {
	uuid := fmt.Sprintf("%08X-%04X-%04X-%04X-%012X",
		time.Now().UnixNano()&0xFFFFFFFF,
		time.Now().UnixNano()&0xFFFF,
		0x4000|time.Now().UnixNano()&0x0FFF,
		0x8000|time.Now().UnixNano()&0x3FFF,
		time.Now().UnixNano()&0xFFFFFFFFFFFF)

	var contextArray []interface{}
	if session != nil && session.ConversationID != "" {
		contextArray = []interface{}{session.ConversationID, session.ResponseID, session.ChoiceID, nil, nil, nil, nil, nil, nil, ""}
		logger.Debug("Using existing session: c=%s, r=%s, rc=%s", session.ConversationID, session.ResponseID, session.ChoiceID)
	} else {
		contextArray = []interface{}{nil, nil, nil, nil, nil, nil, nil, nil, nil, ""}
		logger.Debug("Starting new conversation")
	}

	innerArray := []interface{}{
		[]interface{}{prompt, 0, nil, nil, nil, nil, 0},
		[]interface{}{"zh-CN"},
		contextArray,
		config.Token,
		"b4c8a140d16df3a02e732943458fa040",
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

func handleStreamResponse(w http.ResponseWriter, prompt string, model string, session *GeminiSession) {
	start := time.Now()
	req, err := buildGeminiRequest(prompt, session)
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
		logger.Error("Gemini API returned status %d: %s", resp.StatusCode, string(body))
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
	content := extractFinalContent(bodyStr)
	if content != "" {
		logger.Debug("Extracted stream content (len=%d): %.100s", len(content), content)
		sendStreamChunk(w, flusher, model, content, "", false)
	}

	updateSessionFromResponse(session, bodyStr)

	inputTokens := len(prompt) / 4
	outputTokens := len(content) / 4
	metrics.AddRequest(true, inputTokens, outputTokens)

	sendStreamChunk(w, flusher, model, "", "", true)
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

func handleNonStreamResponse(w http.ResponseWriter, prompt string, model string, session *GeminiSession) {
	start := time.Now()
	req, err := buildGeminiRequest(prompt, session)
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
	if resp.StatusCode != http.StatusOK {
		logger.Error("Gemini API returned status %d: %s", resp.StatusCode, string(body))
		writeError(w, http.StatusBadGateway, fmt.Sprintf("Gemini API error: %d", resp.StatusCode))
		return
	}

	bodyStr := string(body)
	content := extractFinalContent(bodyStr)
	if content == "" {
		logger.Warn("Empty content extracted from response, body preview: %.500s", bodyStr)
	}

	updateSessionFromResponse(session, bodyStr)

	inputTokens := len(prompt) / 4
	outputTokens := len(content) / 4
	metrics.AddRequest(true, inputTokens, outputTokens)

	logger.Info("Non-stream response completed in %.3fms, content length: %d",
		float64(time.Since(start).Microseconds())/1000, len(content))

	finishReason := "stop"
	response := ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []Choice{
			{
				Index: 0,
				Message: &Message{
					Role:    "assistant",
					Content: content,
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
	re := regexp.MustCompile(`\\"rc_[^\\]+\\",\[\\"([^"]*)\\"`)
	matches := re.FindAllStringSubmatch(body, -1)

	if len(matches) > 0 {
		longest := ""
		for _, m := range matches {
			if len(m) > 1 && len(m[1]) > len(longest) {
				longest = m[1]
			}
		}
		if longest != "" {
			return unescapeContent(longest)
		}
	}
	return ""
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
