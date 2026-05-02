package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"main/internal/config"
	"main/internal/gemini"
	"main/internal/httpclient"
	"main/internal/logging"
	"main/internal/metrics"
	"main/internal/support"
	"main/internal/token"
	"main/internal/web"
)

var (
	buildPromptWithMedia   = gemini.BuildPromptWithMedia
	handleStreamResponse    = gemini.HandleStreamResponse
	handleNonStreamResponse = gemini.HandleNonStreamResponse
)

type Server struct {
	configStore *config.Store
	metrics     *metrics.Metrics

	loggerMu sync.RWMutex
	logger   *logging.Logger

	clientMu   sync.RWMutex
	httpClient *http.Client

	tokenManager *token.Manager
	stateStore   *stateStore

	sessions   map[string]*gemini.GeminiSession
	sessionsMu sync.RWMutex

	discoveredModelsMu sync.RWMutex
	discoveredModels   map[string]time.Time
}

type loggingResponseWriter struct {
	http.ResponseWriter
	statusCode int
	size       int
}

type updateCookiesRequest struct {
	Cookies string `json:"cookies"`
	Email   string `json:"email"`
	Proxy   string `json:"proxy"`
	Persist *bool  `json:"persist"`
}

type upsertAccountRequest struct {
	ID      string `json:"id"`
	Email   string `json:"email"`
	Cookies string `json:"cookies"`
	Token   string `json:"token"`
	Proxy   string `json:"proxy"`
	Enabled *bool  `json:"enabled"`
	Weight  int    `json:"weight"`
}

type rebindSessionRequest struct {
	AccountID string `json:"account_id"`
}

type responseCapture struct {
	header     http.Header
	body       []byte
	statusCode int
}

func (r *responseCapture) Header() http.Header        { return r.header }
func (r *responseCapture) WriteHeader(statusCode int) { r.statusCode = statusCode }
func (r *responseCapture) Write(data []byte) (int, error) {
	if r.statusCode == 0 {
		r.statusCode = http.StatusOK
	}
	r.body = append(r.body, data...)
	return len(data), nil
}

type webLoginRequest struct {
	APIKey string `json:"api_key"`
}

func New(configStore *config.Store) (*Server, error) {
	s := &Server{
		configStore:      configStore,
		metrics:          metrics.New(),
		sessions:         make(map[string]*gemini.GeminiSession),
		discoveredModels: make(map[string]time.Time),
	}

	if err := s.reloadRuntime(); err != nil {
		return nil, err
	}
	s.stateStore = newStateStore(configStore.Path())

	s.tokenManager = token.NewManager(s.ConfigSnapshot, s.HTTPClient, s.Logger, s.configStore.Update)
	if err := s.loadPersistentState(); err != nil {
		s.Logger().Warn("加载持久化状态失败: %v", err)
	}
	gemini.Initialize(s.ConfigSnapshot, s.HTTPClient, s.Logger, s.metrics, s.tokenManager)
	return s, nil
}

func (s *Server) Run() error {
	s.printBanner()
	s.tokenManager.StartRefresher()
	s.configStore.Watch(s.reloadConfig)

	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleIndex)
	mux.HandleFunc("/help", web.HandleHelp)
	mux.HandleFunc("/help/", web.HandleHelp)
	mux.HandleFunc("/login", web.HandleLogin)
	mux.HandleFunc("/api/web/login", s.loggingMiddleware(s.handleWebLogin))
	mux.HandleFunc("/api/web/logout", s.loggingMiddleware(s.handleWebLogout))
	mux.HandleFunc("/api/telemetry", s.handleTelemetry)
	mux.HandleFunc("/healthz", s.handleHealthz)
	mux.HandleFunc("/api/session/cookies", s.loggingMiddleware(s.handleUpdateCookies))
	mux.HandleFunc("/api/accounts", s.loggingMiddleware(s.handleAccounts))
	mux.HandleFunc("/api/accounts/", s.loggingMiddleware(s.handleAccountAction))
	mux.HandleFunc("/api/accounts/bindings", s.loggingMiddleware(s.handleAccountBindings))
	mux.HandleFunc("/api/accounts/refresh-all", s.loggingMiddleware(s.handleAccountsRefreshAll))
	mux.HandleFunc("/api/accounts/bindings/", s.loggingMiddleware(s.handleBindingAction))
	mux.HandleFunc("/v1/models", s.loggingMiddleware(s.handleModels))
	mux.HandleFunc("/v1/responses", s.loggingMiddleware(s.handleResponses))
	mux.HandleFunc("/v1/chat/completions", s.loggingMiddleware(s.handleChatCompletions))

	cfg := s.ConfigSnapshot()
	addr := fmt.Sprintf(":%d", cfg.Port)
	s.Logger().Info("服务器已启动，正在监听 %s", addr)
	return http.ListenAndServe(addr, mux)
}

func (s *Server) ConfigSnapshot() config.Config {
	return s.configStore.Snapshot()
}

func (s *Server) Logger() *logging.Logger {
	s.loggerMu.RLock()
	defer s.loggerMu.RUnlock()
	return s.logger
}

func (s *Server) HTTPClient() *http.Client {
	s.clientMu.RLock()
	defer s.clientMu.RUnlock()
	return s.httpClient
}

func (s *Server) reloadConfig() error {
	if err := s.configStore.Reload(); err != nil {
		return err
	}
	if err := s.reloadRuntime(); err != nil {
		return err
	}
	s.tokenManager.RefreshAccountsFromConfig()
	gemini.Initialize(s.ConfigSnapshot, s.HTTPClient, s.Logger, s.metrics, s.tokenManager)
	s.Logger().Info("配置文件已成功重载")
	return nil
}

func (s *Server) reloadRuntime() error {
	cfg := s.ConfigSnapshot()

	logger, err := logging.NewFromConfig(cfg.LogLevel, cfg.LogFile)
	if err != nil {
		return err
	}
	client := httpclient.New(cfg, logger)

	s.loggerMu.Lock()
	s.logger = logger
	s.loggerMu.Unlock()

	s.clientMu.Lock()
	s.httpClient = client
	s.clientMu.Unlock()
	return nil
}

func (s *Server) loadPersistentState() error {
	if s.stateStore == nil {
		return nil
	}
	state, err := s.stateStore.load()
	if err != nil {
		return err
	}
	bindings := make([]token.SessionBinding, 0, len(state.SessionBindings))
	for sessionKey, binding := range state.SessionBindings {
		bindings = append(bindings, token.SessionBinding{SessionKey: sessionKey, AccountID: binding.AccountID, BoundAt: binding.BoundAt, LastUsedAt: binding.LastUsedAt})
	}
	s.tokenManager.RestoreSessionBindings(bindings)
	tokenSnapshots := make(map[string]token.AccountTokenSnapshot, len(state.AccountTokens))
	for accountID, snapshot := range state.AccountTokens {
		tokenSnapshots[accountID] = token.AccountTokenSnapshot{
			SNlM0e:    snapshot.SNlM0e,
			BLToken:   snapshot.BLToken,
			FSID:      snapshot.FSID,
			ReqID:     snapshot.ReqID,
			FetchedAt: snapshot.FetchedAt,
		}
	}
	s.tokenManager.RestoreTokenSnapshots(tokenSnapshots)
	return nil
}

func (s *Server) savePersistentState() {
	if s.stateStore == nil {
		return
	}
	bindings := s.tokenManager.SessionBindings()
	tokenSnapshots := s.tokenManager.TokenSnapshots()
	state := persistentState{SessionBindings: map[string]persistentBinding{}, AccountTokens: map[string]tokenSnapshot{}}
	for _, binding := range bindings {
		state.SessionBindings[binding.SessionKey] = persistentBinding{AccountID: binding.AccountID, BoundAt: binding.BoundAt, LastUsedAt: binding.LastUsedAt}
	}
	for accountID, snapshot := range tokenSnapshots {
		state.AccountTokens[accountID] = tokenSnapshot{
			SNlM0e:    snapshot.SNlM0e,
			BLToken:   snapshot.BLToken,
			FSID:      snapshot.FSID,
			ReqID:     snapshot.ReqID,
			FetchedAt: snapshot.FetchedAt,
		}
	}
	if err := s.stateStore.save(state); err != nil {
		s.Logger().Warn("保存持久化状态失败: %v", err)
	}
}

func (s *Server) printBanner() {
	cfg := s.ConfigSnapshot()
	println("======================================================")
	println("           Gemini Web 2 API 启动成功")
	println("======================================================")
	println("作者: XxxXTeam")
	println("交流群: 1081291958")
	println("------------------------------------------------------")
	println("功能特性:")
	println("1. 兼容 OpenAI Chat API 格式")
	println("2. 支持 SOCKS5/HTTP 代理配置")
	println("3. 自动管理与刷新 Google Gemini 会话 Token")
	println("4. 启动时自动检测代理连通性")
	println("5. 配置文件 (config.json) 自动生成与热加载")
	println("------------------------------------------------------")
	println("使用说明:")
	println("- API端口: " + fmt.Sprintf("%d", cfg.Port))
	println("- API地址: http://localhost:" + fmt.Sprintf("%d", cfg.Port))
	println("- 核心接口: /v1/chat/completions")
	println("- 监控面板: / (Dashboard)")
	println("- 遥测接口: /api/telemetry (JSON)")
	println("- 帮助文档: /help (教程 + 示例)")
	println("======================================================")
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

func (s *Server) loggingMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		reqID := support.NextRequestID()
		lrw := &loggingResponseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		s.Logger().Info("[#%d] 收到请求 --> %s %s %s", reqID, r.Method, r.URL.Path, r.RemoteAddr)
		next(lrw, r)

		duration := time.Since(start)
		s.Logger().Info("[#%d] 请求完成 <-- %d %s %d 字节 %.3fms",
			reqID, lrw.statusCode, http.StatusText(lrw.statusCode), lrw.size, float64(duration.Microseconds())/1000)
	}
}

func (s *Server) writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func (s *Server) writeError(w http.ResponseWriter, status int, message string) {
	resp := gemini.ErrorResponse{}
	resp.Error.Message = message
	resp.Error.Type = "invalid_request_error"
	resp.Error.Code = strings.ToLower(strings.ReplaceAll(http.StatusText(status), " ", "_"))
	s.writeJSON(w, status, resp)
}

func (s *Server) writeMappedError(w http.ResponseWriter, err gemini.OpenAIError) {
	resp := gemini.ErrorResponse{}
	resp.Error.Message = err.Message
	resp.Error.Type = err.Type
	resp.Error.Code = err.Code
	s.writeJSON(w, err.Status, resp)
}

func (s *Server) authenticateRequest(r *http.Request) error {
	auth := r.Header.Get("Authorization")
	cfg := s.ConfigSnapshot()
	if auth == "" {
		if cookie, err := r.Cookie("geminiweb2api_session"); err == nil && cookie.Value == cfg.APIKey {
			return nil
		}
		return fmt.Errorf("缺失 authorization 请求头")
	}

	auth = strings.TrimPrefix(auth, "Bearer ")
	if auth != cfg.APIKey {
		return fmt.Errorf("无效的 api key")
	}
	return nil
}

func (s *Server) webAuthenticated(r *http.Request) bool {
	cookie, err := r.Cookie("geminiweb2api_session")
	return err == nil && cookie.Value == s.ConfigSnapshot().APIKey
}

func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	if !s.webAuthenticated(r) {
		http.Redirect(w, r, "/login", http.StatusFound)
		return
	}
	web.HandleIndex(w, r)
}

func (s *Server) handleWebLogin(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, "请求方法不允许")
		return
	}
	var req webLoginRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if strings.TrimSpace(req.APIKey) != s.ConfigSnapshot().APIKey {
		s.writeError(w, http.StatusUnauthorized, "无效的 api key")
		return
	}
	http.SetCookie(w, &http.Cookie{
		Name:     "geminiweb2api_session",
		Value:    s.ConfigSnapshot().APIKey,
		Path:     "/",
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
		MaxAge:   86400,
	})
	s.writeJSON(w, http.StatusOK, map[string]interface{}{"message": "logged in"})
}

func (s *Server) handleWebLogout(w http.ResponseWriter, r *http.Request) {
	http.SetCookie(w, &http.Cookie{Name: "geminiweb2api_session", Value: "", Path: "/", HttpOnly: true, SameSite: http.SameSiteLaxMode, MaxAge: -1})
	s.writeJSON(w, http.StatusOK, map[string]interface{}{"message": "logged out"})
}

func (s *Server) requireAuth(w http.ResponseWriter, r *http.Request) bool {
	if err := s.authenticateRequest(r); err != nil {
		s.writeError(w, http.StatusUnauthorized, err.Error())
		return false
	}
	return true
}

func (s *Server) handleTelemetry(w http.ResponseWriter, _ *http.Request) {
	note := s.ConfigSnapshot().Note
	uptime := time.Since(s.metrics.StartTime).Seconds()
	response := map[string]interface{}{
		"status":           "running",
		"uptime_seconds":   int64(uptime),
		"total_requests":   atomic.LoadUint64(&s.metrics.TotalRequests),
		"success_requests": atomic.LoadUint64(&s.metrics.SuccessRequests),
		"failed_requests":  atomic.LoadUint64(&s.metrics.FailedRequests),
		"rpm":              s.metrics.GetRPM(),
		"input_tokens":     atomic.LoadUint64(&s.metrics.InputTokens),
		"output_tokens":    atomic.LoadUint64(&s.metrics.OutputTokens),
		"note":             note,
	}
	s.writeJSON(w, http.StatusOK, response)
}

func (s *Server) handleHealthz(w http.ResponseWriter, _ *http.Request) {
	stats := s.tokenManager.PoolStats()
	status := http.StatusOK
	body := map[string]interface{}{"status": "ok", "healthy_accounts": stats.HealthyAccounts, "enabled_accounts": stats.EnabledAccounts}
	if stats.HealthyAccounts == 0 {
		status = http.StatusServiceUnavailable
		body["status"] = "degraded"
	}
	s.writeJSON(w, status, body)
}

func (s *Server) recordDiscoveredModel(model string) {
	model = strings.TrimSpace(model)
	if model == "" {
		return
	}
	s.discoveredModelsMu.Lock()
	defer s.discoveredModelsMu.Unlock()
	s.discoveredModels[model] = time.Now()
}

func (s *Server) currentModelList() []string {
	s.discoveredModelsMu.RLock()
	models := make([]string, 0, len(s.discoveredModels))
	for model := range s.discoveredModels {
		models = append(models, model)
	}
	s.discoveredModelsMu.RUnlock()
	if len(models) > 0 {
		slices.Sort(models)
		return models
	}
	configured := s.ConfigSnapshot().Models
	if len(configured) > 0 {
		return configured
	}
	return []string{"gemini-3-pro", "gemini-3-pro-deep-think", "gemini-3-flash"}
}

func (s *Server) normalizeModel(model string) string {
	model = strings.TrimSpace(model)
	if alias := s.ConfigSnapshot().ModelAliases[model]; strings.TrimSpace(alias) != "" {
		return strings.TrimSpace(alias)
	}
	return model
}

func (s *Server) handleUpdateCookies(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, "请求方法不允许")
		return
	}

	cfg := s.ConfigSnapshot()
	auth := strings.TrimSpace(strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer "))
	if auth == "" || auth != cfg.APIKey {
		s.writeError(w, http.StatusUnauthorized, "无效的回调凭证")
		return
	}

	var req updateCookiesRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	req.Cookies = strings.TrimSpace(req.Cookies)
	if req.Cookies == "" {
		s.writeError(w, http.StatusBadRequest, "cookies 不能为空")
		return
	}

	persist := true
	if req.Persist != nil {
		persist = *req.Persist
	}

	updateFn := s.configStore.UpdateInMemory
	updateErrMsg := "更新运行时配置失败"
	if persist {
		updateFn = s.configStore.Update
		updateErrMsg = "写入配置失败"
	}

	callbackAccountID := accountIDFromEmail(req.Email)
	if err := updateFn(func(cfg *config.Config) error {
		accountID := callbackAccountID
		if accountID != "" {
			account := config.Account{
				ID:      accountID,
				Email:   strings.TrimSpace(req.Email),
				Cookies: req.Cookies,
				Proxy:   strings.TrimSpace(req.Proxy),
				Enabled: true,
				Weight:  1,
			}
			updated := false
			for i := range cfg.Accounts {
				if cfg.Accounts[i].ID == account.ID || (account.Email != "" && cfg.Accounts[i].Email == account.Email) {
					cfg.Accounts[i].ID = account.ID
					cfg.Accounts[i].Email = account.Email
					cfg.Accounts[i].Cookies = account.Cookies
					cfg.Accounts[i].Proxy = account.Proxy
					cfg.Accounts[i].Enabled = true
					if cfg.Accounts[i].Weight <= 0 {
						cfg.Accounts[i].Weight = 1
					}
					updated = true
					break
				}
			}
			if !updated {
				cfg.Accounts = append(cfg.Accounts, account)
			}
			return nil
		}
		cfg.Cookies = req.Cookies
		return nil
	}); err != nil {
		s.Logger().Error("更新 cookies 失败: %v", err)
		s.writeError(w, http.StatusInternalServerError, updateErrMsg)
		return
	}

	if err := s.reloadRuntime(); err != nil {
		s.Logger().Error("重载运行时失败: %v", err)
		s.writeError(w, http.StatusInternalServerError, "重载运行时失败")
		return
	}
	gemini.Initialize(s.ConfigSnapshot, s.HTTPClient, s.Logger, s.metrics, s.tokenManager)

	refreshErr := error(nil)
	if accountID := callbackAccountID; accountID != "" {
		refreshErr = s.tokenManager.RefreshAccountNow(accountID)
	} else {
		refreshErr = s.tokenManager.RefreshTokenNow()
	}
	if refreshErr != nil {
		s.Logger().Error("刷新 token 失败: %v", refreshErr)
		s.writeError(w, http.StatusBadGateway, fmt.Sprintf("cookies 已接收但刷新 token 失败: %v", refreshErr))
		return
	}

	s.Logger().Info("cookies 回调更新成功: email=%s persist=%v", req.Email, persist)
	s.writeJSON(w, http.StatusOK, map[string]interface{}{
		"code":       0,
		"message":    "cookies updated",
		"persist":    persist,
		"email":      req.Email,
		"account_id": callbackAccountID,
	})
}

func accountIDFromEmail(email string) string {
	email = strings.TrimSpace(strings.ToLower(email))
	if email == "" {
		return ""
	}
	replacer := strings.NewReplacer("@", "_", ".", "_", "+", "_", "-", "_")
	return replacer.Replace(email)
}

func (s *Server) handleAccounts(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		if !s.ConfigSnapshot().PublicAccountStatus && !s.requireAuth(w, r) {
			return
		}
		s.writeJSON(w, http.StatusOK, map[string]interface{}{
			"accounts": s.tokenManager.AccountsStatus(),
			"bindings": s.tokenManager.SessionBindings(),
			"stats":    s.tokenManager.PoolStats(),
		})
		return
	}
	if !s.requireAuth(w, r) {
		return
	}
	if r.Method != http.MethodPost && r.Method != http.MethodPut {
		s.writeError(w, http.StatusMethodNotAllowed, "请求方法不允许")
		return
	}

	var req upsertAccountRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	enabled := true
	if req.Enabled != nil {
		enabled = *req.Enabled
	}
	account := config.Account{
		ID:      strings.TrimSpace(req.ID),
		Email:   strings.TrimSpace(req.Email),
		Cookies: strings.TrimSpace(req.Cookies),
		Token:   strings.TrimSpace(req.Token),
		Proxy:   strings.TrimSpace(req.Proxy),
		Enabled: enabled,
		Weight:  req.Weight,
	}
	if err := s.tokenManager.UpsertAccount(account); err != nil {
		s.writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if err := s.reloadRuntime(); err != nil {
		s.writeError(w, http.StatusInternalServerError, "重载运行时失败")
		return
	}
	s.writeJSON(w, http.StatusOK, map[string]interface{}{"message": "account updated", "account_id": account.ID})
}

func (s *Server) handleAccountBindings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeError(w, http.StatusMethodNotAllowed, "请求方法不允许")
		return
	}
	if !s.ConfigSnapshot().PublicAccountStatus && !s.requireAuth(w, r) {
		return
	}
	s.writeJSON(w, http.StatusOK, map[string]interface{}{"bindings": s.tokenManager.SessionBindings(), "stats": s.tokenManager.PoolStats()})
}

func (s *Server) handleAccountsRefreshAll(w http.ResponseWriter, r *http.Request) {
	if !s.requireAuth(w, r) {
		return
	}
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, "请求方法不允许")
		return
	}
	if err := s.tokenManager.RefreshTokenNow(); err != nil {
		s.writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	s.writeJSON(w, http.StatusOK, map[string]interface{}{"message": "all accounts refreshed"})
}

func (s *Server) handleBindingAction(w http.ResponseWriter, r *http.Request) {
	if !s.requireAuth(w, r) {
		return
	}
	path := strings.TrimPrefix(r.URL.Path, "/api/accounts/bindings/")
	parts := strings.Split(strings.Trim(path, "/"), "/")
	if len(parts) < 2 {
		s.writeError(w, http.StatusNotFound, "绑定操作不存在")
		return
	}
	sessionKey := parts[0]
	action := parts[1]
	switch action {
	case "unbind":
		if r.Method != http.MethodPost && r.Method != http.MethodDelete {
			s.writeError(w, http.StatusMethodNotAllowed, "请求方法不允许")
			return
		}
		if err := s.tokenManager.UnbindSession(sessionKey); err != nil {
			s.writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		s.writeJSON(w, http.StatusOK, map[string]interface{}{"message": "session unbound", "session_key": sessionKey})
		s.savePersistentState()
	case "rebind":
		if r.Method != http.MethodPost {
			s.writeError(w, http.StatusMethodNotAllowed, "请求方法不允许")
			return
		}
		var req rebindSessionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			s.writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		if err := s.tokenManager.RebindSession(sessionKey, strings.TrimSpace(req.AccountID)); err != nil {
			s.writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		s.savePersistentState()
		s.writeJSON(w, http.StatusOK, map[string]interface{}{"message": "session rebound", "session_key": sessionKey, "account_id": strings.TrimSpace(req.AccountID)})
	default:
		s.writeError(w, http.StatusNotFound, "绑定操作不存在")
	}
}

func (s *Server) handleAccountAction(w http.ResponseWriter, r *http.Request) {
	if !s.requireAuth(w, r) {
		return
	}
	path := strings.TrimPrefix(r.URL.Path, "/api/accounts/")
	parts := strings.Split(strings.Trim(path, "/"), "/")
	if len(parts) < 2 {
		s.writeError(w, http.StatusNotFound, "账号操作不存在")
		return
	}
	accountID := parts[0]
	action := parts[1]
	switch action {
	case "enable":
		if r.Method != http.MethodPost {
			s.writeError(w, http.StatusMethodNotAllowed, "请求方法不允许")
			return
		}
		if err := s.tokenManager.SetAccountEnabled(accountID, true); err != nil {
			s.writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		s.writeJSON(w, http.StatusOK, map[string]interface{}{"message": "account enabled", "account_id": accountID})
	case "disable":
		if r.Method != http.MethodPost {
			s.writeError(w, http.StatusMethodNotAllowed, "请求方法不允许")
			return
		}
		if err := s.tokenManager.SetAccountEnabled(accountID, false); err != nil {
			s.writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		s.writeJSON(w, http.StatusOK, map[string]interface{}{"message": "account disabled", "account_id": accountID})
	case "refresh":
		if r.Method != http.MethodPost {
			s.writeError(w, http.StatusMethodNotAllowed, "请求方法不允许")
			return
		}
		if err := s.tokenManager.RefreshAccountNow(accountID); err != nil {
			s.writeError(w, http.StatusBadGateway, err.Error())
			return
		}
		s.writeJSON(w, http.StatusOK, map[string]interface{}{"message": "account refreshed", "account_id": accountID})
	case "details":
		if r.Method != http.MethodGet {
			s.writeError(w, http.StatusMethodNotAllowed, "请求方法不允许")
			return
		}
		cfg := s.ConfigSnapshot()
		for _, account := range cfg.Accounts {
			if account.ID == accountID {
				s.writeJSON(w, http.StatusOK, account)
				return
			}
		}
		if accountID == "__default__" && len(cfg.Accounts) == 0 {
			s.writeJSON(w, http.StatusOK, config.Account{
				ID:      "__default__",
				Email:   "default",
				Cookies: cfg.Cookies,
				Token:   cfg.Token,
				Proxy:   cfg.Proxy,
				Enabled: true,
				Weight:  1,
			})
			return
		}
		s.writeError(w, http.StatusNotFound, "account not found")
	case "cookie-health":
		if r.Method != http.MethodGet {
			s.writeError(w, http.StatusMethodNotAllowed, "请求方法不允许")
			return
		}
		health, ok := s.tokenManager.CookieHealth(accountID)
		if !ok {
			s.writeError(w, http.StatusNotFound, "account not found")
			return
		}
		s.writeJSON(w, http.StatusOK, health)
	case "delete":
		if r.Method != http.MethodPost && r.Method != http.MethodDelete {
			s.writeError(w, http.StatusMethodNotAllowed, "请求方法不允许")
			return
		}
		if err := s.tokenManager.DeleteAccount(accountID); err != nil {
			s.writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		s.writeJSON(w, http.StatusOK, map[string]interface{}{"message": "account deleted", "account_id": accountID})
	default:
		s.writeError(w, http.StatusNotFound, "账号操作不存在")
	}
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.Logger().Warn("接口 /v1/models 收到无效的请求方法: %s", r.Method)
		s.writeError(w, http.StatusMethodNotAllowed, "请求方法不允许")
		return
	}

	if err := s.authenticateRequest(r); err != nil {
		s.Logger().Warn("来自 %s 的 /v1/models 请求鉴权失败: %v", r.RemoteAddr, err)
		s.writeError(w, http.StatusUnauthorized, err.Error())
		return
	}

	now := time.Now().Unix()
	modelIDs := s.currentModelList()
	data := make([]gemini.Model, 0, len(modelIDs))
	for _, id := range modelIDs {
		data = append(data, gemini.Model{ID: id, Object: "model", Created: now, OwnedBy: "google"})
	}
	models := gemini.ModelsResponse{
		Object: "list",
		Data:   data,
	}
	s.writeJSON(w, http.StatusOK, models)
}

func (s *Server) handleResponses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, "请求方法不允许")
		return
	}
	if err := s.authenticateRequest(r); err != nil {
		s.writeError(w, http.StatusUnauthorized, err.Error())
		return
	}
	var req gemini.ResponsesRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	inputText := strings.TrimSpace(fmt.Sprint(req.Input))
	if inputText == "" {
		s.writeError(w, http.StatusBadRequest, "input 不能为空")
		return
	}
	chatReq := gemini.ChatCompletionRequest{Model: s.normalizeModel(req.Model), Stream: false, Messages: []gemini.Message{{Role: "user", Content: inputText}}}
	body, _ := json.Marshal(chatReq)
	proxyReq := r.Clone(r.Context())
	proxyReq.Body = io.NopCloser(strings.NewReader(string(body)))
	proxyReq.ContentLength = int64(len(body))
	proxyReq.Header.Set("Content-Type", "application/json")
	wrapper := &responseCapture{header: http.Header{}}
	s.handleChatCompletions(wrapper, proxyReq)
	if wrapper.statusCode >= 400 {
		for k, values := range wrapper.header {
			for _, v := range values {
				w.Header().Add(k, v)
			}
		}
		w.WriteHeader(wrapper.statusCode)
		_, _ = w.Write(wrapper.body)
		return
	}
	var chatResp gemini.ChatCompletionResponse
	if err := json.Unmarshal(wrapper.body, &chatResp); err != nil {
		s.writeError(w, http.StatusBadGateway, "无法解析 chat completion 响应")
		return
	}
	resolvedModel := s.normalizeModel(req.Model)
	s.recordDiscoveredModel(resolvedModel)
	resp := gemini.ResponsesResponse{ID: chatResp.ID, Object: "response", CreatedAt: chatResp.Created, Model: resolvedModel}
	if len(chatResp.Choices) > 0 && chatResp.Choices[0].Message != nil {
		item := struct {
			Type    string `json:"type"`
			Role    string `json:"role"`
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		}{Type: "message", Role: "assistant"}
		item.Content = append(item.Content, struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}{Type: "output_text", Text: fmt.Sprint(chatResp.Choices[0].Message.Content)})
		resp.Output = append(resp.Output, item)
	}
	s.writeJSON(w, http.StatusOK, resp)
}

func (s *Server) getOrCreateSession(sessionKey, conversationID string) (*gemini.GeminiSession, bool) {
	s.sessionsMu.RLock()
	session, exists := s.sessions[sessionKey]
	s.sessionsMu.RUnlock()
	if exists {
		return session, false
	}

	s.sessionsMu.Lock()
	defer s.sessionsMu.Unlock()

	if session, exists = s.sessions[sessionKey]; exists {
		return session, false
	}

	session = &gemini.GeminiSession{}
	if conversationID != "" {
		session.SetConversationID(conversationID)
	}
	s.sessions[sessionKey] = session
	return session, true
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.Logger().Warn("接口 /v1/chat/completions 收到无效的请求方法: %s", r.Method)
		s.writeError(w, http.StatusMethodNotAllowed, "请求方法不允许")
		return
	}

	if err := s.authenticateRequest(r); err != nil {
		s.Logger().Warn("来自 %s 的请求鉴权失败: %v", r.RemoteAddr, err)
		s.writeError(w, http.StatusUnauthorized, err.Error())
		return
	}

	var req gemini.ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.Logger().Error("解析请求体失败: %v", err)
		s.writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	req.Model = s.normalizeModel(req.Model)
	s.Logger().Info("对话请求: 模型=%s, 消息数=%d, 流式=%v", req.Model, len(req.Messages), req.Stream)
	if req.MaxCompletionTokens > 0 && req.MaxTokens == 0 {
		req.MaxTokens = req.MaxCompletionTokens
	}
	s.Logger().Debug("请求消息内容: %+v", req.Messages)

	sessionKey := r.Header.Get("X-Session-ID")
	if sessionKey == "" {
		sessionKey = fmt.Sprintf("default-%s", r.RemoteAddr)
	}
	if req.ConversationID != "" {
		sessionKey = req.ConversationID
	}

	session, isNewSession := s.getOrCreateSession(sessionKey, req.ConversationID)
	sessionSnapshot := session.Snapshot()

	if req.ConversationID != "" {
		if sessionSnapshot.ConversationID != req.ConversationID {
			session.SetConversationID(req.ConversationID)
		}
		if isNewSession {
			s.Logger().Debug("根据 conversation_id 创建了新会话: %s", req.ConversationID)
			isNewSession = false
		} else {
			s.Logger().Debug("正在恢复对话: %s", req.ConversationID)
		}
	} else if isNewSession {
		s.Logger().Debug("创建了新会话: %s", sessionKey)
	} else {
		s.Logger().Debug("正在使用现有会话: %s (c=%s)", sessionKey, sessionSnapshot.ConversationID)
	}

	snlm0eToken, _ := s.tokenManager.GetTokenForSession(sessionKey, isNewSession)
	prompt, images := buildPromptWithMedia(req)

	if req.Stream {
		s.Logger().Debug("开始流式响应")
		s.recordDiscoveredModel(req.Model)
		handleStreamResponse(w, prompt, images, req.Model, session, req.Tools, sessionKey, snlm0eToken, req.StreamOptions, s.writeError, s.writeMappedError)
		s.savePersistentState()
		return
	}

	s.Logger().Debug("开始非流式响应")
	s.recordDiscoveredModel(req.Model)
	handleNonStreamResponse(w, prompt, images, req.Model, session, req.Tools, sessionKey, snlm0eToken, s.writeError, s.writeMappedError, s.writeJSON)
	s.savePersistentState()
}
