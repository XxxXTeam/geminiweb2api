package token

import (
	"fmt"
	"io"
	"math"
	"net/http"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"main/internal/config"
	"main/internal/httpclient"
	"main/internal/logging"
)

const defaultAccountID = "__default__"

type TokenInfo struct {
	SNlM0e    string
	BLToken   string
	FSID      string
	ReqID     int64
	FetchedAt time.Time
	mutex     sync.RWMutex
}

type AnonToken struct {
	SNlM0e    string
	FetchedAt time.Time
	IsBad     bool
}

type pageState struct {
	RequestToken string
	BLToken      string
	FSID         string
}

type AccountStatus struct {
	ID                  string         `json:"id"`
	Email               string         `json:"email"`
	Enabled             bool           `json:"enabled"`
	Weight              int            `json:"weight"`
	StateCode           string         `json:"state_code"`
	StateLabel          string         `json:"state_label"`
	ActionRequired      string         `json:"action_required,omitempty"`
	Retryable           bool           `json:"retryable"`
	NextRetryAt         time.Time      `json:"next_retry_at,omitempty"`
	TokenReady          bool           `json:"token_ready"`
	HasProxy            bool           `json:"has_proxy"`
	UsingCookies        bool           `json:"using_cookies"`
	HasManualToken      bool           `json:"has_manual_token"`
	BoundSessions       int            `json:"bound_sessions"`
	ConsecutiveFailures int            `json:"consecutive_failures"`
	BackoffUntil        time.Time      `json:"backoff_until,omitempty"`
	LastUsedAt          time.Time      `json:"last_used_at,omitempty"`
	LastError           string         `json:"last_error,omitempty"`
	LastTokenRefreshAt  time.Time      `json:"last_token_refresh_at,omitempty"`
	RecentFailures      []FailureEvent `json:"recent_failures,omitempty"`
}

type FailureEvent struct {
	At        time.Time `json:"at"`
	Code      string    `json:"code"`
	Label     string    `json:"label"`
	Reason    string    `json:"reason"`
	Action    string    `json:"action,omitempty"`
	Retryable bool      `json:"retryable"`
}

type accountState struct {
	Code           string
	Label          string
	ActionRequired string
	Retryable      bool
	NextRetryAt    time.Time
}

type PoolStats struct {
	TotalAccounts    int `json:"total_accounts"`
	EnabledAccounts  int `json:"enabled_accounts"`
	HealthyAccounts  int `json:"healthy_accounts"`
	BackoffAccounts  int `json:"backoff_accounts"`
	NotReadyAccounts int `json:"not_ready_accounts"`
	DisabledAccounts int `json:"disabled_accounts"`
	BoundSessions    int `json:"bound_sessions"`
}

type SessionBinding struct {
	SessionKey string    `json:"session_key"`
	AccountID  string    `json:"account_id"`
	BoundAt    time.Time `json:"bound_at"`
	LastUsedAt time.Time `json:"last_used_at"`
}

type AccountTokenSnapshot struct {
	SNlM0e    string    `json:"snlm0e"`
	BLToken   string    `json:"bl_token"`
	FSID      string    `json:"fsid"`
	ReqID     int64     `json:"req_id"`
	FetchedAt time.Time `json:"fetched_at"`
}

type CookieHealth struct {
	AccountID            string           `json:"account_id"`
	CookieCount          int              `json:"cookie_count"`
	ImportantMissing     []string         `json:"important_missing"`
	ImportantPresent     map[string]bool  `json:"important_present"`
	AbuseExemption       CookieTimeHint   `json:"abuse_exemption"`
	AnalyticsTimeHints   []CookieTimeHint `json:"analytics_time_hints,omitempty"`
	OpaqueSessionCookies []string         `json:"opaque_session_cookies,omitempty"`
	StateCode            string           `json:"state_code"`
	StateLabel           string           `json:"state_label"`
	TokenReady           bool             `json:"token_ready"`
	LastError            string           `json:"last_error,omitempty"`
}

type CookieTimeHint struct {
	Name      string    `json:"name"`
	Source    string    `json:"source"`
	Epoch     int64     `json:"epoch,omitempty"`
	Time      time.Time `json:"time,omitempty"`
	AgeSec    int64     `json:"age_sec,omitempty"`
	ValueSeen bool      `json:"value_seen"`
}

type SelectedAccount struct {
	ID           string
	Email        string
	Cookies      string
	Proxy        string
	Token        string
	BLToken      string
	FSID         string
	ReqID        string
	TokenFetched bool
}

type sessionBinding struct {
	AccountID  string
	BoundAt    time.Time
	LastUsedAt time.Time
}

type accountRuntime struct {
	cfg                 config.Account
	tokenInfo           *TokenInfo
	sessionTokens       map[string]*AnonToken
	consecutiveFailures int
	backoffUntil        time.Time
	lastUsedAt          time.Time
	lastError           string
	recentFailures      []FailureEvent
}

type Manager struct {
	getConfig    func() config.Config
	getClient    func() *http.Client
	getLogger    func() *logging.Logger
	updateConfig func(func(*config.Config) error) error

	mu             sync.RWMutex
	accounts       map[string]*accountRuntime
	sessionBinding map[string]*sessionBinding
	roundRobin     uint64
	clientMu       sync.Mutex
	proxyClients   map[string]*http.Client
}

func NewManager(getConfig func() config.Config, getClient func() *http.Client, getLogger func() *logging.Logger, updateConfig func(func(*config.Config) error) error) *Manager {
	m := &Manager{
		getConfig:      getConfig,
		getClient:      getClient,
		getLogger:      getLogger,
		updateConfig:   updateConfig,
		accounts:       make(map[string]*accountRuntime),
		sessionBinding: make(map[string]*sessionBinding),
		proxyClients:   make(map[string]*http.Client),
	}
	m.reloadAccountsLocked()
	return m
}

func (m *Manager) clientForProxy(proxyValue string) *http.Client {
	proxyValue = strings.TrimSpace(proxyValue)
	if proxyValue == "" {
		return m.getClient()
	}
	m.clientMu.Lock()
	defer m.clientMu.Unlock()
	if client := m.proxyClients[proxyValue]; client != nil {
		return client
	}
	client, _, _ := httpclient.NewWithProxy(m.getConfig(), proxyValue, m.getLogger())
	m.proxyClients[proxyValue] = client
	return client
}

func (m *Manager) StartRefresher() {
	m.RefreshAccountsFromConfig()
	m.refreshAllAccountsIfNeeded(true)

	go func() {
		ticker := time.NewTicker(25 * time.Minute)
		defer ticker.Stop()
		for range ticker.C {
			m.RefreshTokenIfNeeded()
		}
	}()
}

func (m *Manager) RefreshAccountsFromConfig() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.reloadAccountsLocked()
}

func (m *Manager) RefreshTokenIfNeeded() {
	m.RefreshAccountsFromConfig()
	m.refreshAllAccountsIfNeeded(false)
}

func (m *Manager) RefreshTokenNow() error {
	m.RefreshAccountsFromConfig()
	ids := m.accountIDs()
	var errs []string
	for _, id := range ids {
		if err := m.RefreshAccountNow(id); err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", id, err))
		}
	}
	if len(errs) > 0 && len(errs) == len(ids) {
		return fmt.Errorf("%s", strings.Join(errs, "; "))
	}
	return nil
}

func (m *Manager) RefreshAccountNow(accountID string) error {
	m.RefreshAccountsFromConfig()
	m.mu.Lock()
	acc, ok := m.accounts[accountID]
	if !ok {
		m.mu.Unlock()
		return fmt.Errorf("account not found: %s", accountID)
	}
	cfg := acc.cfg
	m.mu.Unlock()

	if strings.TrimSpace(cfg.Cookies) == "" {
		return nil
	}
	if err := m.fetchToken(accountID); err != nil {
		m.mu.Lock()
		if acc := m.accounts[accountID]; acc != nil {
			acc.tokenInfo.mutex.RLock()
			hasUsableToken := strings.TrimSpace(acc.cfg.Token) != "" || strings.TrimSpace(acc.tokenInfo.SNlM0e) != ""
			acc.tokenInfo.mutex.RUnlock()
			acc.lastError = err.Error()
			if hasUsableToken {
				acc.backoffUntil = time.Time{}
				acc.consecutiveFailures = 0
			}
		}
		m.mu.Unlock()
		return err
	}

	m.mu.Lock()
	if acc := m.accounts[accountID]; acc != nil {
		acc.sessionTokens = make(map[string]*AnonToken)
		acc.lastError = ""
		acc.backoffUntil = time.Time{}
		acc.consecutiveFailures = 0
	}
	m.mu.Unlock()
	return nil
}

func (m *Manager) GetTokenForSession(sessionKey string, isNewSession bool) (string, int) {
	selected, err := m.SelectAccountForSession(sessionKey, isNewSession)
	if err != nil {
		m.getLogger().Warn("为会话 %s 选择账号失败: %v", sessionKey, err)
		return "", 0
	}
	return selected.Token, 0
}

func (m *Manager) SelectAccountForSession(sessionKey string, isNewSession bool) (SelectedAccount, error) {
	m.RefreshAccountsFromConfig()
	accountID, err := m.pickAccountID(sessionKey, isNewSession)
	if err != nil {
		return SelectedAccount{}, err
	}
	return m.GetSelectedAccount(accountID, sessionKey, isNewSession)
}

func (m *Manager) GetSelectedAccount(accountID, sessionKey string, isNewSession bool) (SelectedAccount, error) {
	m.RefreshAccountsFromConfig()
	m.mu.RLock()
	acc, ok := m.accounts[accountID]
	m.mu.RUnlock()
	if !ok {
		return SelectedAccount{}, fmt.Errorf("account not found: %s", accountID)
	}

	if err := m.ensureAccountReady(accountID); err != nil {
		return SelectedAccount{}, err
	}

	m.mu.Lock()
	acc = m.accounts[accountID]
	if sessionKey != "" {
		m.bindSessionLocked(sessionKey, accountID)
		if token, exists := acc.sessionTokens[sessionKey]; exists && !isNewSession && !token.IsBad && time.Since(token.FetchedAt) < 25*time.Minute {
			acc.lastUsedAt = time.Now()
			binding := m.sessionBinding[sessionKey]
			binding.LastUsedAt = time.Now()
			m.mu.Unlock()
			return m.snapshotSelectedAccount(accountID, token.SNlM0e), nil
		}
	}
	m.mu.Unlock()

	tokenValue := ""
	if sessionKey != "" {
		if t, err := m.FetchAnonymousTokenForAccount(accountID); err == nil && t != "" {
			m.mu.Lock()
			if acc = m.accounts[accountID]; acc != nil {
				acc.sessionTokens[sessionKey] = &AnonToken{SNlM0e: t, FetchedAt: time.Now()}
				acc.lastUsedAt = time.Now()
				if binding := m.sessionBinding[sessionKey]; binding != nil {
					binding.LastUsedAt = time.Now()
				}
			}
			m.mu.Unlock()
			tokenValue = t
		}
	}
	return m.snapshotSelectedAccount(accountID, tokenValue), nil
}

func (m *Manager) MarkSessionTokenBad(sessionKey string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	binding := m.sessionBinding[sessionKey]
	if binding == nil {
		return
	}
	if acc, exists := m.accounts[binding.AccountID]; exists {
		if token, ok := acc.sessionTokens[sessionKey]; ok {
			token.IsBad = true
		}
		m.recordFailureLocked(acc, "session token marked bad")
		delete(m.sessionBinding, sessionKey)
		m.getLogger().Warn("会话 %s 的账号 %s 已标记为失效并解除绑定", sessionKey, binding.AccountID)
	}
}

func (m *Manager) MarkAccountSuccess(accountID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	acc := m.accounts[accountID]
	if acc == nil {
		return
	}
	acc.consecutiveFailures = 0
	acc.backoffUntil = time.Time{}
	acc.lastError = ""
	acc.lastUsedAt = time.Now()
}

func (m *Manager) MarkAccountFailure(accountID string, reason string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	acc := m.accounts[accountID]
	if acc == nil {
		return
	}
	m.recordFailureLocked(acc, reason)
	for sessionKey, binding := range m.sessionBinding {
		if binding.AccountID == accountID {
			delete(m.sessionBinding, sessionKey)
		}
	}
	acc.sessionTokens = make(map[string]*AnonToken)
}

func (m *Manager) GetToken() string {
	selected, err := m.SelectAccountForSession("", false)
	if err != nil {
		return ""
	}
	return selected.Token
}

func (m *Manager) GetBLToken() string {
	selected, err := m.SelectAccountForSession("", false)
	if err != nil {
		return ""
	}
	return selected.BLToken
}

func (m *Manager) GetFSID() string {
	selected, err := m.SelectAccountForSession("", false)
	if err != nil {
		return ""
	}
	return selected.FSID
}

func (m *Manager) NextReqID() string {
	selected, err := m.SelectAccountForSession("", false)
	if err != nil {
		return strconv.FormatInt(seedReqID(), 10)
	}
	return selected.ReqID
}

func (m *Manager) AccountsStatus() []AccountStatus {
	m.RefreshAccountsFromConfig()
	m.mu.RLock()
	defer m.mu.RUnlock()
	statuses := make([]AccountStatus, 0, len(m.accounts))
	boundCounts := make(map[string]int)
	for _, binding := range m.sessionBinding {
		boundCounts[binding.AccountID]++
	}
	for _, id := range sortedAccountIDs(m.accounts) {
		acc := m.accounts[id]
		acc.tokenInfo.mutex.RLock()
		tokenReady := strings.TrimSpace(acc.cfg.Token) != "" || strings.TrimSpace(acc.tokenInfo.SNlM0e) != ""
		lastTokenRefreshAt := acc.tokenInfo.FetchedAt
		status := AccountStatus{
			ID:                  acc.cfg.ID,
			Email:               acc.cfg.Email,
			Enabled:             acc.cfg.Enabled,
			Weight:              normalizedWeight(acc.cfg.Weight),
			TokenReady:          tokenReady,
			HasProxy:            strings.TrimSpace(acc.cfg.Proxy) != "",
			UsingCookies:        strings.TrimSpace(acc.cfg.Cookies) != "",
			HasManualToken:      strings.TrimSpace(acc.cfg.Token) != "",
			BoundSessions:       boundCounts[id],
			ConsecutiveFailures: acc.consecutiveFailures,
			BackoffUntil:        acc.backoffUntil,
			LastUsedAt:          acc.lastUsedAt,
			LastError:           acc.lastError,
			LastTokenRefreshAt:  lastTokenRefreshAt,
			RecentFailures:      append([]FailureEvent(nil), acc.recentFailures...),
		}
		acc.tokenInfo.mutex.RUnlock()
		state := classifyAccountState(acc, tokenReady)
		status.StateCode = state.Code
		status.StateLabel = state.Label
		status.ActionRequired = state.ActionRequired
		status.Retryable = state.Retryable
		status.NextRetryAt = state.NextRetryAt
		statuses = append(statuses, status)
	}
	return statuses
}

func (m *Manager) SessionBindings() []SessionBinding {
	m.mu.RLock()
	defer m.mu.RUnlock()
	bindings := make([]SessionBinding, 0, len(m.sessionBinding))
	for sessionKey, binding := range m.sessionBinding {
		bindings = append(bindings, SessionBinding{
			SessionKey: sessionKey,
			AccountID:  binding.AccountID,
			BoundAt:    binding.BoundAt,
			LastUsedAt: binding.LastUsedAt,
		})
	}
	sort.Slice(bindings, func(i, j int) bool {
		return bindings[i].SessionKey < bindings[j].SessionKey
	})
	return bindings
}

func (m *Manager) CookieHealth(accountID string) (CookieHealth, bool) {
	m.mu.RLock()
	acc := m.accounts[accountID]
	m.mu.RUnlock()
	if acc == nil {
		return CookieHealth{}, false
	}

	cookies := parseCookiePairs(acc.cfg.Cookies)
	important := []string{"COMPASS", "GOOGLE_ABUSE_EXEMPTION", "SID", "__Secure-1PSID", "__Secure-3PSID", "SAPISID", "__Secure-1PAPISID", "__Secure-3PAPISID", "SIDCC", "__Secure-1PSIDCC", "__Secure-3PSIDCC", "__Secure-1PSIDTS", "__Secure-3PSIDTS"}
	present := make(map[string]bool, len(important))
	missing := make([]string, 0)
	for _, key := range important {
		_, ok := cookies[key]
		present[key] = ok
		if !ok {
			missing = append(missing, key)
		}
	}

	statuses := m.AccountsStatus()
	var status AccountStatus
	for _, candidate := range statuses {
		if candidate.ID == accountID {
			status = candidate
			break
		}
	}

	health := CookieHealth{
		AccountID:            accountID,
		CookieCount:          len(cookies),
		ImportantMissing:     missing,
		ImportantPresent:     present,
		AbuseExemption:       cookieTimeHint("GOOGLE_ABUSE_EXEMPTION", cookies["GOOGLE_ABUSE_EXEMPTION"], `(?:^|:)TM=(\d{10})(?:[:;]|$)`, "TM"),
		AnalyticsTimeHints:   analyticsCookieTimeHints(cookies),
		OpaqueSessionCookies: opaqueSessionCookies(cookies),
		StateCode:            status.StateCode,
		StateLabel:           status.StateLabel,
		TokenReady:           status.TokenReady,
		LastError:            status.LastError,
	}
	return health, true
}

func parseCookiePairs(raw string) map[string]string {
	cookies := map[string]string{}
	for _, part := range strings.Split(raw, ";") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		key, value, ok := strings.Cut(part, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		if key == "" {
			continue
		}
		cookies[key] = strings.TrimSpace(value)
	}
	return cookies
}

func cookieTimeHint(name string, value string, pattern string, source string) CookieTimeHint {
	hint := CookieTimeHint{Name: name, Source: source, ValueSeen: strings.TrimSpace(value) != ""}
	if value == "" {
		return hint
	}
	matches := regexp.MustCompile(pattern).FindStringSubmatch(value)
	if len(matches) < 2 {
		return hint
	}
	epoch, err := strconv.ParseInt(matches[1], 10, 64)
	if err != nil {
		return hint
	}
	hint.Epoch = epoch
	hint.Time = time.Unix(epoch, 0).UTC()
	hint.AgeSec = int64(time.Since(hint.Time).Seconds())
	return hint
}

func analyticsCookieTimeHints(cookies map[string]string) []CookieTimeHint {
	hints := make([]CookieTimeHint, 0)
	for key, value := range cookies {
		if strings.HasPrefix(key, "_ga_") {
			hints = append(hints, cookieTimeHint(key, value, `\$t(\d{10})`, "$t"))
		}
	}
	slices.SortFunc(hints, func(a, b CookieTimeHint) int { return strings.Compare(a.Name, b.Name) })
	return hints
}

func opaqueSessionCookies(cookies map[string]string) []string {
	keys := make([]string, 0)
	for _, key := range []string{"__Secure-1PSIDTS", "__Secure-3PSIDTS"} {
		if strings.HasPrefix(cookies[key], "sidts-") {
			keys = append(keys, key)
		}
	}
	return keys
}

func (m *Manager) TokenSnapshots() map[string]AccountTokenSnapshot {
	m.mu.RLock()
	defer m.mu.RUnlock()
	snapshots := make(map[string]AccountTokenSnapshot, len(m.accounts))
	for _, id := range sortedAccountIDs(m.accounts) {
		acc := m.accounts[id]
		if acc == nil {
			continue
		}
		acc.tokenInfo.mutex.RLock()
		snapshot := AccountTokenSnapshot{
			SNlM0e:    acc.tokenInfo.SNlM0e,
			BLToken:   acc.tokenInfo.BLToken,
			FSID:      acc.tokenInfo.FSID,
			ReqID:     acc.tokenInfo.ReqID,
			FetchedAt: acc.tokenInfo.FetchedAt,
		}
		acc.tokenInfo.mutex.RUnlock()
		if strings.TrimSpace(snapshot.SNlM0e) == "" && strings.TrimSpace(snapshot.BLToken) == "" && strings.TrimSpace(snapshot.FSID) == "" {
			continue
		}
		snapshots[id] = snapshot
	}
	return snapshots
}

func (m *Manager) RestoreTokenSnapshots(snapshots map[string]AccountTokenSnapshot) {
	if len(snapshots) == 0 {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	for accountID, snapshot := range snapshots {
		acc := m.accounts[accountID]
		if acc == nil {
			continue
		}
		acc.tokenInfo.mutex.Lock()
		acc.tokenInfo.SNlM0e = strings.TrimSpace(snapshot.SNlM0e)
		acc.tokenInfo.BLToken = strings.TrimSpace(snapshot.BLToken)
		acc.tokenInfo.FSID = strings.TrimSpace(snapshot.FSID)
		acc.tokenInfo.ReqID = snapshot.ReqID
		acc.tokenInfo.FetchedAt = snapshot.FetchedAt
		acc.tokenInfo.mutex.Unlock()
	}
}

func (m *Manager) RestoreSessionBindings(bindings []SessionBinding) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, binding := range bindings {
		if _, exists := m.accounts[binding.AccountID]; !exists {
			continue
		}
		m.sessionBinding[binding.SessionKey] = &sessionBinding{
			AccountID:  binding.AccountID,
			BoundAt:    binding.BoundAt,
			LastUsedAt: binding.LastUsedAt,
		}
	}
}

func (m *Manager) PoolStats() PoolStats {
	m.RefreshAccountsFromConfig()
	m.mu.RLock()
	defer m.mu.RUnlock()
	stats := PoolStats{
		TotalAccounts: len(m.accounts),
		BoundSessions: len(m.sessionBinding),
	}
	now := time.Now()
	for _, id := range sortedAccountIDs(m.accounts) {
		acc := m.accounts[id]
		if acc == nil {
			continue
		}
		if acc.cfg.Enabled {
			stats.EnabledAccounts++
		} else {
			stats.DisabledAccounts++
		}
		acc.tokenInfo.mutex.RLock()
		tokenReady := strings.TrimSpace(acc.cfg.Token) != "" || strings.TrimSpace(acc.tokenInfo.SNlM0e) != ""
		acc.tokenInfo.mutex.RUnlock()
		if !acc.cfg.Enabled {
			continue
		}
		if !acc.backoffUntil.IsZero() && acc.backoffUntil.After(now) {
			stats.BackoffAccounts++
			continue
		}
		if !tokenReady {
			stats.NotReadyAccounts++
			continue
		}
		stats.HealthyAccounts++
	}
	return stats
}

func (m *Manager) UpsertAccount(account config.Account) error {
	account.ID = strings.TrimSpace(account.ID)
	if account.ID == "" {
		return fmt.Errorf("account id is required")
	}
	if normalizedWeight(account.Weight) != account.Weight {
		account.Weight = normalizedWeight(account.Weight)
	}
	return m.getConfigStoreUpdate(func(cfg *config.Config) error {
		for i := range cfg.Accounts {
			if cfg.Accounts[i].ID == account.ID {
				if strings.TrimSpace(account.Cookies) == "" {
					account.Cookies = cfg.Accounts[i].Cookies
				}
				if strings.TrimSpace(account.Token) == "" {
					account.Token = cfg.Accounts[i].Token
				}
				if strings.TrimSpace(account.Proxy) == "" {
					account.Proxy = cfg.Accounts[i].Proxy
				}
				cfg.Accounts[i] = account
				return nil
			}
		}
		cfg.Accounts = append(cfg.Accounts, account)
		return nil
	})
}

func (m *Manager) SetAccountEnabled(accountID string, enabled bool) error {
	return m.getConfigStoreUpdate(func(cfg *config.Config) error {
		for i := range cfg.Accounts {
			if cfg.Accounts[i].ID == accountID {
				cfg.Accounts[i].Enabled = enabled
				return nil
			}
		}
		return fmt.Errorf("account not found: %s", accountID)
	})
}

func (m *Manager) DeleteAccount(accountID string) error {
	if strings.TrimSpace(accountID) == "" {
		return fmt.Errorf("account id is required")
	}
	if accountID == defaultAccountID {
		return fmt.Errorf("default account cannot be deleted")
	}
	if err := m.getConfigStoreUpdate(func(cfg *config.Config) error {
		filtered := cfg.Accounts[:0]
		found := false
		for _, account := range cfg.Accounts {
			if account.ID == accountID {
				found = true
				continue
			}
			filtered = append(filtered, account)
		}
		if !found {
			return fmt.Errorf("account not found: %s", accountID)
		}
		cfg.Accounts = filtered
		return nil
	}); err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.accounts, accountID)
	for sessionKey, binding := range m.sessionBinding {
		if binding.AccountID == accountID {
			delete(m.sessionBinding, sessionKey)
		}
	}
	return nil
}

func (m *Manager) UnbindSession(sessionKey string) error {
	if strings.TrimSpace(sessionKey) == "" {
		return fmt.Errorf("session key is required")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	binding := m.sessionBinding[sessionKey]
	if binding == nil {
		return fmt.Errorf("session binding not found: %s", sessionKey)
	}
	if acc := m.accounts[binding.AccountID]; acc != nil {
		delete(acc.sessionTokens, sessionKey)
	}
	delete(m.sessionBinding, sessionKey)
	return nil
}

func (m *Manager) RebindSession(sessionKey, accountID string) error {
	if strings.TrimSpace(sessionKey) == "" {
		return fmt.Errorf("session key is required")
	}
	if strings.TrimSpace(accountID) == "" {
		return fmt.Errorf("account id is required")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	acc := m.accounts[accountID]
	if acc == nil {
		return fmt.Errorf("account not found: %s", accountID)
	}
	if !m.accountAvailableLocked(acc) {
		return fmt.Errorf("account is not available: %s", accountID)
	}
	if existing := m.sessionBinding[sessionKey]; existing != nil {
		if oldAcc := m.accounts[existing.AccountID]; oldAcc != nil {
			delete(oldAcc.sessionTokens, sessionKey)
		}
	}
	m.bindSessionLocked(sessionKey, accountID)
	delete(acc.sessionTokens, sessionKey)
	return nil
}

func (m *Manager) pickAccountID(sessionKey string, isNewSession bool) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if sessionKey != "" {
		if binding := m.sessionBinding[sessionKey]; binding != nil && !isNewSession {
			if acc := m.accounts[binding.AccountID]; acc != nil && m.accountAvailableLocked(acc) {
				binding.LastUsedAt = time.Now()
				return binding.AccountID, nil
			}
			delete(m.sessionBinding, sessionKey)
		}
	}

	candidates := m.availableAccountIDsLocked()
	if len(candidates) == 0 {
		return "", fmt.Errorf("no healthy accounts available")
	}
	idx := int(m.roundRobin % uint64(len(candidates)))
	m.roundRobin++
	accountID := candidates[idx]
	if sessionKey != "" {
		m.bindSessionLocked(sessionKey, accountID)
	}
	return accountID, nil
}

func (m *Manager) snapshotSelectedAccount(accountID string, sessionToken string) SelectedAccount {
	m.mu.RLock()
	acc := m.accounts[accountID]
	m.mu.RUnlock()
	selected := SelectedAccount{}
	if acc == nil {
		return selected
	}
	acc.tokenInfo.mutex.RLock()
	selected = SelectedAccount{
		ID:           acc.cfg.ID,
		Email:        acc.cfg.Email,
		Cookies:      acc.cfg.Cookies,
		Proxy:        acc.cfg.Proxy,
		Token:        firstNonEmpty(sessionToken, acc.tokenInfo.SNlM0e, acc.cfg.Token),
		BLToken:      acc.tokenInfo.BLToken,
		FSID:         acc.tokenInfo.FSID,
		TokenFetched: !acc.tokenInfo.FetchedAt.IsZero(),
	}
	acc.tokenInfo.mutex.RUnlock()
	acc.tokenInfo.mutex.Lock()
	selected.ReqID = nextReqIDLocked(acc.tokenInfo)
	acc.tokenInfo.mutex.Unlock()
	return selected
}

func (m *Manager) ensureAccountReady(accountID string) error {
	m.mu.RLock()
	acc := m.accounts[accountID]
	m.mu.RUnlock()
	if acc == nil {
		return fmt.Errorf("account not found: %s", accountID)
	}

	acc.tokenInfo.mutex.RLock()
	needRefresh := acc.tokenInfo.SNlM0e == "" || acc.tokenInfo.BLToken == "" || acc.tokenInfo.FSID == "" || time.Since(acc.tokenInfo.FetchedAt) > 30*time.Minute
	hasUsableToken := strings.TrimSpace(acc.cfg.Token) != "" || strings.TrimSpace(acc.tokenInfo.SNlM0e) != ""
	acc.tokenInfo.mutex.RUnlock()
	if needRefresh && strings.TrimSpace(acc.cfg.Cookies) != "" {
		if err := m.fetchToken(accountID); err != nil {
			m.mu.Lock()
			if acc := m.accounts[accountID]; acc != nil {
				acc.lastError = err.Error()
				if hasUsableToken {
					acc.backoffUntil = time.Time{}
					acc.consecutiveFailures = 0
				} else {
					m.recordFailureLocked(acc, err.Error())
				}
			}
			m.mu.Unlock()
			if hasUsableToken {
				return nil
			}
			return err
		}
	}
	return nil
}

func (m *Manager) FetchAnonymousTokenForAccount(accountID string) (string, error) {
	m.mu.RLock()
	acc := m.accounts[accountID]
	m.mu.RUnlock()
	if acc == nil {
		return "", fmt.Errorf("account not found: %s", accountID)
	}

	endpoints := httpclient.CurrentGeminiEndpoints(m.getConfig())
	req, err := http.NewRequest("GET", endpoints.Home, nil)
	if err != nil {
		return "", fmt.Errorf("create request failed: %w", err)
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8")
	req.Header.Set("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
	if strings.TrimSpace(acc.cfg.Cookies) != "" {
		req.Header.Set("Cookie", acc.cfg.Cookies)
	}
	resp, err := m.clientForProxy(acc.cfg.Proxy).Do(req)
	if err != nil {
		return "", fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("unexpected status: %d", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read body failed: %w", err)
	}
	state := extractPageState(body)
	m.updateTokenInfo(accountID, state)
	if state.RequestToken == "" {
		return "", missingRequestTokenError(body)
	}
	m.MarkAccountSuccess(accountID)
	return state.RequestToken, nil
}

func (m *Manager) fetchToken(accountID string) error {
	m.mu.RLock()
	acc := m.accounts[accountID]
	m.mu.RUnlock()
	if acc == nil {
		return fmt.Errorf("account not found: %s", accountID)
	}
	if strings.TrimSpace(acc.cfg.Cookies) == "" {
		if strings.TrimSpace(acc.cfg.Token) != "" {
			m.updateTokenInfo(accountID, pageState{RequestToken: acc.cfg.Token})
			return nil
		}
		return nil
	}

	endpoints := httpclient.CurrentGeminiEndpoints(m.getConfig())
	req, err := http.NewRequest("GET", endpoints.Home, nil)
	if err != nil {
		return fmt.Errorf("create request failed: %w", err)
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")
	req.Header.Set("Cookie", acc.cfg.Cookies)
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8")
	req.Header.Set("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
	resp, err := m.clientForProxy(acc.cfg.Proxy).Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status: %d", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read body failed: %w", err)
	}
	state := extractPageState(body)
	m.updateTokenInfo(accountID, state)
	if state.RequestToken != "" {
		m.MarkAccountSuccess(accountID)
		m.getLogger().Info("账号 %s 页面态获取成功: token长度=%d, BL=%s, f.sid=%s", accountID, len(state.RequestToken), state.BLToken, state.FSID)
		return nil
	}
	if strings.TrimSpace(acc.cfg.Token) != "" {
		m.updateTokenInfo(accountID, pageState{RequestToken: acc.cfg.Token})
		m.MarkAccountSuccess(accountID)
		m.getLogger().Info("账号 %s 正在使用配置文件中的令牌", accountID)
		return nil
	}
	return missingRequestTokenError(body)
}

func missingRequestTokenError(body []byte) error {
	bodyText := strings.ToLower(string(body))
	state := extractPageState(body)
	if state.BLToken != "" || state.FSID != "" {
		return fmt.Errorf("request token not found in Gemini app page")
	}
	if strings.Contains(bodyText, "before you continue") || strings.Contains(bodyText, "使用前须知") || strings.Contains(bodyText, "accounts.google") || strings.Contains(bodyText, "sign in") || strings.Contains(bodyText, "登录") {
		return fmt.Errorf("Gemini returned login/consent page; open gemini.google.com in the same browser, accept prompts, then copy the full Cookie again")
	}
	if strings.Contains(bodyText, "captcha") || strings.Contains(bodyText, "unusual traffic") || strings.Contains(bodyText, "sorry/index") {
		return fmt.Errorf("Gemini returned anti-abuse challenge; verify the browser session and proxy before copying Cookie again")
	}
	return fmt.Errorf("request token not found in page")
}

func classifyAccountState(acc *accountRuntime, tokenReady bool) accountState {
	now := time.Now()
	if acc == nil {
		return accountState{Code: "missing", Label: "账号不存在", ActionRequired: "检查账号池配置", Retryable: false}
	}
	if !acc.cfg.Enabled {
		return accountState{Code: "disabled", Label: "已禁用", ActionRequired: "启用账号后再参与调度", Retryable: false}
	}
	if !acc.backoffUntil.IsZero() && acc.backoffUntil.After(now) {
		failure := classifyFailure(acc.lastError)
		return accountState{Code: "backoff", Label: "避退中", ActionRequired: failure.Action, Retryable: failure.Retryable, NextRetryAt: acc.backoffUntil}
	}
	if tokenReady {
		return accountState{Code: "ready", Label: "健康", Retryable: true}
	}
	if strings.TrimSpace(acc.cfg.Cookies) == "" && strings.TrimSpace(acc.cfg.Token) == "" {
		return accountState{Code: "empty_credentials", Label: "无登录态", ActionRequired: "导入 Cookie 或手动 Token", Retryable: false}
	}
	if acc.lastError != "" {
		failure := classifyFailure(acc.lastError)
		return accountState{Code: failure.Code, Label: failure.Label, ActionRequired: failure.Action, Retryable: failure.Retryable}
	}
	return accountState{Code: "not_ready", Label: "未就绪", ActionRequired: "点击刷新验证登录态", Retryable: true}
}

func classifyFailure(reason string) FailureEvent {
	lower := strings.ToLower(reason)
	event := FailureEvent{Code: "unknown_error", Label: "未知错误", Reason: reason, Action: "查看日志并重试"}
	switch {
	case reason == "":
		event.Code = "none"
		event.Label = "无错误"
		event.Reason = ""
		event.Action = ""
	case strings.Contains(lower, "login/consent") || strings.Contains(lower, "使用前须知") || strings.Contains(lower, "sign in") || strings.Contains(lower, "accounts.google"):
		event.Code = "login_consent_required"
		event.Label = "需要登录/同意"
		event.Reason = reason
		event.Action = "在对应浏览器打开 gemini.google.com，完成登录/同意后重新抓 Session"
	case strings.Contains(lower, "anti-abuse") || strings.Contains(lower, "captcha") || strings.Contains(lower, "unusual traffic") || strings.Contains(lower, "sorry/index"):
		event.Code = "anti_abuse_challenge"
		event.Label = "风控验证"
		event.Action = "检查代理和浏览器风控状态，通过验证后重新抓 Session"
	case strings.Contains(lower, "request token not found") || strings.Contains(lower, "snlm0e"):
		event.Code = "request_token_missing"
		event.Label = "请求 Token 缺失"
		event.Action = "Cookie 不完整或已过期，重新抓完整 Cookie"
	case strings.Contains(lower, "unexpected status: 401") || strings.Contains(lower, "unauthorized"):
		event.Code = "unauthorized"
		event.Label = "未授权"
		event.Action = "登录态失效，重新抓 Session"
	case strings.Contains(lower, "unexpected status: 403") || strings.Contains(lower, "forbidden"):
		event.Code = "forbidden"
		event.Label = "账号受限"
		event.Action = "账号或地区受限，检查浏览器页面状态"
	case strings.Contains(lower, "unexpected status: 429") || strings.Contains(lower, "rate") || strings.Contains(lower, "quota"):
		event.Code = "rate_limited"
		event.Label = "限流"
		event.Action = "等待冷却或切换账号"
	case strings.Contains(lower, "request failed") || strings.Contains(lower, "timeout") || strings.Contains(lower, "connect") || strings.Contains(lower, "connection"):
		event.Code = "network_error"
		event.Label = "网络错误"
		event.Action = "检查网络和代理后重试"
	}
	event.Retryable = event.Code == "rate_limited" || event.Code == "network_error" || event.Code == "unknown_error"
	return event
}

func extractPageState(body []byte) pageState {
	return pageState{
		RequestToken: firstMatch(body, []string{
			`"SNlM0e":"([^"]+)"`,
			`SNlM0e\\x22:\\x22([^\\]+)\\x22`,
			`WIZ_global_data[^}]*"SNlM0e":"([^"]+)"`,
			`SNlM0e\\":\\"([^\\]+)\\"`,
			`"SNlM0e"\s*:\s*"([^"]+)"`,
		}),
		BLToken: firstMatch(body, []string{
			`"cfb2h":"([^"]+)"`,
			`cfb2h\\x22:\\x22([^\\]+)\\x22`,
			`"cfb2h"\s*:\s*"([^"]+)"`,
			`cfb2h\\":\\"([^\\]+)\\"`,
		}),
		FSID: firstMatch(body, []string{
			`[?&]f\.sid=([-0-9]+)`,
			`"f\.sid"\s*:\s*"?(?:\\u003d)?([-0-9]+)"?`,
			`"f\.sid":"([-0-9]+)"`,
			`f\.sid\\":\\"([-0-9]+)\\"`,
			`"FdrFJe":"([-0-9]+)"`,
			`FdrFJe\\x22:\\x22([-0-9]+)\\x22`,
		}),
	}
}

func firstMatch(body []byte, patterns []string) string {
	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		matches := re.FindSubmatch(body)
		if len(matches) > 1 {
			value := string(matches[1])
			if strings.TrimSpace(value) != "" {
				return value
			}
		}
	}
	return ""
}

func (m *Manager) updateTokenInfo(accountID string, state pageState) {
	if state.RequestToken == "" && state.BLToken == "" && state.FSID == "" {
		return
	}
	m.mu.RLock()
	acc := m.accounts[accountID]
	m.mu.RUnlock()
	if acc == nil {
		return
	}
	acc.tokenInfo.mutex.Lock()
	defer acc.tokenInfo.mutex.Unlock()
	if state.RequestToken != "" {
		acc.tokenInfo.SNlM0e = state.RequestToken
	}
	if state.BLToken != "" {
		acc.tokenInfo.BLToken = state.BLToken
	}
	if state.FSID != "" {
		acc.tokenInfo.FSID = state.FSID
	}
	if acc.tokenInfo.ReqID == 0 {
		acc.tokenInfo.ReqID = seedReqID()
	}
	acc.tokenInfo.FetchedAt = time.Now()
}

func (m *Manager) refreshAllAccountsIfNeeded(force bool) {
	for _, id := range m.accountIDs() {
		m.mu.RLock()
		acc := m.accounts[id]
		m.mu.RUnlock()
		if acc == nil || strings.TrimSpace(acc.cfg.Cookies) == "" {
			continue
		}
		if !force {
			acc.tokenInfo.mutex.RLock()
			needRefresh := acc.tokenInfo.SNlM0e == "" || acc.tokenInfo.BLToken == "" || acc.tokenInfo.FSID == "" || time.Since(acc.tokenInfo.FetchedAt) > 30*time.Minute
			acc.tokenInfo.mutex.RUnlock()
			if !needRefresh {
				continue
			}
		}
		if err := m.fetchToken(id); err != nil {
			m.mu.Lock()
			if acc := m.accounts[id]; acc != nil {
				acc.tokenInfo.mutex.RLock()
				hasUsableToken := strings.TrimSpace(acc.cfg.Token) != "" || strings.TrimSpace(acc.tokenInfo.SNlM0e) != ""
				acc.tokenInfo.mutex.RUnlock()
				acc.lastError = err.Error()
				if !hasUsableToken {
					m.recordFailureLocked(acc, err.Error())
				} else {
					acc.backoffUntil = time.Time{}
					acc.consecutiveFailures = 0
				}
			}
			m.mu.Unlock()
			m.getLogger().Warn("账号 %s 自动刷新令牌失败: %v", id, err)
		}
	}
}

func (m *Manager) reloadAccountsLocked() {
	cfg := m.getConfig()
	oldAccounts := m.accounts
	accounts := configuredAccounts(cfg)
	newAccounts := make(map[string]*accountRuntime, len(accounts))
	for _, account := range accounts {
		runtime := oldAccounts[account.ID]
		if runtime == nil {
			runtime = &accountRuntime{tokenInfo: &TokenInfo{}, sessionTokens: make(map[string]*AnonToken)}
		}
		runtime.cfg = account
		if runtime.tokenInfo == nil {
			runtime.tokenInfo = &TokenInfo{}
		}
		if runtime.sessionTokens == nil {
			runtime.sessionTokens = make(map[string]*AnonToken)
		}
		newAccounts[account.ID] = runtime
	}
	m.accounts = newAccounts
	for sessionKey, binding := range m.sessionBinding {
		if _, exists := m.accounts[binding.AccountID]; !exists {
			delete(m.sessionBinding, sessionKey)
		}
	}
}

func configuredAccounts(cfg config.Config) []config.Account {
	if len(cfg.Accounts) == 0 {
		return []config.Account{{
			ID:      defaultAccountID,
			Email:   "default",
			Cookies: cfg.Cookies,
			Token:   cfg.Token,
			Enabled: true,
			Weight:  1,
		}}
	}
	accounts := make([]config.Account, 0, len(cfg.Accounts))
	for i, account := range cfg.Accounts {
		account.ID = strings.TrimSpace(account.ID)
		if account.ID == "" {
			account.ID = fmt.Sprintf("account-%d", i+1)
		}
		account.Weight = normalizedWeight(account.Weight)
		accounts = append(accounts, account)
	}
	return accounts
}

func (m *Manager) accountIDs() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return sortedAccountIDs(m.accounts)
}

func sortedAccountIDs(accounts map[string]*accountRuntime) []string {
	ids := make([]string, 0, len(accounts))
	for id := range accounts {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

func (m *Manager) availableAccountIDsLocked() []string {
	now := time.Now()
	weighted := make([]string, 0, len(m.accounts))
	for _, id := range sortedAccountIDs(m.accounts) {
		acc := m.accounts[id]
		if acc == nil || !acc.cfg.Enabled || (!acc.backoffUntil.IsZero() && acc.backoffUntil.After(now)) {
			continue
		}
		for i := 0; i < normalizedWeight(acc.cfg.Weight); i++ {
			weighted = append(weighted, id)
		}
	}
	return weighted
}

func (m *Manager) accountAvailableLocked(acc *accountRuntime) bool {
	if acc == nil || !acc.cfg.Enabled {
		return false
	}
	return acc.backoffUntil.IsZero() || !acc.backoffUntil.After(time.Now())
}

func (m *Manager) bindSessionLocked(sessionKey, accountID string) {
	now := time.Now()
	m.sessionBinding[sessionKey] = &sessionBinding{AccountID: accountID, BoundAt: now, LastUsedAt: now}
	if acc := m.accounts[accountID]; acc != nil {
		acc.lastUsedAt = now
	}
}

func (m *Manager) recordFailureLocked(acc *accountRuntime, reason string) {
	acc.consecutiveFailures++
	seconds := math.Min(1800, 30*math.Pow(2, float64(acc.consecutiveFailures-1)))
	acc.backoffUntil = time.Now().Add(time.Duration(seconds) * time.Second)
	acc.lastError = reason
	acc.lastUsedAt = time.Now()
	failure := classifyFailure(reason)
	failure.At = time.Now()
	acc.recentFailures = append([]FailureEvent{failure}, acc.recentFailures...)
	if len(acc.recentFailures) > 5 {
		acc.recentFailures = acc.recentFailures[:5]
	}
	if acc.cfg.ID != "" {
		m.getLogger().Warn("账号 %s 进入避退，失败次数=%d，恢复时间=%s，原因=%s", acc.cfg.ID, acc.consecutiveFailures, acc.backoffUntil.Format(time.RFC3339), reason)
	}
}

func nextReqIDLocked(info *TokenInfo) string {
	if info.ReqID == 0 {
		info.ReqID = seedReqID()
	}
	current := info.ReqID
	info.ReqID += 100000
	return strconv.FormatInt(current, 10)
}

func normalizedWeight(weight int) int {
	if weight <= 0 {
		return 1
	}
	return weight
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func (m *Manager) getConfigStoreUpdate(mutator func(*config.Config) error) error {
	if m.updateConfig == nil {
		return fmt.Errorf("config store updates not wired")
	}
	if err := m.updateConfig(mutator); err != nil {
		return err
	}
	m.RefreshAccountsFromConfig()
	return nil
}

func seedReqID() int64 {
	base := time.Now().UnixNano()%9000000 + 1000000
	if base%2 == 0 {
		base++
	}
	return base
}
