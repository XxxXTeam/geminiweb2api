package token

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"main/internal/config"
	"main/internal/logging"
)

func TestRefreshAccountNowKeepsExistingTokenOnRefreshFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = io.WriteString(w, `<html><body>sign in to continue</body></html>`)
	}))
	defer server.Close()

	cfg := config.Config{
		GeminiHomeURL: server.URL,
		Accounts: []config.Account{{
			ID:      "acc-1",
			Email:   "first@example.com",
			Cookies: "SID=test",
			Enabled: true,
			Weight:  1,
		}},
	}
	logger := logging.New(logging.LevelError, io.Discard, nil)
	m := NewManager(
		func() config.Config { return cfg },
		func() *http.Client { return server.Client() },
		func() *logging.Logger { return logger },
		nil,
	)

	m.mu.Lock()
	acc := m.accounts["acc-1"]
	acc.tokenInfo.SNlM0e = "existing-token"
	acc.tokenInfo.BLToken = "existing-bl"
	acc.tokenInfo.FSID = "12345"
	acc.tokenInfo.ReqID = 1001
	acc.tokenInfo.FetchedAt = time.Now()
	acc.lastError = "older transient error"
	acc.consecutiveFailures = 2
	acc.backoffUntil = time.Now().Add(5 * time.Minute)
	m.mu.Unlock()

	err := m.RefreshAccountNow("acc-1")
	if err == nil {
		t.Fatal("expected refresh error")
	}

	statuses := m.AccountsStatus()
	if len(statuses) != 1 {
		t.Fatalf("expected 1 account status, got %d", len(statuses))
	}
	status := statuses[0]
	if !status.TokenReady {
		t.Fatal("expected token_ready to remain true after refresh failure")
	}
	if status.StateCode != "ready" {
		t.Fatalf("expected state_code ready, got %q", status.StateCode)
	}
	if status.LastError == "" {
		t.Fatal("expected last_error to capture refresh failure")
	}
	if status.ConsecutiveFailures != 0 {
		t.Fatalf("expected consecutive failures to reset after preserving usable token, got %d", status.ConsecutiveFailures)
	}
	if !status.BackoffUntil.IsZero() {
		t.Fatal("expected backoff to clear after preserving usable token")
	}

	m.mu.RLock()
	refreshed := m.accounts["acc-1"]
	m.mu.RUnlock()
	refreshed.tokenInfo.mutex.RLock()
	defer refreshed.tokenInfo.mutex.RUnlock()
	if refreshed.tokenInfo.SNlM0e != "existing-token" || refreshed.tokenInfo.BLToken != "existing-bl" || refreshed.tokenInfo.FSID != "12345" {
		t.Fatal("expected existing token info to be preserved on refresh failure")
	}
}

func TestAutoRefreshKeepsExistingTokenOnRefreshFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = io.WriteString(w, `<html><body>captcha challenge</body></html>`)
	}))
	defer server.Close()

	cfg := config.Config{
		GeminiHomeURL: server.URL,
		Accounts: []config.Account{{
			ID:      "acc-1",
			Email:   "first@example.com",
			Cookies: "SID=test",
			Enabled: true,
			Weight:  1,
		}},
	}
	logger := logging.New(logging.LevelError, io.Discard, nil)
	m := NewManager(
		func() config.Config { return cfg },
		func() *http.Client { return server.Client() },
		func() *logging.Logger { return logger },
		nil,
	)

	m.mu.Lock()
	acc := m.accounts["acc-1"]
	acc.tokenInfo.SNlM0e = "existing-token"
	acc.tokenInfo.BLToken = "existing-bl"
	acc.tokenInfo.FSID = "12345"
	acc.tokenInfo.FetchedAt = time.Now().Add(-31 * time.Minute)
	m.mu.Unlock()

	m.refreshAllAccountsIfNeeded(true)

	statuses := m.AccountsStatus()
	if len(statuses) != 1 {
		t.Fatalf("expected 1 account status, got %d", len(statuses))
	}
	status := statuses[0]
	if !status.TokenReady {
		t.Fatal("expected token_ready to remain true after auto refresh failure")
	}
	if status.StateCode != "ready" {
		t.Fatalf("expected state_code ready, got %q", status.StateCode)
	}
	if status.LastError == "" {
		t.Fatal("expected last_error to capture auto refresh failure")
	}
	if status.ConsecutiveFailures != 0 {
		t.Fatalf("expected consecutive failures to remain cleared, got %d", status.ConsecutiveFailures)
	}
	if !status.BackoffUntil.IsZero() {
		t.Fatal("expected backoff to stay cleared after auto refresh failure with usable token")
	}
	if stats := m.PoolStats(); stats.HealthyAccounts != 1 {
		t.Fatalf("expected account to remain healthy, got stats %+v", stats)
	}
	if _, err := m.SelectAccountForSession("session-after-auto-refresh", false); err != nil {
		t.Fatalf("expected account to remain selectable after auto refresh failure: %v", err)
	}
	if refreshed := m.snapshotSelectedAccount("acc-1", ""); refreshed.Token != "existing-token" {
		t.Fatalf("expected existing token to be preserved, got %q", refreshed.Token)
	}
}

func TestCookieHealthReportsImportantCookiesAndTimeHints(t *testing.T) {
	cfg := config.Config{
		Accounts: []config.Account{{
			ID:      "acc-1",
			Cookies: "SID=sid; __Secure-1PSID=one; __Secure-3PSID=three; SAPISID=sapi; __Secure-1PAPISID=p1; __Secure-3PAPISID=p3; SIDCC=sidcc; __Secure-1PSIDCC=cc1; __Secure-3PSIDCC=cc3; __Secure-1PSIDTS=sidts-abc; __Secure-3PSIDTS=sidts-def; COMPASS=gemini-pd=abc; GOOGLE_ABUSE_EXEMPTION=ID=x:TM=1777537040:C=>:IP=1.2.3.4-:S=y; _ga_TEST=GS2.1.s1777536739$o1$g0$t1777536739$j60$l0$h0",
			Enabled: true,
			Weight:  1,
		}},
	}
	logger := logging.New(logging.LevelError, io.Discard, nil)
	m := NewManager(
		func() config.Config { return cfg },
		func() *http.Client { return http.DefaultClient },
		func() *logging.Logger { return logger },
		nil,
	)

	health, ok := m.CookieHealth("acc-1")
	if !ok {
		t.Fatal("expected account health")
	}
	if health.CookieCount != 14 {
		t.Fatalf("expected 14 cookies, got %d", health.CookieCount)
	}
	if len(health.ImportantMissing) != 0 {
		t.Fatalf("expected no missing important cookies, got %v", health.ImportantMissing)
	}
	if !health.ImportantPresent["COMPASS"] || !health.ImportantPresent["GOOGLE_ABUSE_EXEMPTION"] {
		t.Fatal("expected COMPASS and GOOGLE_ABUSE_EXEMPTION to be present")
	}
	if health.AbuseExemption.Epoch != 1777537040 {
		t.Fatalf("expected abuse exemption epoch, got %d", health.AbuseExemption.Epoch)
	}
	if len(health.AnalyticsTimeHints) != 1 || health.AnalyticsTimeHints[0].Epoch != 1777536739 {
		t.Fatalf("unexpected analytics hints: %+v", health.AnalyticsTimeHints)
	}
	if len(health.OpaqueSessionCookies) != 2 {
		t.Fatalf("expected opaque PSIDTS cookies, got %v", health.OpaqueSessionCookies)
	}
}

func TestSelectAccountForSessionUsesWeightedRoundRobin(t *testing.T) {
	cfg := config.Config{Accounts: []config.Account{
		{ID: "acc-1", Enabled: true, Weight: 2, Token: "token-1"},
		{ID: "acc-2", Enabled: true, Weight: 1, Token: "token-2"},
	}}
	m := newTestManager(cfg)

	var got []string
	for i := 0; i < 6; i++ {
		selected, err := m.SelectAccountForSession("", false)
		if err != nil {
			t.Fatalf("select account: %v", err)
		}
		got = append(got, selected.ID)
	}
	want := []string{"acc-1", "acc-1", "acc-2", "acc-1", "acc-1", "acc-2"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("unexpected weighted sequence: got %v want %v", got, want)
		}
	}
}

func TestSelectAccountForSessionKeepsExistingBinding(t *testing.T) {
	cfg := config.Config{Accounts: []config.Account{
		{ID: "acc-1", Enabled: true, Weight: 1, Token: "token-1"},
		{ID: "acc-2", Enabled: true, Weight: 1, Token: "token-2"},
	}}
	m := newTestManager(cfg)

	first, err := m.SelectAccountForSession("session-a", false)
	if err != nil {
		t.Fatalf("first select: %v", err)
	}
	second, err := m.SelectAccountForSession("session-a", false)
	if err != nil {
		t.Fatalf("second select: %v", err)
	}
	if second.ID != first.ID {
		t.Fatalf("expected sticky session account %q, got %q", first.ID, second.ID)
	}

	newSession, err := m.SelectAccountForSession("session-a", true)
	if err != nil {
		t.Fatalf("new session select: %v", err)
	}
	if newSession.ID == first.ID {
		t.Fatalf("expected new session to re-enter round robin, got same account %q", newSession.ID)
	}
}

func TestSelectAccountForSessionSkipsBackoffAccount(t *testing.T) {
	cfg := config.Config{Accounts: []config.Account{
		{ID: "acc-1", Enabled: true, Weight: 1, Token: "token-1"},
		{ID: "acc-2", Enabled: true, Weight: 1, Token: "token-2"},
	}}
	m := newTestManager(cfg)
	m.MarkAccountFailure("acc-1", "rate limited")

	for i := 0; i < 3; i++ {
		selected, err := m.SelectAccountForSession("", false)
		if err != nil {
			t.Fatalf("select account: %v", err)
		}
		if selected.ID != "acc-2" {
			t.Fatalf("expected backoff account to be skipped, got %q", selected.ID)
		}
	}
}

func newTestManager(cfg config.Config) *Manager {
	logger := logging.New(logging.LevelError, io.Discard, nil)
	return NewManager(
		func() config.Config { return cfg },
		func() *http.Client { return http.DefaultClient },
		func() *logging.Logger { return logger },
		nil,
	)
}
