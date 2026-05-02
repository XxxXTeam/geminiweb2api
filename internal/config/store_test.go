package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadAppliesEnvironmentOverrides(t *testing.T) {
	t.Setenv("GEMINIWEB2API_API_KEY", "env-key")
	t.Setenv("GEMINIWEB2API_PROXY", "http://127.0.0.1:7890")
	t.Setenv("GEMINIWEB2API_PORT", "9090")
	t.Setenv("GEMINIWEB2API_LOG_LEVEL", "debug")
	t.Setenv("GEMINIWEB2API_PUBLIC_ACCOUNT_STATUS", "true")

	dir := t.TempDir()
	path := filepath.Join(dir, "config.json")
	if err := os.WriteFile(path, []byte(`{"api_key":"file-key","port":8080,"log_level":"info"}`), 0o600); err != nil {
		t.Fatal(err)
	}

	store := NewStore(path)
	if err := store.Load(); err != nil {
		t.Fatal(err)
	}
	cfg := store.Snapshot()
	if cfg.APIKey != "env-key" || cfg.Proxy != "http://127.0.0.1:7890" || cfg.Port != 9090 || cfg.LogLevel != "debug" || !cfg.PublicAccountStatus {
		t.Fatalf("environment overrides were not applied: %+v", cfg)
	}
}
