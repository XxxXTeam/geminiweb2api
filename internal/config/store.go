package config

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"sync"
	"time"
)

const (
	DefaultPort     = 8080
	DefaultLogLevel = "info"
)

type Config struct {
	APIKey              string            `json:"api_key"`
	Token               string            `json:"token"`
	Cookies             string            `json:"cookies"`
	Tokens              []string          `json:"tokens"`
	Accounts            []Account         `json:"accounts"`
	Proxy               string            `json:"proxy"`
	Models              []string          `json:"models"`
	ModelAliases        map[string]string `json:"model_aliases"`
	GeminiURL           string            `json:"gemini_url"`
	GeminiHomeURL       string            `json:"gemini_home_url"`
	Port                int               `json:"port"`
	LogFile             string            `json:"log_file"`
	LogLevel            string            `json:"log_level"`
	PublicAccountStatus bool              `json:"public_account_status"`
	Note                []string          `json:"note"`
}

type Account struct {
	ID      string `json:"id"`
	Email   string `json:"email"`
	Cookies string `json:"cookies"`
	Token   string `json:"token"`
	Proxy   string `json:"proxy"`
	Enabled bool   `json:"enabled"`
	Weight  int    `json:"weight"`
}

type Store struct {
	path string

	mu  sync.RWMutex
	cfg Config
}

func NewStore(path string) *Store {
	return &Store{path: path}
}

func (s *Store) Path() string {
	return s.path
}

func (s *Store) Snapshot() Config {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.cfg
}

func (s *Store) Load() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, err := os.Stat(s.path); os.IsNotExist(err) {
		defaultConfig := Config{
			Port:     DefaultPort,
			LogLevel: DefaultLogLevel,
			APIKey:   "your-api-key-here",
			Proxy:    "",
			Note:     []string{"Auto-generated config"},
		}

		data, err := json.MarshalIndent(defaultConfig, "", "  ")
		if err != nil {
			return fmt.Errorf("failed to marshal default config: %w", err)
		}
		if err := os.WriteFile(s.path, data, 0644); err != nil {
			return fmt.Errorf("failed to write default config: %w", err)
		}
		applyEnvOverrides(&defaultConfig)
		s.cfg = defaultConfig
		return nil
	}

	file, err := os.Open(s.path)
	if err != nil {
		return err
	}
	defer file.Close()

	var cfg Config
	if err := json.NewDecoder(file).Decode(&cfg); err != nil {
		return err
	}

	if cfg.Port == 0 {
		cfg.Port = DefaultPort
	}
	if cfg.LogLevel == "" {
		cfg.LogLevel = DefaultLogLevel
	}

	applyEnvOverrides(&cfg)
	s.cfg = cfg
	return nil
}

func (s *Store) Reload() error {
	return s.Load()
}

func (s *Store) Update(mutator func(*Config) error) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	cfg := s.cfg
	if err := mutator(&cfg); err != nil {
		return err
	}

	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}
	if err := os.WriteFile(s.path, data, 0644); err != nil {
		return fmt.Errorf("failed to write config: %w", err)
	}

	applyEnvOverrides(&cfg)
	s.cfg = cfg
	return nil
}

func applyEnvOverrides(cfg *Config) {
	if value := os.Getenv("GEMINIWEB2API_API_KEY"); value != "" {
		cfg.APIKey = value
	}
	if value := os.Getenv("GEMINIWEB2API_PROXY"); value != "" {
		cfg.Proxy = value
	}
	if value := os.Getenv("GEMINIWEB2API_PORT"); value != "" {
		if port, err := strconv.Atoi(value); err == nil && port > 0 {
			cfg.Port = port
		}
	}
	if value := os.Getenv("GEMINIWEB2API_LOG_LEVEL"); value != "" {
		cfg.LogLevel = value
	}
	if value := os.Getenv("GEMINIWEB2API_PUBLIC_ACCOUNT_STATUS"); value != "" {
		if parsed, err := strconv.ParseBool(value); err == nil {
			cfg.PublicAccountStatus = parsed
		}
	}
}

func (s *Store) UpdateInMemory(mutator func(*Config) error) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	cfg := s.cfg
	if err := mutator(&cfg); err != nil {
		return err
	}

	s.cfg = cfg
	return nil
}

func (s *Store) Watch(onReload func() error) {
	go func() {
		var lastModTime time.Time
		for {
			time.Sleep(5 * time.Second)

			info, err := os.Stat(s.path)
			if err != nil {
				continue
			}

			modTime := info.ModTime()
			if !lastModTime.IsZero() && modTime.After(lastModTime) {
				if err := onReload(); err != nil {
					continue
				}
			}
			lastModTime = modTime
		}
	}()
}
