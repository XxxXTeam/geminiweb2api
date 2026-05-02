package httpclient

import (
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"main/internal/config"
	"main/internal/logging"
)

const (
	DefaultGeminiURL     = "https://gemini.google.com/_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate"
	DefaultGeminiHomeURL = "https://gemini.google.com/"
)

type GeminiEndpoints struct {
	URL     string
	Home    string
	Origin  string
	Referer string
}

func CurrentGeminiEndpoints(cfg config.Config) GeminiEndpoints {
	postURL := strings.TrimSpace(cfg.GeminiURL)
	if postURL == "" {
		postURL = DefaultGeminiURL
	}

	homeURL := strings.TrimSpace(cfg.GeminiHomeURL)
	if homeURL == "" {
		homeURL = DefaultGeminiHomeURL
	}

	origin := "https://gemini.google.com"
	referer := "https://gemini.google.com/"
	if u, err := url.Parse(homeURL); err == nil && u.Scheme != "" && u.Host != "" {
		origin = u.Scheme + "://" + u.Host
		referer = origin + "/"
	} else if u, err := url.Parse(postURL); err == nil && u.Scheme != "" && u.Host != "" {
		origin = u.Scheme + "://" + u.Host
		referer = origin + "/"
	}

	return GeminiEndpoints{
		URL:     postURL,
		Home:    homeURL,
		Origin:  origin,
		Referer: referer,
	}
}

func New(cfg config.Config, logger *logging.Logger) *http.Client {
	client, proxyConfigured, proxyValue := NewWithProxy(cfg, strings.TrimSpace(cfg.Proxy), logger)
	if proxyConfigured {
		go testProxyConnectivity(client, proxyValue, logger)
	} else {
		logger.Info("HTTP 客户端已初始化 (未配置显式代理)")
	}
	return client
}

func NewWithProxy(cfg config.Config, proxyOverride string, logger *logging.Logger) (*http.Client, bool, string) {
	dialer := &net.Dialer{
		Timeout:   10 * time.Second,
		KeepAlive: 30 * time.Second,
	}

	transport := &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		DialContext:           dialer.DialContext,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          100,
		IdleConnTimeout:       60 * time.Second,
		TLSHandshakeTimeout:   8 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		MaxIdleConnsPerHost:   10,
	}

	proxyConfigured := false
	proxyValue := strings.TrimSpace(proxyOverride)
	if proxyValue == "" {
		proxyValue = strings.TrimSpace(cfg.Proxy)
	}
	if proxyValue != "" {
		proxyURL, err := url.Parse(proxyValue)
		if err == nil {
			transport.Proxy = http.ProxyURL(proxyURL)
			proxyConfigured = true
		} else {
			logger.Warn("无效的代理 URL: %s，将回退到系统环境变量代理，错误: %v", proxyValue, err)
		}
	}

	client := &http.Client{
		Transport: transport,
		Timeout:   300 * time.Second,
	}

	return client, proxyConfigured, proxyValue
}

func testProxyConnectivity(client *http.Client, proxyStr string, logger *logging.Logger) {
	logger.Info("正在测试代理连通性: %s", proxyStr)

	req, _ := http.NewRequest("HEAD", "https://www.google.com", nil)
	req.Header.Set("User-Agent", "Mozilla/5.0")
	resp, err := client.Do(req)
	if err != nil {
		logger.Warn("代理检测失败 (可能是暂时的): %v。请求仍将尝试进行重试。", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 200 && resp.StatusCode < 400 {
		logger.Info("代理连通性验证成功: %s", proxyStr)
		return
	}
	logger.Warn("代理检测返回异常状态码: %d。请检查您的代理设置。", resp.StatusCode)
}

func IsConnectionError(err error) bool {
	if err == nil {
		return false
	}
	errStr := strings.ToLower(err.Error())
	return strings.Contains(errStr, "timeout") ||
		strings.Contains(errStr, "connection refused") ||
		strings.Contains(errStr, "proxy") ||
		strings.Contains(errStr, "dial") ||
		strings.Contains(errStr, "eof")
}
