package gemini

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestIsTransientNetworkError(t *testing.T) {
	if !isTransientNetworkError(context.DeadlineExceeded) {
		t.Fatal("expected context deadline exceeded to be transient")
	}
	if !isTransientNetworkError(errors.New("context deadline exceeded (Client.Timeout or context cancellation while reading body)")) {
		t.Fatal("expected client timeout while reading body to be transient")
	}
	if isTransientNetworkError(errors.New("Gemini returned login/consent page")) {
		t.Fatal("expected login/consent errors to remain non-transient")
	}
}

func TestParseToolCalls_FencedBlock(t *testing.T) {
	tools := []Tool{{Function: Function{Name: "get_weather"}}}
	content := "before\n```tool_call\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Shanghai\",\"unit\":\"c\"}}\n```\nafter"

	clean, calls := parseToolCalls(content, tools)
	if len(calls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(calls))
	}
	if calls[0].Function.Name != "get_weather" {
		t.Fatalf("unexpected tool name: %s", calls[0].Function.Name)
	}
	if calls[0].Function.Arguments != `{"city":"Shanghai","unit":"c"}` {
		t.Fatalf("unexpected arguments: %s", calls[0].Function.Arguments)
	}
	if strings.Contains(clean, "tool_call") {
		t.Fatalf("expected fenced block removed, got: %s", clean)
	}
}

func TestParseToolCalls_InlineAndStringifiedArgs(t *testing.T) {
	tools := []Tool{
		{Function: Function{Name: "search_web"}},
		{Function: Function{Name: "calculator"}},
	}
	content := strings.Join([]string{
		"Please run:",
		`{"name":"search_web","arguments":{"q":"golang regexp"}}`,
		"and then",
		`{"name":"calculator","arguments":"{\"expr\":\"15*37\"}"}`,
	}, "\n")

	clean, calls := parseToolCalls(content, tools)
	if len(calls) != 2 {
		t.Fatalf("expected 2 tool calls, got %d", len(calls))
	}
	if calls[0].Function.Name != "search_web" || calls[0].Function.Arguments != `{"q":"golang regexp"}` {
		t.Fatalf("unexpected first call: %+v", calls[0])
	}
	if calls[1].Function.Name != "calculator" || calls[1].Function.Arguments != `{"expr":"15*37"}` {
		t.Fatalf("unexpected second call: %+v", calls[1])
	}
	if strings.Contains(clean, `"name":"search_web"`) || strings.Contains(clean, `"name":"calculator"`) {
		t.Fatalf("expected tool json removed from content, got: %s", clean)
	}
}

func TestParseToolCalls_IgnoreUnknownAndDeduplicate(t *testing.T) {
	tools := []Tool{{Function: Function{Name: "get_weather"}}}
	content := strings.Join([]string{
		`{"name":"unknown_tool","arguments":{"x":1}}`,
		`{"name":"get_weather","arguments":{"city":"Beijing"}}`,
		`{"name":"get_weather","arguments":{"city":"Beijing"}}`,
	}, "\n")

	clean, calls := parseToolCalls(content, tools)
	if len(calls) != 1 {
		t.Fatalf("expected 1 deduplicated tool call, got %d", len(calls))
	}
	if calls[0].Function.Arguments != `{"city":"Beijing"}` {
		t.Fatalf("unexpected arguments: %s", calls[0].Function.Arguments)
	}
	if !strings.Contains(clean, "unknown_tool") {
		t.Fatalf("unknown tool should be preserved in content, got: %s", clean)
	}
}

func TestExtractDeepThinkContent(t *testing.T) {
	rcNode := []interface{}{
		"rc_testid",
		[]interface{}{"placeholder text"},
		nil, nil, nil, nil, nil,
		[]interface{}{1},
		"zh",
		nil, nil,
		nil,
		nil, nil, nil, nil, nil, nil, nil, nil,
		[]interface{}{false},
		nil, nil, nil, nil, nil, nil,
		[]interface{}{},
		nil, nil, nil, nil, nil, nil, nil, nil,
		[]interface{}{
			[]interface{}{"**Step One**\n\nFirst thinking step.\n\n\n**Step Two**\n\nSecond thinking step.\n\n\n"},
			[]interface{}{
				[]interface{}{
					"**Step One**\n\nFirst thinking step.\n\n\n**Step Two**\n\nSecond thinking step.\n\n\n",
					"", "",
					[]interface{}{
						[]interface{}{nil, []interface{}{nil, 0, "Step One", nil}},
						[]interface{}{nil, []interface{}{nil, 0, "First thinking step."}},
						[]interface{}{nil, []interface{}{nil, 0, "Step Two", nil}},
						[]interface{}{nil, []interface{}{nil, 0, "Second thinking step."}},
					},
				},
			},
		},
	}

	inner := []interface{}{
		[]interface{}{rcNode},
		nil, nil,
		"rc_testid",
	}

	innerJSON, err := json.Marshal(inner)
	if err != nil {
		t.Fatal(err)
	}

	var result contentResult
	visitRCNodesV2(inner, &result)

	if result.Content == "" {
		t.Fatal("expected non-empty content")
	}
	if !strings.Contains(result.Content, "placeholder") {
		t.Fatalf("unexpected content: %s", result.Content)
	}
	if result.ReasoningContent == "" {
		t.Fatalf("expected non-empty reasoning content. rcNode len=%d, innerJSON=%s", len(rcNode), string(innerJSON))
	}
	if !strings.Contains(result.ReasoningContent, "Step One") {
		t.Fatalf("expected 'Step One' in reasoning, got: %s", result.ReasoningContent)
	}
	if !strings.Contains(result.ReasoningContent, "Step Two") {
		t.Fatalf("expected 'Step Two' in reasoning, got: %s", result.ReasoningContent)
	}
}

func TestExtractThinkingFromRCNode(t *testing.T) {
	payload := `["rc_test",["placeholder"],null,null,null,null,null,null,null,[1],"zh",null,null,null,null,null,null,null,null,null,null,null,[false],null,null,null,null,null,null,[],null,null,null,null,null,null,null,null,[["**Step 1**\n\nThinking text here\n\n\n**Step 2**\n\nMore thinking\n\n\n"]]]`
	var items []interface{}
	if err := json.Unmarshal([]byte(payload), &items); err != nil {
		t.Fatal(err)
	}
	thinking := extractThinkingFromRCNode(items)
	if thinking == "" {
		t.Fatal("expected non-empty thinking content")
	}
	if !strings.Contains(thinking, "Step 1") || !strings.Contains(thinking, "Step 2") {
		t.Fatalf("expected thinking to contain steps, got: %s", thinking)
	}
}

func TestExtractThinkingFromRCNode_NoThinking(t *testing.T) {
	payload := `["rc_test",["hello world"],null,null,null,null,null,null,null,[1],"zh"]`
	var items []interface{}
	if err := json.Unmarshal([]byte(payload), &items); err != nil {
		t.Fatal(err)
	}
	thinking := extractThinkingFromRCNode(items)
	if thinking != "" {
		t.Fatalf("expected empty thinking for normal response, got: %s", thinking)
	}
}

func TestExtractThinkingFromRCNode_IgnoresMetadata(t *testing.T) {
	items := []interface{}{
		"rc_4083678137dd176e",
		[]interface{}{"9.9更大。"},
		nil,
		[]interface{}{"rc_4083678137dd176e", "US", "e6fa609c3fa255c0", "e6fa609c3fa255c0", "3.1 Pro"},
	}

	thinking := extractThinkingFromRCNode(items)
	if thinking != "" {
		t.Fatalf("expected metadata to be ignored, got: %s", thinking)
	}
}

func TestParseDataURI(t *testing.T) {
	mimeType, data, ok := parseDataURI("data:image/png;base64,iVBORw0KGgo=")
	if !ok {
		t.Fatal("expected data URI to parse")
	}
	if mimeType != "image/png" {
		t.Fatalf("unexpected mime type: %s", mimeType)
	}
	if data != "iVBORw0KGgo=" {
		t.Fatalf("unexpected data: %s", data)
	}
}

func TestExtractMultimodalContent_TextAndImages(t *testing.T) {
	msg := Message{
		Role: "user",
		Content: []interface{}{
			map[string]interface{}{"type": "text", "text": "Describe this"},
			map[string]interface{}{
				"type": "image_url",
				"image_url": map[string]interface{}{
					"url": "data:image/png;base64,iVBORw0KGgo=",
				},
			},
			map[string]interface{}{
				"type": "image_url",
				"image_url": map[string]interface{}{
					"url":    "https://example.com/cat.jpg",
					"detail": "high",
				},
			},
			map[string]interface{}{"type": "text", "text": "Use concise language"},
		},
	}

	parsed := extractMultimodalContent(msg)

	if parsed.Text != "Describe this\nUse concise language" {
		t.Fatalf("unexpected text: %q", parsed.Text)
	}
	if len(parsed.Images) != 2 {
		t.Fatalf("expected 2 images, got %d", len(parsed.Images))
	}
	if parsed.Images[0].MimeType != "image/png" || parsed.Images[0].Base64 != "iVBORw0KGgo=" {
		t.Fatalf("unexpected data URI image: %+v", parsed.Images[0])
	}
	if parsed.Images[1].URL != "https://example.com/cat.jpg" {
		t.Fatalf("unexpected URL image: %+v", parsed.Images[1])
	}
}

func TestExtractMessageContentUsesMultimodalText(t *testing.T) {
	msg := Message{
		Role: "user",
		Content: []interface{}{
			map[string]interface{}{"type": "text", "text": "first"},
			map[string]interface{}{"type": "image_url", "image_url": map[string]interface{}{"url": "data:image/jpeg;base64,abc"}},
			map[string]interface{}{"type": "text", "text": "second"},
		},
	}

	got := extractMessageContent(msg)
	if got != "first\nsecond" {
		t.Fatalf("unexpected text content: %q", got)
	}
}

func TestDownloadImageAsBase64(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte{1, 2, 3, 4})
	}))
	defer server.Close()

	image, err := downloadImageAsBase64(server.URL, server.Client())
	if err != nil {
		t.Fatalf("download image: %v", err)
	}
	if image.MimeType != "image/png" {
		t.Fatalf("unexpected mime type: %s", image.MimeType)
	}
	if image.Base64 != "AQIDBA==" {
		t.Fatalf("unexpected base64: %s", image.Base64)
	}
	if image.URL != server.URL {
		t.Fatalf("unexpected url: %s", image.URL)
	}
}

func TestBuildPromptWithMedia_TextOnlyMatchesBuildPrompt(t *testing.T) {
	req := ChatCompletionRequest{
		Messages: []Message{{Role: "user", Content: "hello"}},
	}

	legacy := BuildPrompt(req)
	prompt, images := BuildPromptWithMedia(req)

	if prompt != legacy {
		t.Fatalf("expected text prompt to match legacy output, got %q vs %q", prompt, legacy)
	}
	if len(images) != 0 {
		t.Fatalf("expected no images, got %d", len(images))
	}
}

func TestBuildPromptWithMedia_CollectsImages(t *testing.T) {
	req := ChatCompletionRequest{
		Messages: []Message{{
			Role: "user",
			Content: []interface{}{
				map[string]interface{}{"type": "text", "text": "see attached"},
				map[string]interface{}{"type": "image_url", "image_url": map[string]interface{}{"url": "data:image/png;base64,AAAA"}},
			},
		}},
	}

	prompt, images := BuildPromptWithMedia(req)

	if !strings.Contains(prompt, "User: see attached") {
		t.Fatalf("unexpected prompt: %s", prompt)
	}
	if len(images) != 1 {
		t.Fatalf("expected 1 image, got %d", len(images))
	}
	if images[0].MimeType != "image/png" || images[0].Base64 != "AAAA" {
		t.Fatalf("unexpected image: %+v", images[0])
	}
}

func TestGetExperimentalFeatureMode(t *testing.T) {
	tests := []struct {
		model string
		want  int
		ok    bool
	}{
		{model: "gemini-3-pro-image", want: featureModeImage, ok: true},
		{model: "gemini-3-pro-video", want: featureModeVideo, ok: true},
		{model: "gemini-3-pro", want: 0, ok: false},
	}

	for _, tt := range tests {
		got, ok := getExperimentalFeatureMode(tt.model)
		if got != tt.want || ok != tt.ok {
			t.Fatalf("model %s => (%d,%v), want (%d,%v)", tt.model, got, ok, tt.want, tt.ok)
		}
	}
}

func TestGetExperimentalRequestConfig(t *testing.T) {
	imageCfg, ok := getExperimentalRequestConfig("gemini-3-pro-image")
	if !ok {
		t.Fatal("expected image experimental config")
	}
	if imageCfg.FeatureMode != featureModeImage || imageCfg.Ef != featureModeImage || imageCfg.Xpc != "MODE_CATEGORY_FAST" {
		t.Fatalf("unexpected image config: %+v", imageCfg)
	}
	if imageCfg.Lo == nil || *imageCfg.Lo {
		t.Fatalf("expected image Lo=false, got %+v", imageCfg.Lo)
	}

	videoCfg, ok := getExperimentalRequestConfig("gemini-3-pro-video")
	if !ok {
		t.Fatal("expected video experimental config")
	}
	if videoCfg.FeatureMode != featureModeVideo || videoCfg.Ef != featureModeVideo || videoCfg.Xpc != "MODE_CATEGORY_FAST" {
		t.Fatalf("unexpected video config: %+v", videoCfg)
	}
	if videoCfg.Lo == nil || *videoCfg.Lo {
		t.Fatalf("expected video Lo=false, got %+v", videoCfg.Lo)
	}
}
