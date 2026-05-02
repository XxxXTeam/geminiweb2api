from mitmproxy import http
from pathlib import Path
import json
import time

OUT = Path("mitm-gemini-tools.ndjson")
SENSITIVE_HEADERS = {"cookie", "authorization", "x-client-data"}
BODY_LIMIT = 500 * 1024
MULTIPART_PART_LIMIT = 16 * 1024


def scrub_headers(headers):
    result = {}
    for key, value in headers.items():
        if key.lower() in SENSITIVE_HEADERS:
            result[key] = f"<redacted:{len(value)}>"
        else:
            result[key] = value
    return result


def body_text(content, content_type=""):
    if not content:
        return ""
    if "multipart/form-data" in (content_type or "").lower():
        return multipart_body_text(content)
    text = content.decode("utf-8", errors="replace")
    if len(text) > BODY_LIMIT:
        return text[:BODY_LIMIT] + "\n<trimmed>"
    return text


def multipart_body_text(content):
    text = content.decode("utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    result = []
    in_binary_part = False
    binary_size = 0
    trimmed_binary = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("--"):
            if trimmed_binary:
                result.append(f"<part-trimmed:{binary_size}>\r\n")
            result.append(line)
            in_binary_part = False
            binary_size = 0
            trimmed_binary = False
            continue
        if line.lower().startswith("content-type: image/") or line.lower().startswith("content-type: video/"):
            in_binary_part = True
            result.append(line)
            continue
        if in_binary_part and stripped and not line.lower().startswith("content-"):
            binary_size += len(line)
            if binary_size <= MULTIPART_PART_LIMIT:
                result.append(line)
            else:
                trimmed_binary = True
            continue
        result.append(line)

    captured = "".join(result)
    if len(captured) > BODY_LIMIT:
        return captured[:BODY_LIMIT] + "\n<trimmed>"
    return captured


def header_value(headers, name):
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return ""


def content_length(headers):
    value = header_value(headers, "content-length")
    if not value:
        return 0
    try:
        return int(value)
    except ValueError:
        return 0


def append(event):
    with OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def request(flow: http.HTTPFlow):
    if "gemini.google.com" not in (flow.request.pretty_host or ""):
        return
    flow.metadata["capture_gemini"] = True
    content_type = header_value(flow.request.headers, "content-type")
    append({
        "ts": time.time(),
        "type": "request",
        "id": flow.id,
        "method": flow.request.method,
        "url": flow.request.pretty_url,
        "content_type": content_type,
        "content_length": content_length(flow.request.headers),
        "headers": scrub_headers(flow.request.headers),
        "body": body_text(flow.request.raw_content, content_type),
    })


def response(flow: http.HTTPFlow):
    if not flow.metadata.get("capture_gemini"):
        return
    content_type = header_value(flow.response.headers, "content-type")
    append({
        "ts": time.time(),
        "type": "response",
        "id": flow.id,
        "status_code": flow.response.status_code,
        "content_type": content_type,
        "headers": scrub_headers(flow.response.headers),
        "body": body_text(flow.response.content, content_type),
    })
