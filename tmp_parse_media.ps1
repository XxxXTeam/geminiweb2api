param(
  [string]$Path = "mitm-gemini-tools.ndjson"
)

function ConvertTo-CompactJson($Value, [int]$Depth = 30) {
  if ($null -eq $Value) { return "null" }
  return ($Value | ConvertTo-Json -Compress -Depth $Depth)
}

function Read-CaptureEvents([string]$CapturePath) {
  if (-not (Test-Path -LiteralPath $CapturePath)) {
    throw "Capture file not found: $CapturePath"
  }
  Get-Content -LiteralPath $CapturePath | Where-Object { $_.Trim() -ne "" } | ForEach-Object { $_ | ConvertFrom-Json }
}

function Get-RequestBody([object]$Event) {
  if ($null -eq $Event.body) { return "" }
  return [uri]::UnescapeDataString([string]$Event.body)
}

function Get-QueryValue([string]$Url, [string]$Name) {
  try {
    $uri = [uri]$Url
    $query = [System.Web.HttpUtility]::ParseQueryString($uri.Query)
    return $query[$Name]
  } catch {
    return $null
  }
}

function Get-RpcIds([object]$Event) {
  $fromUrl = Get-QueryValue $Event.url "rpcids"
  if ($fromUrl) { return $fromUrl }
  $body = Get-RequestBody $Event
  if ($body -match '(?:^|&)rpcids=([^&]+)') { return $matches[1] }
  return ""
}

function ConvertFrom-JsonPreserve([string]$Text) {
  return ($Text | ConvertFrom-Json -NoEnumerate)
}

function Get-FReqText([object]$Event) {
  $body = Get-RequestBody $Event
  $idx = $body.IndexOf('f.req=')
  if ($idx -lt 0) { return "" }
  $start = $idx + 6
  $end = $body.IndexOf('&at=', $start)
  if ($end -lt 0) { $end = $body.IndexOf('&', $start) }
  if ($end -lt 0) { $end = $body.Length }
  return $body.Substring($start, $end - $start)
}

function Parse-BatchCalls([object]$Event) {
  $fReq = Get-FReqText $Event
  if (-not $fReq) { return @() }
  try {
    $outer = ConvertFrom-JsonPreserve $fReq
    $calls = @()
    if ($outer.Count -eq 0 -or $null -eq $outer[0]) { return $calls }
    for ($i = 0; $i -lt $outer[0].Count; $i++) {
      $call = $outer[0][$i]
      if ($null -eq $call -or $call.Count -lt 2) { continue }
      $payloadText = [string]$call[1]
      $payload = $null
      $payloadParseError = $null
      if ($payloadText) {
        try { $payload = ConvertFrom-JsonPreserve $payloadText } catch { $payloadParseError = $_.Exception.Message }
      }
      $calls += [pscustomobject]@{
        index = $i
        rpcid = [string]$call[0]
        mode = if ($call.Count -gt 3) { [string]$call[3] } else { "" }
        payloadText = $payloadText
        payload = $payload
        payloadParseError = $payloadParseError
      }
    }
    return $calls
  } catch {
    return @([pscustomobject]@{ index = 0; rpcid = Get-RpcIds $Event; mode = ""; payloadText = ""; payload = $null; payloadParseError = $_.Exception.Message })
  }
}

function Parse-StreamGenerateBody([object]$Event) {
  $fReq = Get-FReqText $Event
  if (-not $fReq) { return $null }
  try {
    if ($fReq.StartsWith('[[')) {
      $calls = Parse-BatchCalls $Event
      $candidate = @($calls | Where-Object { $_.rpcid -match 'StreamGenerate|BardFrontendService|CNgdBe' } | Select-Object -First 1)
      if ($candidate.Count -eq 0 -and $calls.Count -eq 1) { $candidate = @($calls[0]) }
      if ($candidate.Count -eq 0 -or -not $candidate[0].payloadText) { return $null }
      return $candidate[0].payload
    }
    $outerText = '[' + $fReq + ']'
    $outer = ConvertFrom-JsonPreserve $outerText
    $innerText = [string]$outer[1]
    return (ConvertFrom-JsonPreserve $innerText)
  } catch {
    return [pscustomobject]@{ parseError = $_.Exception.Message; raw = $fReq.Substring(0, [Math]::Min(500, $fReq.Length)) }
  }
}

function Get-NonEmptyFields([object[]]$Arr, [int]$MinIndex = 0) {
  $fields = @()
  if ($null -eq $Arr) { return $fields }
  for ($i = $MinIndex; $i -lt $Arr.Count; $i++) {
    $value = $Arr[$i]
    if ($null -eq $value) { continue }
    $json = ConvertTo-CompactJson $value
    if ($json -in @('null', '[]', '""')) { continue }
    $fields += [pscustomobject]@{ index = $i; value = $json }
  }
  return $fields
}

function Get-MessagePartSummary([object]$Field0) {
  $json = ConvertTo-CompactJson $Field0
  $signals = @()
  if ($json -match 'data:image\/') { $signals += 'data-image' }
  if ($json -match 'data:video\/') { $signals += 'data-video' }
  if ($json -match 'image_url|mimeType|mime_type') { $signals += 'image-field' }
  if ($json -match 'https?:\\/\\/[^" ]+') { $signals += 'url' }
  [pscustomobject]@{
    prompt = if ($Field0 -and $Field0.Count -gt 0) { [string]$Field0[0] } else { "" }
    raw = $json
    signals = $signals
  }
}

function Find-MediaSignals([string]$Text) {
  $signals = @()
  $patterns = @(
    @{ name = 'dataImage'; regex = 'data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=_-]+' },
    @{ name = 'dataVideo'; regex = 'data:video/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=_-]+' },
    @{ name = 'googleusercontentUrl'; regex = 'https?:\\/\\/[^"\s]*googleusercontent\.com[^"\s]*' },
    @{ name = 'mimeType'; regex = 'mimeType|mime_type|image/png|image/jpeg|video/mp4|video/webm' },
    @{ name = 'base64Like'; regex = '[A-Za-z0-9+/]{120,}={0,2}' }
  )
  foreach ($p in $patterns) {
    $matches = [regex]::Matches($Text, $p.regex)
    if ($matches.Count -gt 0) {
      $samples = @()
      foreach ($m in $matches | Select-Object -First 3) {
        $samples += $m.Value.Substring(0, [Math]::Min(160, $m.Value.Length))
      }
      $signals += [pscustomobject]@{ type = $p.name; count = $matches.Count; samples = $samples }
    }
  }
  return $signals
}

$events = Read-CaptureEvents $Path
$requests = @($events | Where-Object { $_.type -eq 'request' })
$responses = @($events | Where-Object { $_.type -eq 'response' })
$streamRequests = @($requests | Where-Object { $_.url -like '*StreamGenerate*' -or (Get-RpcIds $_) -match 'StreamGenerate' })
$otherBatchRequests = @($requests | Where-Object { $_.url -like '*batchexecute*' -and $_ -notin $streamRequests })

$streamSummaries = @()
$streamIndex = 0
foreach ($req in $streamRequests) {
  $streamIndex++
  $arr = Parse-StreamGenerateBody $req
  $batchCalls = Parse-BatchCalls $req
  if ($arr -and $arr.PSObject.Properties.Name -contains 'parseError') {
    $streamSummaries += [pscustomobject]@{ index = $streamIndex; id = $req.id; parseError = $arr.parseError; raw = $arr.raw; batchCalls = $batchCalls }
    continue
  }
  $streamSummaries += [pscustomobject]@{
    index = $streamIndex
    id = $req.id
    url = $req.url
    contentType = $req.content_type
    contentLength = $req.content_length
    rpcids = Get-RpcIds $req
    arrLength = if ($arr) { $arr.Count } else { 0 }
    message = Get-MessagePartSummary $arr[0]
    nonEmptyFields = Get-NonEmptyFields $arr
    highIndexFields = Get-NonEmptyFields $arr 69
    batchCalls = $batchCalls
  }
}

$batchSummaries = @()
foreach ($req in $otherBatchRequests) {
  $body = Get-RequestBody $req
  $calls = Parse-BatchCalls $req
  $signals = Find-MediaSignals $body
  $payloadSignals = @()
  foreach ($call in $calls) {
    $payloadSignals += [pscustomobject]@{
      rpcid = $call.rpcid
      mode = $call.mode
      payloadLength = $call.payloadText.Length
      payloadPreview = $call.payloadText.Substring(0, [Math]::Min(1000, $call.payloadText.Length))
      parseError = $call.payloadParseError
      nonEmptyFields = if ($call.payload -is [object[]]) { Get-NonEmptyFields $call.payload } else { @() }
      highIndexFields = if ($call.payload -is [object[]]) { Get-NonEmptyFields $call.payload 69 } else { @() }
      mediaSignals = Find-MediaSignals $call.payloadText
    }
  }
  $batchSummaries += [pscustomobject]@{
    id = $req.id
    url = $req.url
    rpcids = Get-RpcIds $req
    contentType = $req.content_type
    contentLength = $req.content_length
    likelyUpload = (($req.content_type -like '*multipart*') -or ($body -match 'upload|image|video|file|blob|media'))
    bodyPreview = $body.Substring(0, [Math]::Min(1000, $body.Length))
    batchCalls = $payloadSignals
    mediaSignals = $signals
  }
}

$responseSignals = @()
foreach ($resp in $responses) {
  $body = [string]$resp.body
  $signals = Find-MediaSignals $body
  if ($signals.Count -eq 0) { continue }
  $responseSignals += [pscustomobject]@{
    id = $resp.id
    statusCode = $resp.status_code
    contentType = $resp.content_type
    mediaSignals = $signals
  }
}

$rpcCounts = @{}
foreach ($req in $requests | Where-Object { $_.url -like '*batchexecute*' }) {
  $rpc = Get-RpcIds $req
  if (-not $rpc) { continue }
  if (-not $rpcCounts.ContainsKey($rpc)) { $rpcCounts[$rpc] = 0 }
  $rpcCounts[$rpc]++
}
$likelyMediaRpcids = @(
  $rpcCounts.Keys |
    Sort-Object { $rpcCounts[$_] } -Descending |
    Select-Object -First 8
)

$diagnostics = [pscustomobject]@{
  streamGenerateDetected = ($streamSummaries.Count -gt 0)
  batchexecuteRpcCount = $rpcCounts.Count
  topRpcids = @($likelyMediaRpcids | ForEach-Object { [pscustomobject]@{ rpcid = $_; count = $rpcCounts[$_] } })
  note = if ($streamSummaries.Count -eq 0) { "当前抓包未出现显式 StreamGenerate；主要是 batchexecute 外层 RPC，需要继续看 payload。" } else { "已检测到 StreamGenerate 候选。" }
}

[pscustomobject]@{
  eventCount = $events.Count
  requestCount = $requests.Count
  responseCount = $responses.Count
  streamGenerate = $streamSummaries
  otherBatchExecute = $batchSummaries
  responseMediaSignals = $responseSignals
  diagnostics = $diagnostics
} | ConvertTo-Json -Depth 50
