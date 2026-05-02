FROM golang:1.25-alpine AS builder

WORKDIR /src

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -o /out/geminiweb2api ./cmd/geminiweb2api


FROM alpine:3.21

WORKDIR /app

RUN apk add --no-cache ca-certificates tzdata wget su-exec && update-ca-certificates \
    && addgroup -S app && adduser -S -G app app \
    && mkdir -p /app && chown -R app:app /app

COPY --from=builder /out/geminiweb2api /app/geminiweb2api
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN chown app:app /app/geminiweb2api \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8080

VOLUME ["/app"]

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 CMD sh -c 'wget -q -O /dev/null http://127.0.0.1:8080/api/telemetry || exit 1'

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
