#!/bin/sh
set -eu

APP_BIN="/app/geminiweb2api"
APP_DIR="/app"

can_write_app_dir() {
  test_file="$APP_DIR/.perm-check"
  if touch "$test_file" 2>/dev/null; then
    rm -f "$test_file" 2>/dev/null || true
    return 0
  fi
  return 1
}

if [ "$(id -u)" = "0" ]; then
  if can_write_app_dir; then
    chown -R app:app "$APP_DIR" 2>/dev/null || true
    exec su-exec app "$APP_BIN"
  fi

  echo "[WARN] /app volume is not writable by non-root user; falling back to root runtime"
  exec "$APP_BIN"
fi

exec "$APP_BIN"
