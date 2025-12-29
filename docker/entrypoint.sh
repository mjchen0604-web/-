#!/usr/bin/env sh
set -eu

CONFIG_PATH="${CONFIG_PATH:-/data/config.yaml}"
PORT="${PORT:-8317}"
AUTH_DIR="${AUTH_DIR:-/data/auths}"
WRITABLE_PATH="${WRITABLE_PATH:-/data}"
STATIC_DIR="${MANAGEMENT_STATIC_PATH:-${WRITABLE_PATH}/static}"
STATIC_FILE="${STATIC_DIR}/management.html"
PANEL_URL="https://github.com/router-for-me/Cli-Proxy-API-Management-Center/releases/latest/download/management.html"
LOCAL_PANEL="/CLIProxyAPI/management.html"

export WRITABLE_PATH

mkdir -p "$(dirname "$CONFIG_PATH")" "${AUTH_DIR}" "${WRITABLE_PATH}/logs"

mkdir -p "${STATIC_DIR}"
if [ -s "${LOCAL_PANEL}" ]; then
  cp -f "${LOCAL_PANEL}" "${STATIC_FILE}"
elif [ ! -s "${STATIC_FILE}" ]; then
  if command -v wget >/dev/null 2>&1; then
    wget -q -O "${STATIC_FILE}" "${PANEL_URL}" || true
  elif command -v curl >/dev/null 2>&1; then
    curl -fsSL "${PANEL_URL}" -o "${STATIC_FILE}" || true
  fi
fi

if [ ! -s "${CONFIG_PATH}" ]; then
  cat > "${CONFIG_PATH}" <<EOF
host: ""
port: ${PORT}
auth-dir: "${AUTH_DIR}"
logging-to-file: true
usage-statistics-enabled: true
EOF
fi

exec /CLIProxyAPI/CLIProxyAPI -config "${CONFIG_PATH}"
