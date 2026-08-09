#!/usr/bin/env bash
set -Eeuo pipefail

# Health supervisor for the public scoreboard stack.  It intentionally waits
# for consecutive failures before taking action so a short Cloudflare/network
# flap cannot create a restart loop.

readonly API_UNIT="helicopter-public-api.service"
readonly FRONTEND_UNIT="helicopter-public-frontend.service"
readonly TUNNEL_A_UNIT="cloudflared-shp6000-a.service"
readonly TUNNEL_B_UNIT="cloudflared-shp6000-b.service"

readonly API_HEALTH_URL="http://127.0.0.1:7862/api/health"
readonly FRONTEND_URL="http://127.0.0.1:7860/"
readonly PUBLIC_HEALTH_URL="https://shp6000.rwkvos.com/api/health"
readonly PUBLIC_FRONTEND_URL="https://shp6000.rwkvos.com/"
readonly FRONTEND_MARKER="RWKV Skills"
readonly TUNNEL_A_METRICS_URL="http://127.0.0.1:20241/metrics"
readonly TUNNEL_B_METRICS_URL="http://127.0.0.1:20242/metrics"

readonly CHECK_INTERVAL_SECONDS="${HELICOPTER_WATCHDOG_INTERVAL_SECONDS:-15}"
readonly LOCAL_FAILURE_LIMIT="${HELICOPTER_WATCHDOG_LOCAL_FAILURE_LIMIT:-3}"
readonly PUBLIC_FAILURE_LIMIT="${HELICOPTER_WATCHDOG_PUBLIC_FAILURE_LIMIT:-4}"
readonly PUBLIC_ROLLING_RESTART_LIMIT="${HELICOPTER_WATCHDOG_PUBLIC_RESTART_LIMIT:-12}"
readonly TUNNEL_ZERO_CONNECTION_LIMIT="${HELICOPTER_WATCHDOG_TUNNEL_ZERO_LIMIT:-4}"
readonly TUNNEL_METRICS_UNAVAILABLE_LIMIT="${HELICOPTER_WATCHDOG_TUNNEL_METRICS_UNAVAILABLE_LIMIT:-12}"
readonly WATCHDOG_LOCK_FILE="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/helicopter-public-watchdog.lock"

api_failures=0
frontend_failures=0
public_failures=0
tunnel_a_zero_checks=0
tunnel_b_zero_checks=0
tunnel_a_metrics_unavailable_checks=0
tunnel_b_metrics_unavailable_checks=0

log_message() {
  local message="$*"
  printf '%s %s\n' "$(date -Is)" "$message" >&2
}

probe_get() {
  HTTPS_PROXY= HTTP_PROXY= ALL_PROXY= \
    https_proxy= http_proxy= all_proxy= \
    NO_PROXY='*' no_proxy='*' \
    curl --noproxy '*' --fail --silent --show-error \
    --connect-timeout 3 --max-time 10 \
    --output /dev/null "$1"
}

probe_frontend_content() {
  HTTPS_PROXY= HTTP_PROXY= ALL_PROXY= \
    https_proxy= http_proxy= all_proxy= \
    NO_PROXY='*' no_proxy='*' \
    curl --noproxy '*' --fail --silent --show-error \
    --connect-timeout 3 --max-time 10 "$1" \
    | grep --fixed-strings "$FRONTEND_MARKER" >/dev/null
}

probe_public_stack() {
  # The API probe verifies readiness through the public route.  The root GET
  # also validates a build marker so an unrelated 200 page cannot look healthy.
  # Both probes explicitly bypass inherited proxy state.
  probe_get "$PUBLIC_HEALTH_URL" && probe_frontend_content "$PUBLIC_FRONTEND_URL"
}

tunnel_ha_connections() {
  local metrics_url="$1"
  local value
  value="$({
    HTTPS_PROXY= HTTP_PROXY= ALL_PROXY= \
      https_proxy= http_proxy= all_proxy= \
      NO_PROXY='*' no_proxy='*' \
      curl --noproxy '*' --fail --silent --show-error \
      --connect-timeout 2 --max-time 4 "$metrics_url"
  } | awk '$1 == "cloudflared_tunnel_ha_connections" { print int($2); exit }')" || return 1
  [[ "$value" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$value"
}

wait_for_tunnel_connections() {
  local metrics_url="$1"
  local timeout_seconds="${2:-45}"
  local waited=0
  local connections
  while (( waited < timeout_seconds )); do
    if connections="$(tunnel_ha_connections "$metrics_url")" && (( connections > 0 )); then
      return 0
    fi
    sleep 3
    ((waited += 3))
  done
  return 1
}

monitor_tunnel_connections() {
  local unit="$1"
  local metrics_url="$2"
  local label="$3"
  local counter_name="$4"
  local unavailable_counter_name="$5"
  local -n zero_checks="$counter_name"
  local -n unavailable_checks="$unavailable_counter_name"
  local connections

  if ! connections="$(tunnel_ha_connections "$metrics_url")"; then
    zero_checks=0
    ((unavailable_checks += 1))
    if (( unavailable_checks == 1 || unavailable_checks == TUNNEL_METRICS_UNAVAILABLE_LIMIT )); then
      log_message "connector ${label} metrics unavailable ${unavailable_checks}/${TUNNEL_METRICS_UNAVAILABLE_LIMIT}; refusing blind restart until the sibling is confirmed healthy"
    fi
    return 1
  fi

  if (( unavailable_checks > 0 )); then
    log_message "connector ${label} metrics recovered after ${unavailable_checks} unavailable checks"
  fi
  unavailable_checks=0

  if (( connections > 0 )); then
    if (( zero_checks > 0 )); then
      log_message "connector ${label} HA recovered after ${zero_checks} confirmed-zero checks"
    fi
    zero_checks=0
    return 0
  fi

  ((zero_checks += 1))
  log_message "connector ${label} has zero HA connections ${zero_checks}/${TUNNEL_ZERO_CONNECTION_LIMIT}"
  if (( zero_checks < TUNNEL_ZERO_CONNECTION_LIMIT )); then
    return 1
  fi

  restart_unit "$unit" "connector ${label} remained without HA connections"
  if wait_for_tunnel_connections "$metrics_url"; then
    log_message "connector ${label} recovered HA connections after restart"
  else
    log_message "connector ${label} still has no HA connections after restart"
  fi
  zero_checks=0
}

restart_unit() {
  local unit="$1"
  local reason="$2"
  log_message "restarting ${unit}: ${reason}"
  systemctl --user reset-failed "$unit" 2>/dev/null || true
  systemctl --user restart "$unit"
}

ensure_unit_active() {
  local unit="$1"
  if systemctl --user is-active --quiet "$unit"; then
    return 0
  fi
  restart_unit "$unit" "unit is not active"
}

rolling_tunnel_restart() {
  local a_connections
  local b_connections
  if ! a_connections="$(tunnel_ha_connections "$TUNNEL_A_METRICS_URL")"; then
    if b_connections="$(tunnel_ha_connections "$TUNNEL_B_METRICS_URL")" \
      && (( b_connections > 0 )) \
      && (( tunnel_a_metrics_unavailable_checks >= TUNNEL_METRICS_UNAVAILABLE_LIMIT )); then
      restart_unit "$TUNNEL_A_UNIT" "connector A metrics remained unavailable while connector B is healthy"
      wait_for_tunnel_connections "$TUNNEL_A_METRICS_URL" \
        && log_message "connector A metrics and HA connections recovered after restart" \
        || log_message "connector A metrics remain unavailable after restart"
      tunnel_a_metrics_unavailable_checks=0
      return 0
    fi
    log_message "public health remains down but connector A metrics are unavailable and no safe sibling-backed restart is available"
    return 1
  fi
  if ! b_connections="$(tunnel_ha_connections "$TUNNEL_B_METRICS_URL")"; then
    if (( a_connections > 0 )) \
      && (( tunnel_b_metrics_unavailable_checks >= TUNNEL_METRICS_UNAVAILABLE_LIMIT )); then
      restart_unit "$TUNNEL_B_UNIT" "connector B metrics remained unavailable while connector A is healthy"
      wait_for_tunnel_connections "$TUNNEL_B_METRICS_URL" \
        && log_message "connector B metrics and HA connections recovered after restart" \
        || log_message "connector B metrics remain unavailable after restart"
      tunnel_b_metrics_unavailable_checks=0
      return 0
    fi
    log_message "public health remains down but connector B metrics are unavailable and no safe sibling-backed restart is available"
    return 1
  fi

  # A connector restart can briefly leave stale edge routes that return 502,
  # even when the sibling connector is healthy.  Never restart a connector
  # whose HA metric is non-zero merely because an external probe failed.
  if (( a_connections > 0 && b_connections > 0 )); then
    log_message "public health failed ${public_failures} checks but connectors are healthy (A=${a_connections}, B=${b_connections}); not restarting"
    return 0
  fi

  if (( a_connections == 0 )); then
    restart_unit "$TUNNEL_A_UNIT" "connector A has zero HA connections"
    if wait_for_tunnel_connections "$TUNNEL_A_METRICS_URL"; then
      log_message "connector A recovered HA connections"
    else
      log_message "connector A still has zero HA connections after restart"
    fi
  fi

  if (( b_connections == 0 )); then
    restart_unit "$TUNNEL_B_UNIT" "connector B has zero HA connections"
    if wait_for_tunnel_connections "$TUNNEL_B_METRICS_URL"; then
      log_message "connector B recovered HA connections"
    else
      log_message "connector B still has zero HA connections after restart"
    fi
  fi

  if probe_public_stack; then
    log_message "public health recovered after repairing zero-connection connector(s)"
    return 0
  fi
}

if [[ "${1:-}" == "--check-once" ]]; then
  api_status=fail
  frontend_status=fail
  public_status=fail
  probe_get "$API_HEALTH_URL" && api_status=ok
  probe_frontend_content "$FRONTEND_URL" && frontend_status=ok
  probe_public_stack && public_status=ok
  a_connections="$(tunnel_ha_connections "$TUNNEL_A_METRICS_URL" || printf 'unknown')"
  b_connections="$(tunnel_ha_connections "$TUNNEL_B_METRICS_URL" || printf 'unknown')"
  printf 'api=%s frontend=%s public=%s tunnel_a=%s tunnel_b=%s\n' \
    "$api_status" "$frontend_status" "$public_status" "$a_connections" "$b_connections"
  [[ "$api_status" == ok && "$frontend_status" == ok && "$public_status" == ok ]]
  exit
fi

# The systemd service is the sole long-running supervisor.  Keep check-once
# lock-free so operators can inspect health while the daemon owns this lock.
exec 9>"$WATCHDOG_LOCK_FILE"
if ! flock --nonblock 9; then
  log_message "another watchdog instance already owns ${WATCHDOG_LOCK_FILE}; exiting duplicate"
  exit 0
fi

log_message "watchdog started (interval=${CHECK_INTERVAL_SECONDS}s, local_limit=${LOCAL_FAILURE_LIMIT}, public_restart_limit=${PUBLIC_ROLLING_RESTART_LIMIT})"

while true; do
  ensure_unit_active "$API_UNIT" || true
  ensure_unit_active "$FRONTEND_UNIT" || true
  ensure_unit_active "$TUNNEL_A_UNIT" || true
  ensure_unit_active "$TUNNEL_B_UNIT" || true
  monitor_tunnel_connections \
    "$TUNNEL_A_UNIT" "$TUNNEL_A_METRICS_URL" A \
    tunnel_a_zero_checks tunnel_a_metrics_unavailable_checks || true
  monitor_tunnel_connections \
    "$TUNNEL_B_UNIT" "$TUNNEL_B_METRICS_URL" B \
    tunnel_b_zero_checks tunnel_b_metrics_unavailable_checks || true

  if probe_get "$API_HEALTH_URL"; then
    if (( api_failures > 0 )); then
      log_message "API health recovered after ${api_failures} failed checks"
    fi
    api_failures=0
  else
    ((api_failures += 1))
    log_message "API health failure ${api_failures}/${LOCAL_FAILURE_LIMIT}"
    if (( api_failures >= LOCAL_FAILURE_LIMIT )); then
      restart_unit "$API_UNIT" "API health threshold reached" || true
      api_failures=0
      sleep 5
    fi
  fi

  if probe_frontend_content "$FRONTEND_URL"; then
    if (( frontend_failures > 0 )); then
      log_message "frontend health recovered after ${frontend_failures} failed checks"
    fi
    frontend_failures=0
  else
    ((frontend_failures += 1))
    log_message "frontend health failure ${frontend_failures}/${LOCAL_FAILURE_LIMIT}"
    if (( frontend_failures >= LOCAL_FAILURE_LIMIT )); then
      restart_unit "$FRONTEND_UNIT" "frontend health threshold reached" || true
      frontend_failures=0
      sleep 5
    fi
  fi

  # Only diagnose the external path when both origin processes are healthy.
  # Otherwise the local recovery above is the authoritative first action.
  if (( api_failures == 0 && frontend_failures == 0 )); then
    if probe_public_stack; then
      if (( public_failures > 0 )); then
        log_message "public health recovered after ${public_failures} failed checks"
      fi
      public_failures=0
    else
      ((public_failures += 1))
      log_message "public health failure ${public_failures}/${PUBLIC_ROLLING_RESTART_LIMIT}"

      # At the early threshold, repair only a connector that systemd already
      # considers dead.  Live connectors are left alone during short edge or
      # DNS incidents because cloudflared has its own reconnect loop.
      if (( public_failures == PUBLIC_FAILURE_LIMIT )); then
        ensure_unit_active "$TUNNEL_A_UNIT" || true
        ensure_unit_active "$TUNNEL_B_UNIT" || true
      fi

      # A long outage receives a rolling restart, never a simultaneous one,
      # so at least one connector remains available throughout recovery.
      if (( public_failures >= PUBLIC_ROLLING_RESTART_LIMIT )); then
        rolling_tunnel_restart || true
        public_failures=0
        sleep 8
      fi
    fi
  fi

  sleep "$CHECK_INTERVAL_SECONDS"
done
