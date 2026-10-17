#!/usr/bin/env bash
# Keep trying to get one large single-GPU Brev instance (>= 80 GB VRAM) until one is RUNNING.
#
#   bash deploy/brev_provision_loop.sh                 # default: check every 3 min
#   INTERVAL=120 STUCK_MIN=15 bash deploy/brev_provision_loop.sh
#   DRY_RUN=1 bash deploy/brev_provision_loop.sh       # print what it would do, create nothing
#
# Each round:
#   1. Any of your instances RUNNING + shell READY  -> done (notify, exit).
#   2. One of ours (name starts with $PREFIX) still deploying and younger than STUCK_MIN -> wait.
#   3. Ours FAILED, or deploying longer than STUCK_MIN -> delete it (only $PREFIX-* instances, never others).
#   4. Nothing of ours in flight -> `brev create --detached` with the fallback type list (cheapest first).
# At most one $PREFIX-* instance exists at a time, so this never stacks up billing.
# Needs: brev (logged in), jq.
set -euo pipefail

PREFIX="${PREFIX:-risklens-gpu}"
INTERVAL="${INTERVAL:-180}"          # seconds between rounds
STUCK_MIN="${STUCK_MIN:-20}"         # deploying longer than this = give up on it
MAX_ROUNDS="${MAX_ROUNDS:-60}"       # 60 x 3 min = 3 h, then stop
DRY_RUN="${DRY_RUN:-0}"
# Single GPU, >= 80 GB, cheapest first (from `brev search -v 80`). A100: BF16 LoRA fine, NVFP4 served via W4A16.
TYPES="${TYPES:-hyperstack_A100_80G,massedcompute_A100_sxm4_80G,massedcompute_A100_sxm4_80G_DGX,denvr_A100_sxm4_80G,hyperstack_H100,scaleway_H100,gpu_1x_h100_pcie,g7e.2xlarge,gpu-h100-sxm.1gpu-16vcpu-200gb,gpu-h200-sxm.1gpu-16vcpu-200gb}"

STATE="${TMPDIR:-/tmp}/${PREFIX}.started"   # "<name> <epoch>" of the instance we are waiting on

log() { printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*"; }

notify() {
  log "$1"
  command -v osascript >/dev/null && osascript -e "display notification \"$1\" with title \"Brev\" sound name \"Glass\"" || true
}

instances() { brev ls --json 2>/dev/null | jq -c '.workspaces // [] | .[]'; }

for ((round = 1; round <= MAX_ROUNDS; round++)); do
  all="$(instances)"

  ready="$(jq -r 'select(.status == "RUNNING" and .shell_status == "READY") | "\(.name) (\(.instance_type))"' <<<"$all" | head -1)"
  if [[ -n "$ready" ]]; then
    notify "READY: $ready - billing has started. Connect: brev shell ${ready%% *}"
    others="$(jq -r --arg n "${ready%% *}" 'select(.name != $n) | "  \(.name)  \(.status)/\(.build_status)  \(.instance_type)"' <<<"$all")"
    [[ -n "$others" ]] && log "Other instances still exist (delete any you don't need to avoid double billing):" && echo "$others"
    rm -f "$STATE"
    exit 0
  fi

  ours="$(jq -r --arg p "$PREFIX-" 'select(.name | startswith($p)) | "\(.name) \(.status) \(.build_status)"' <<<"$all")"
  if [[ -n "$ours" ]]; then
    read -r name status build <<<"$(head -1 <<<"$ours")"
    started="$(awk -v n="$name" '$1 == n {print $2}' "$STATE" 2>/dev/null || true)"
    [[ -z "$started" ]] && started="$(date +%s)" && echo "$name $started" >"$STATE"
    age_min=$(( ($(date +%s) - started) / 60 ))
    if [[ "$status$build" == *FAIL* || "$status$build" == *ERROR* || $age_min -ge $STUCK_MIN ]]; then
      log "round $round: $name is $status/$build after ${age_min}m - deleting and retrying"
      [[ "$DRY_RUN" == 1 ]] || brev delete "$name" || log "delete failed; will retry next round"
      rm -f "$STATE"
    else
      log "round $round: waiting on $name ($status/$build, ${age_min}m of ${STUCK_MIN}m)"
    fi
  else
    name="$PREFIX-$(date +%m%d-%H%M)"
    log "round $round: creating $name (types: $TYPES)"
    if [[ "$DRY_RUN" == 1 ]]; then
      brev create "$name" --type "$TYPES" --dry-run || true
    elif brev create "$name" --type "$TYPES" --detached; then
      echo "$name $(date +%s)" >"$STATE"
    else
      log "create failed (no capacity on any type?) - retrying in ${INTERVAL}s"
    fi
  fi

  sleep "$INTERVAL"
done

notify "Gave up after $MAX_ROUNDS rounds without a RUNNING instance."
exit 1
