#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/gpu-compose.sh --service NAME [options] -- [compose arguments]

Build a temporary GPU override from deepiri-gpu compose-gpu, print the
deepiri-gpu env-hints guidance, and run Docker Compose with the override.

Options:
  --service NAME        Compose service that runs HeloX (required unless
                        HELOX_COMPOSE_SERVICE is set)
  --compose-file PATH   Base Compose file; may be repeated
  --backend BACKEND     Force cuda, rocm, mps, or cpu (default: auto-detect)
  -h, --help            Show this help

If --compose-file is omitted, COMPOSE_FILE or a standard Compose filename in
the current directory is used. Normal `docker compose` remains the CPU/default
path; this wrapper is only needed when GPU-aware overrides are wanted.
EOF
}

service_name=${HELOX_COMPOSE_SERVICE:-}
backend=""
compose_files=()
compose_args=()

while (($#)); do
    case "$1" in
        --service)
            [[ $# -ge 2 ]] || { echo "--service requires a value" >&2; exit 2; }
            service_name=$2
            shift 2
            ;;
        --compose-file)
            [[ $# -ge 2 ]] || { echo "--compose-file requires a value" >&2; exit 2; }
            compose_files+=("$2")
            shift 2
            ;;
        --backend)
            [[ $# -ge 2 ]] || { echo "--backend requires a value" >&2; exit 2; }
            backend=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            compose_args=("$@")
            break
            ;;
        *)
            compose_args=("$@")
            break
            ;;
    esac
done

if [[ -z "$service_name" ]]; then
    echo "Set HELOX_COMPOSE_SERVICE or pass --service NAME." >&2
    exit 2
fi

if command -v deepiri-gpu >/dev/null 2>&1; then
    gpu_cli=(deepiri-gpu)
elif command -v poetry >/dev/null 2>&1; then
    gpu_cli=(poetry run deepiri-gpu)
else
    echo "deepiri-gpu is required. Run 'poetry install' or put it on PATH." >&2
    exit 127
fi

if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
    echo "Docker Compose v2 ('docker compose') is required." >&2
    exit 127
fi

if ((${#compose_files[@]} == 0)); then
    if [[ -n ${COMPOSE_FILE:-} ]]; then
        IFS=: read -r -a compose_files <<<"$COMPOSE_FILE"
    else
        for candidate in compose.yaml compose.yml docker-compose.yaml docker-compose.yml; do
            if [[ -f "$candidate" ]]; then
                compose_files+=("$candidate")
                break
            fi
        done
    fi
fi

if ((${#compose_files[@]} == 0)); then
    echo "No Compose file found. Pass --compose-file PATH or set COMPOSE_FILE." >&2
    exit 2
fi

for compose_file in "${compose_files[@]}"; do
    if [[ ! -f "$compose_file" ]]; then
        echo "Compose file not found: $compose_file" >&2
        exit 2
    fi
done

backend_args=()
if [[ -n "$backend" ]]; then
    backend_args=(--backend "$backend")
fi

echo "GPU Compose guidance for service '$service_name':"
"${gpu_cli[@]}" compose-gpu "${backend_args[@]}"
echo
echo "Environment guidance (review and export values appropriate for this host):"
"${gpu_cli[@]}" env-hints "${backend_args[@]}"

compose_json=$("${gpu_cli[@]}" compose-gpu "${backend_args[@]}" --json)
override_file=$(mktemp "${TMPDIR:-/tmp}/diri-helox-gpu-compose.XXXXXX.json")
trap 'rm -f -- "$override_file"' EXIT

if command -v python3 >/dev/null 2>&1; then
    python_bin=python3
elif command -v python >/dev/null 2>&1; then
    python_bin=python
else
    echo "Python 3 is required to render the Compose override." >&2
    exit 127
fi

HELOX_GPU_COMPOSE_JSON=$compose_json "$python_bin" - "$service_name" "$override_file" <<'PY'
import json
import os
import sys

service_name, output_path = sys.argv[1:]
gpu = json.loads(os.environ["HELOX_GPU_COMPOSE_JSON"])
service = {}

device_requests = gpu.get("device_requests", [])
if device_requests:
    service["deploy"] = {
        "resources": {"reservations": {"devices": device_requests}}
    }

run_args = iter(gpu.get("run_gpu_args", []))
devices = []
groups = []
for argument in run_args:
    if argument == "--device":
        device = next(run_args)
        devices.append(f"{device}:{device}")
    elif argument == "--group-add":
        groups.append(next(run_args))

if devices:
    service["devices"] = devices
if groups:
    service["group_add"] = groups
if gpu.get("environment"):
    service["environment"] = gpu["environment"]

with open(output_path, "w", encoding="utf-8") as handle:
    json.dump({"services": {service_name: service}}, handle, indent=2)
    handle.write("\n")
PY

compose_file_args=()
for compose_file in "${compose_files[@]}"; do
    compose_file_args+=(-f "$compose_file")
done

if ((${#compose_args[@]} == 0)); then
    compose_args=(up)
fi

docker compose "${compose_file_args[@]}" -f "$override_file" "${compose_args[@]}"
