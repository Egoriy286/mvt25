#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if command -v docker >/dev/null 2>&1; then
    docker_cli="$(command -v docker)"
elif [[ -x "${HOME}/.docker/bin/docker" ]]; then
    docker_cli="${HOME}/.docker/bin/docker"
elif [[ -x /Applications/Docker.app/Contents/Resources/bin/docker ]]; then
    docker_cli=/Applications/Docker.app/Contents/Resources/bin/docker
else
    echo 'Запустите Docker Desktop, затем повторите открытие в контейнере.' >&2
    exit 1
fi

"${docker_cli}" info >/dev/null
"${docker_cli}" build --platform linux/arm64 \
    --tag mvt25/fenics-legacy:local \
    "${project_root}/.devcontainer"
