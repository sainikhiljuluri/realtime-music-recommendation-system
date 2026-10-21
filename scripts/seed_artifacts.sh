#!/usr/bin/env bash
# Copy a trained artifact directory into the named Docker volume that docker-compose.yml mounts.
#
# A bind mount of the artifacts directory failed under Colima's virtiofs mount (EDEADLK / I/O errors
# on reads inside the containers), so the stack reads from a named volume. The copy is a tar stream,
# so the volume is written from inside the Docker VM.
#
#   scripts/seed_artifacts.sh artifacts-music4all-10k
set -euo pipefail
SRC="${1:?usage: seed_artifacts.sh <artifact dir>}"
VOLUME="${VOLUME:-rtrec_artifacts}"
docker volume create "$VOLUME" >/dev/null
tar -C "$SRC" -cf - . | docker run --rm -i -v "$VOLUME:/a" alpine sh -c 'tar -xf - -C /a'
echo "seeded volume $VOLUME from $SRC"
