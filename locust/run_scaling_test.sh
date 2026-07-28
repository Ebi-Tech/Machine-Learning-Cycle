#!/bin/bash
# ==============================================================
# Automated Locust Scaling Test
# Runs the same Locust test at 1, 2, and 4 container replicas
# and saves CSV results for each configuration.
#
# Prerequisites:
#   - Docker Desktop running
#   - docker compose build already completed
#   - pip install locust (in your local Python environment)
#
# Usage:
#   chmod +x locust/run_scaling_test.sh
#   ./locust/run_scaling_test.sh
# ==============================================================

set -e

RESULTS_DIR="locust/results"
mkdir -p "$RESULTS_DIR"

HOST="http://localhost:8080"
USERS=100
SPAWN_RATE=20
DURATION="60s"

echo "=========================================="
echo " Locust Scaling Comparison Test"
echo "=========================================="
echo " Users: $USERS"
echo " Spawn rate: $SPAWN_RATE/sec"
echo " Duration per test: $DURATION"
echo " Target: $HOST"
echo "=========================================="

for SCALE in 1 2 4; do
    echo ""
    echo "---------- Scale: $SCALE container(s) ----------"

    echo "Starting containers at scale=$SCALE..."
    docker compose up -d --scale app=$SCALE

    echo "Waiting 15 seconds for containers to stabilize and nginx DNS to refresh..."
    sleep 15

    echo "Verifying API health..."
    if ! curl -s --fail "$HOST/health" > /dev/null 2>&1; then
        echo "ERROR: API not responding at $HOST/health. Aborting."
        docker compose logs app
        exit 1
    fi
    echo "API healthy. Running Locust test..."

    locust -f locust/locustfile.py \
        --host="$HOST" \
        --headless \
        -u "$USERS" \
        -r "$SPAWN_RATE" \
        --run-time "$DURATION" \
        --csv="$RESULTS_DIR/scale_${SCALE}" \
        --csv-full-history \
        --print-stats \
        2>&1 | tee "$RESULTS_DIR/scale_${SCALE}_output.txt"

    echo "Results saved to $RESULTS_DIR/scale_${SCALE}_*.csv"
    echo ""
done

echo ""
echo "=========================================="
echo " All tests complete"
echo "=========================================="
echo " Results directory: $RESULTS_DIR/"
echo ""
echo " Files generated per scale level:"
echo "   scale_N_stats.csv        - Summary statistics"
echo "   scale_N_stats_history.csv - Stats over time"
echo "   scale_N_failures.csv     - Failed requests"
echo "   scale_N_output.txt       - Full Locust console output"
echo ""
echo " Use these for the README comparison table."
echo "=========================================="

echo ""
echo "Stopping containers..."
docker compose down
echo "Done."
