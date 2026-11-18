#!/bin/bash
# Run commands one after another, logging start/exit codes. Used to queue training
# runs; concurrency is limited by the memory guard's job slots (train/memguard.py),
# so several chains can be launched at once.
#
#   nohup scripts/run_chain.sh <name> "<cmd 1>" "<cmd 2>" ... > runs/queue/<name>.log 2>&1 &
#
# Exact commands used for the results in results/ are in results/launch_commands.txt.
export PYTHONUNBUFFERED=1
command -v uv >/dev/null || export PATH=/opt/homebrew/bin:$PATH
cd "$(dirname "$0")/.."
name=$1; shift
for c in "$@"; do echo "[$(date +%T)] START $c"; eval "$c"; rc=$?; echo "[$(date +%T)] EXIT $rc $c"; done
echo "[$(date +%T)] CHAIN_DONE $name"
