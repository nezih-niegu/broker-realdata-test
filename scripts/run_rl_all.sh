#!/usr/bin/env bash
# Runs the full RL benchmark (3 seeds per model) two jobs at a time; skips finished jobs.
# Then the risk-return frontier. Re-running resumes where it stopped.
cd "$(dirname "$0")/.." && mkdir -p logs
[ -f results/rl_benchmark/Nash_closed_form_no_QI_channel.json ] || \
  python3 scripts/run_rl_benchmark.py --algo baselines > logs/baselines.log 2>&1
jobs=()
for s in 0 1 2; do jobs+=("DPS $s" "DPS-linear $s" "DPS $s --extended" "PPO $s" "PPO $s --extended" "A2C $s" "RecurrentPPO $s --steps 100000"); done
for s in 0 1 2; do jobs+=("SAC $s" "TQC $s" "TD3 $s" "DDPG $s"); done
printf '%s\n' "${jobs[@]}" | xargs -P 2 -I{} bash -c '
  set -- {}; algo=$1; seed=$2; shift 2
  ext=""; [[ "$*" == *--extended* ]] && ext="_ext"
  name="${algo}${ext}_seed${seed}"
  [ -f results/rl_benchmark/$name.json ] && exit 0
  pgrep -f "algo $algo --seed $seed $*\$" >/dev/null && exit 0
  python3 -u scripts/run_rl_benchmark.py --algo $algo --seed $seed "$@" > logs/$name.log 2>&1
  grep -v -i warn logs/$name.log | tail -n 1'
python3 -u scripts/run_rl_frontier.py > logs/frontier.log 2>&1; grep -v -i warn logs/frontier.log | grep -v epoch
