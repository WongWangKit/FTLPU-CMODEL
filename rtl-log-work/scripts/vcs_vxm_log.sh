#!/usr/bin/env bash
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"
mkdir -p build/vcs/vxm_log
python3 sim/cmodel/generate_log_vectors.py
vcs -full64 -sverilog -timescale=1ns/1ps -top lpu_vxm_log_tb \
  -f sim/filelist_vxm_log.f -Mdir=build/vcs/vxm_log/csrc \
  -o build/vcs/vxm_log/simv -l build/vcs/vxm_log/compile.log
build/vcs/vxm_log/simv -no_save -l build/vcs/vxm_log/run.log
