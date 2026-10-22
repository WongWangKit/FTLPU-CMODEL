#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cmodel_dir=${FTLPU_CMODEL_DIR:-"$repo_dir/../FTLPU-CMODEL"}
vector_build_dir="$repo_dir/build/cmodel_vectors"
repo_tmp_dir="$repo_dir/build/tmp"
generator="$vector_build_dir/mem_stream_roundtrip"
sxm_generator="$vector_build_dir/sxm_local_transpose"
wavefront_generator="$vector_build_dir/sxm_wavefront_transpose"
vxm_generator="$vector_build_dir/vxm_int8_add"
vxm_fp16_generator="$vector_build_dir/vxm_fp16_relu"
vxm_fp32_arithmetic_generator="$vector_build_dir/vxm_fp32_arithmetic"
mxm_generator="$vector_build_dir/mxm_vector_identity"
mxm_accumulator_generator="$vector_build_dir/mxm_accumulator"
mxm_column_generator="$vector_build_dir/mxm_column_direct16"
mxm_int8_generator="$vector_build_dir/mxm_int8_dequant"
mxm_block8_generator="$vector_build_dir/mxm_block8"
smollm2_ffn_generator="$vector_build_dir/smollm2_ffn"

mkdir -p "$vector_build_dir" "$repo_tmp_dir"
export TMPDIR="$repo_tmp_dir"
export TEMP="$repo_tmp_dir"
export TMP="$repo_tmp_dir"

g++ -std=c++20 -O2 \
  -I "$cmodel_dir/include" \
  "$repo_dir/sim/cmodel/mem_stream_roundtrip.cpp" \
  "$cmodel_dir/src/mem/sram.cpp" \
  "$cmodel_dir/src/mxm/accumulator.cpp" \
  "$cmodel_dir/src/mxm/block_accumulator.cpp" \
  -o "$generator"

"$generator" \
  "$repo_dir/sim/vectors/mem_stream_roundtrip.hex" \
  "$repo_dir/sim/vectors/mem_stream_schedule.hex"

g++ -std=c++20 -O2 \
  -I "$cmodel_dir/include" \
  "$repo_dir/sim/cmodel/sxm_local_transpose.cpp" \
  "$cmodel_dir/src/mem/sram.cpp" \
  "$cmodel_dir/src/mxm/accumulator.cpp" \
  "$cmodel_dir/src/mxm/block_accumulator.cpp" \
  -o "$sxm_generator"

"$sxm_generator" \
  "$repo_dir/sim/vectors/sxm_local_transpose_init.hex" \
  "$repo_dir/sim/vectors/sxm_local_transpose_golden.hex" \
  "$repo_dir/sim/vectors/sxm_local_transpose_schedule.hex"

g++ -std=c++20 -O2 \
  -I "$cmodel_dir/include" \
  "$repo_dir/sim/cmodel/sxm_wavefront_transpose.cpp" \
  "$cmodel_dir/src/mem/sram.cpp" \
  "$cmodel_dir/src/mxm/accumulator.cpp" \
  "$cmodel_dir/src/mxm/block_accumulator.cpp" \
  -o "$wavefront_generator"

"$wavefront_generator" \
  "$repo_dir/sim/vectors/sxm_wavefront_transpose_init.hex" \
  "$repo_dir/sim/vectors/sxm_wavefront_transpose_golden.hex" \
  "$repo_dir/sim/vectors/sxm_wavefront_transpose_schedule.hex"

g++ -std=c++20 -O2 \
  -I "$cmodel_dir/include" \
  "$repo_dir/sim/cmodel/vxm_int8_add.cpp" \
  "$cmodel_dir/src/mem/sram.cpp" \
  "$cmodel_dir/src/mxm/accumulator.cpp" \
  "$cmodel_dir/src/mxm/block_accumulator.cpp" \
  -o "$vxm_generator"

"$vxm_generator" \
  "$repo_dir/sim/vectors/vxm_int8_add_init.hex" \
  "$repo_dir/sim/vectors/vxm_int8_add_golden.hex" \
  "$repo_dir/sim/vectors/vxm_int8_add_schedule.hex"

g++ -std=c++20 -O2 \
  -I "$cmodel_dir/include" \
  "$repo_dir/sim/cmodel/vxm_fp16_relu.cpp" \
  "$cmodel_dir/src/mem/sram.cpp" \
  "$cmodel_dir/src/mxm/accumulator.cpp" \
  "$cmodel_dir/src/mxm/block_accumulator.cpp" \
  -o "$vxm_fp16_generator"

"$vxm_fp16_generator" \
  "$repo_dir/sim/vectors/vxm_fp16_relu_init.hex" \
  "$repo_dir/sim/vectors/vxm_fp16_relu_golden.hex" \
  "$repo_dir/sim/vectors/vxm_fp16_relu_schedule.hex"

g++ -std=c++20 -O2 \
  -I "$cmodel_dir/include" \
  "$repo_dir/sim/cmodel/vxm_fp32_arithmetic.cpp" \
  "$cmodel_dir/src/mem/sram.cpp" \
  "$cmodel_dir/src/mxm/accumulator.cpp" \
  "$cmodel_dir/src/mxm/block_accumulator.cpp" \
  -o "$vxm_fp32_arithmetic_generator"

"$vxm_fp32_arithmetic_generator" \
  "$repo_dir/sim/vectors/vxm_fp32_arithmetic_init.hex" \
  "$repo_dir/sim/vectors/vxm_fp32_arithmetic_golden.hex" \
  "$repo_dir/sim/vectors/vxm_fp32_arithmetic_schedule.hex"

g++ -std=c++20 -O2 \
  -I "$cmodel_dir/include" \
  "$repo_dir/sim/cmodel/mxm_vector_identity.cpp" \
  "$cmodel_dir/src/mem/sram.cpp" \
  "$cmodel_dir/src/mxm/accumulator.cpp" \
  "$cmodel_dir/src/mxm/block_accumulator.cpp" \
  -o "$mxm_generator"

"$mxm_generator" \
  "$repo_dir/sim/vectors/mxm_vector_identity_init.hex" \
  "$repo_dir/sim/vectors/mxm_vector_identity_golden.hex" \
  "$repo_dir/sim/vectors/mxm_vector_identity_schedule.hex"

"$mxm_generator" \
  "$repo_dir/sim/vectors/mxm_vector_fp16_init.hex" \
  "$repo_dir/sim/vectors/mxm_vector_fp16_golden.hex" \
  "$repo_dir/sim/vectors/mxm_vector_fp16_schedule.hex" \
  --fp16

g++ -std=c++20 -O2 \
  -I "$cmodel_dir/include" \
  "$repo_dir/sim/cmodel/mxm_accumulator.cpp" \
  "$cmodel_dir/src/mem/sram.cpp" \
  "$cmodel_dir/src/mxm/accumulator.cpp" \
  "$cmodel_dir/src/mxm/block_accumulator.cpp" \
  -o "$mxm_accumulator_generator"

"$mxm_accumulator_generator" \
  "$repo_dir/sim/vectors/mxm_accumulator_init.hex" \
  "$repo_dir/sim/vectors/mxm_accumulator_golden.hex" \
  "$repo_dir/sim/vectors/mxm_accumulator_schedule.hex"

g++ -std=c++20 -O2 \
  -I "$cmodel_dir/include" \
  "$repo_dir/sim/cmodel/mxm_column_direct16.cpp" \
  "$cmodel_dir/src/mem/sram.cpp" \
  "$cmodel_dir/src/mxm/accumulator.cpp" \
  "$cmodel_dir/src/mxm/block_accumulator.cpp" \
  -o "$mxm_column_generator"

"$mxm_column_generator" \
  "$repo_dir/sim/vectors/mxm_column_direct16_init.hex" \
  "$repo_dir/sim/vectors/mxm_column_direct16_golden.hex" \
  "$repo_dir/sim/vectors/mxm_column_direct16_schedule.hex"

g++ -std=c++20 -O2 \
  -I "$cmodel_dir/include" \
  "$repo_dir/sim/cmodel/mxm_int8_dequant.cpp" \
  "$cmodel_dir/src/mem/sram.cpp" \
  "$cmodel_dir/src/mxm/accumulator.cpp" \
  "$cmodel_dir/src/mxm/block_accumulator.cpp" \
  -o "$mxm_int8_generator"

"$mxm_int8_generator" \
  "$repo_dir/sim/vectors/mxm_int8_dequant_init.hex" \
  "$repo_dir/sim/vectors/mxm_int8_dequant_golden.hex" \
  "$repo_dir/sim/vectors/mxm_int8_dequant_schedule.hex"

g++ -std=c++20 -O2 \
  -I "$cmodel_dir/include" \
  "$repo_dir/sim/cmodel/mxm_block8.cpp" \
  "$cmodel_dir/src/mem/sram.cpp" \
  "$cmodel_dir/src/mxm/accumulator.cpp" \
  "$cmodel_dir/src/mxm/block_accumulator.cpp" \
  -o "$mxm_block8_generator"

"$mxm_block8_generator" \
  "$repo_dir/sim/vectors/mxm_block8_init.hex" \
  "$repo_dir/sim/vectors/mxm_block8_golden.hex" \
  "$repo_dir/sim/vectors/mxm_block8_schedule.hex"

g++ -std=c++20 -O2 \
  -I "$cmodel_dir/include" -I "$repo_dir/sim/cmodel" \
  "$repo_dir/sim/cmodel/smollm2_ffn.cpp" \
  "$cmodel_dir/src/mem/sram.cpp" \
  "$cmodel_dir/src/mxm/accumulator.cpp" \
  "$cmodel_dir/src/mxm/block_accumulator.cpp" \
  -o "$smollm2_ffn_generator"

"$smollm2_ffn_generator" \
  "$repo_dir/sim/vectors/smollm2_ffn_init.hex" \
  "$repo_dir/sim/vectors/smollm2_ffn_golden.hex" \
  "$repo_dir/sim/vectors/smollm2_ffn_gate_schedule.hex" \
  "$repo_dir/sim/vectors/smollm2_ffn_swiglu_schedule.hex" \
  "$repo_dir/sim/vectors/smollm2_ffn_down_schedule.hex"
