from pathlib import Path
R=Path(__file__).resolve().parents[2]/'rtl-log-work'
p=R/'rtl/vxm/lpu_vxm_tile_execution.sv';s=p.read_text()
# Retain the legacy metadata slot for other positions. C3's Repeat metadata
# follows the same fixed eight-edge pipeline as the public execution stage.
s=s.replace('inflight_repeat_q','single_inflight_repeat_q').replace('inflight_accumulator_write_q','single_inflight_accumulator_write_q')
pos=s.index('  logic pending_end_marker_q;')
s=s[:pos]+'''  logic [STAGES*VXM_REPEAT_CONTROL_WIDTH-1:0] inflight_repeat_q;
  logic [STAGES-1:0] inflight_accumulator_write_q;
  logic [3:0] inflight_count_q[0:STAGES-1];
  logic [STAGES-1:0] stage_issue_allowed;
'''+s[pos:]
# All reads use aligned metadata, while old sequential writes keep targeting
# single_* and are optimized away on C3.
comb_end=s.index('  always_ff @(posedge clk_i or negedge rst_ni)')
s=s[:comb_end].replace('!single_inflight_accumulator_write_q','!inflight_accumulator_write_q').replace('single_inflight_repeat_q[stage*','inflight_repeat_q[stage*')+s[comb_end:]
# Sequential consumers at retirement must read aligned output tags too.
s=s.replace('single_inflight_accumulator_write_q[source_stage]','inflight_accumulator_write_q[source_stage]')
s=s.replace('single_inflight_repeat_q[\n                source_stage*','inflight_repeat_q[\n                source_stage*')
pos=s.index('  always_ff @(posedge clk_i or negedge rst_ni)')
s=s[:pos]+'''  // Track busy independently from ready: an II=1 ALU stays ready while
  // requests are in flight, so ready can no longer serve as an idle flag.
  for(genvar tag_stage=0;tag_stage<STAGES;tag_stage++) begin:g_retirement_tags
    if(tag_stage%4==3) begin:g_c3
      logic [VXM_REPEAT_CONTROL_WIDTH-1:0] repeat_q[0:7];
      logic [7:0] accumulator_q,valid_q;
      always_ff @(posedge clk_i or negedge rst_ni) begin
        if(!rst_ni) begin
          accumulator_q<=0;valid_q<=0;
          for(integer i=0;i<8;i++)repeat_q[i]<=0;
        end else begin
          valid_q<={valid_q[6:0],stage_accept[tag_stage]};
          accumulator_q<={accumulator_q[6:0],stage_accept[tag_stage] &&
            accumulator_write_for(tag_stage,
              pending_instruction_q[tag_stage*VXM_LOCAL_MAX_INSTRUCTION_WIDTH+:VXM_LOCAL_MAX_INSTRUCTION_WIDTH],
              pending_repeat_q[tag_stage*VXM_REPEAT_CONTROL_WIDTH+:VXM_REPEAT_CONTROL_WIDTH])};
          if(stage_accept[tag_stage]) repeat_q[0]<=pending_repeat_q[
            tag_stage*VXM_REPEAT_CONTROL_WIDTH+:VXM_REPEAT_CONTROL_WIDTH];
          for(integer i=1;i<8;i++) if(valid_q[i-1])repeat_q[i]<=repeat_q[i-1];
        end
      end
      assign inflight_repeat_q[tag_stage*VXM_REPEAT_CONTROL_WIDTH+:VXM_REPEAT_CONTROL_WIDTH]=repeat_q[7];
      assign inflight_accumulator_write_q[tag_stage]=accumulator_q[7];
    end else begin:g_single
      assign inflight_repeat_q[tag_stage*VXM_REPEAT_CONTROL_WIDTH+:VXM_REPEAT_CONTROL_WIDTH]=
        single_inflight_repeat_q[tag_stage*VXM_REPEAT_CONTROL_WIDTH+:VXM_REPEAT_CONTROL_WIDTH];
      assign inflight_accumulator_write_q[tag_stage]=single_inflight_accumulator_write_q[tag_stage];
    end
    always_ff @(posedge clk_i or negedge rst_ni) begin
      if(!rst_ni)inflight_count_q[tag_stage]<=0;
      else case({stage_accept[tag_stage],stage_result_all[tag_stage]})
        2'b10:inflight_count_q[tag_stage]<=inflight_count_q[tag_stage]+1'b1;
        2'b01:inflight_count_q[tag_stage]<=inflight_count_q[tag_stage]-1'b1;
        default:begin end
      endcase
    end
    // Accumulator read-after-write dependencies still serialize. Independent
    // special functions do not wait for older independent elements.
    assign stage_issue_allowed[tag_stage]=inflight_count_q[tag_stage]==0 ||
      !accumulator_write_for(tag_stage,
        pending_instruction_q[tag_stage*VXM_LOCAL_MAX_INSTRUCTION_WIDTH+:VXM_LOCAL_MAX_INSTRUCTION_WIDTH],
        pending_repeat_q[tag_stage*VXM_REPEAT_CONTROL_WIDTH+:VXM_REPEAT_CONTROL_WIDTH]);
  end

'''+s[pos:]
s=s.replace('pending_valid_q[stage] && config_loaded_q)', 'pending_valid_q[stage] && config_loaded_q && stage_issue_allowed[stage])')
# Generate variable is named stage; verify this specific binding changed.
assert 'config_loaded_q && stage_issue_allowed[stage]' in s
s=s.replace('idle_o = !(|pending_valid_q)', 'idle_o = c3_pipeline_idle && !(|pending_valid_q)')
pos=s.index('  // Track busy independently')
s=s[:pos]+'''  logic c3_pipeline_idle;
  always_comb begin
    c3_pipeline_idle=1'b1;
    for(integer busy_stage=0;busy_stage<STAGES;busy_stage++)
      if(inflight_count_q[busy_stage]!=0)c3_pipeline_idle=1'b0;
  end

'''+s[pos:]
p.write_text(s)
print('Aligned Tile retirement metadata, busy tracking and accumulator hazards.')
