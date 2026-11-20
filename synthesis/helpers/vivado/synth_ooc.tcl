# Usage: mkdir <project>/vivado_ooc, copy this file, clock.xdc and <project>/deepsets_prj/solution1/syn/verilog/*.dat there, cd there, then
#   source <Vivado 2024.1>/settings64.sh; vivado -mode batch -nojournal -source synth_ooc.tcl
# Used for run-log row VSYN-H5 (2026-11-19).
# Out-of-context Vivado synthesis of the H5 HLS RTL (top I/O is 816 bits, so no pin placement).
set_param general.maxThreads 8
set rtl ../deepsets_prj/solution1/syn/verilog
read_verilog [glob $rtl/*.v]
add_files [glob $rtl/*.dat]
read_xdc -mode out_of_context clock.xdc
synth_design -top deepsets -part xcvu13p-flga2577-2-e -mode out_of_context

write_checkpoint -force post_synth.dcp
report_utilization -file post_synth_util.rpt
report_utilization -hierarchical -hierarchical_depth 2 -file post_synth_util_hier.rpt
report_timing_summary -max_paths 20 -file post_synth_timing.rpt
