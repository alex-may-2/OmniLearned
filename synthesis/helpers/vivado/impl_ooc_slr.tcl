# Usage: run from <project>/vivado_ooc after synth_ooc.tcl: vivado -mode batch -nojournal -source impl_ooc_slr.tcl
# Used for run-log row VIMPL-H5 (2026-11-19, failed on routing congestion).
# Out-of-context place and route of H5 from post_synth.dcp, confined to one SLR (SLR1) with a hard pblock.
set_param general.maxThreads 8
open_checkpoint post_synth.dcp
create_pblock pb_slr
resize_pblock pb_slr -add [get_slrs SLR1]
add_cells_to_pblock pb_slr -top
set_property IS_SOFT FALSE [get_pblocks pb_slr]
set_property CONTAIN_ROUTING TRUE [get_pblocks pb_slr]
opt_design
place_design
phys_opt_design
write_checkpoint -force post_place.dcp
report_timing_summary -max_paths 20 -file post_place_timing.rpt
route_design
phys_opt_design
write_checkpoint -force post_route.dcp
report_utilization -file post_route_util.rpt
report_utilization -pblocks [get_pblocks pb_slr] -file post_route_util_pblock.rpt
report_utilization -slr -file post_route_util_slr.rpt
report_timing_summary -max_paths 20 -file post_route_timing.rpt
report_route_status -file post_route_status.rpt
report_drc -file post_route_drc.rpt
