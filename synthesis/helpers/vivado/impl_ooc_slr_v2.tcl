# Usage: mkdir <project>/vivado_ooc/impl_<tag>, cd there, then
#   vivado -mode batch -nojournal -source <path>/impl_ooc_slr_v2.tcl -tclargs <period ns>
# Used for run-log rows VIMPL2-H5-2p780 and VIMPL2-H5-3p125 (2026-11-19, both failed on routing congestion).
# Retry of one-SLR OOC place and route with congestion directives. Usage: -tclargs <clock period ns>.
# Run from its own directory; reads ../post_synth.dcp.
set period [lindex $argv 0]
set_param general.maxThreads 8
open_checkpoint ../post_synth.dcp
create_clock -name ap_clk -period $period [get_ports ap_clk]
create_pblock pb_slr
resize_pblock pb_slr -add [get_slrs SLR1]
add_cells_to_pblock pb_slr -top
set_property IS_SOFT FALSE [get_pblocks pb_slr]
opt_design -directive Explore
place_design -directive AltSpreadLogic_high
phys_opt_design -directive AggressiveExplore
write_checkpoint -force post_place.dcp
report_timing_summary -max_paths 20 -file post_place_timing.rpt
route_design -directive AlternateCLBRouting
phys_opt_design -directive AggressiveExplore
write_checkpoint -force post_route.dcp
report_utilization -file post_route_util.rpt
report_utilization -slr -file post_route_util_slr.rpt
report_timing_summary -max_paths 20 -file post_route_timing.rpt
report_route_status -file post_route_status.rpt
report_drc -file post_route_drc.rpt
