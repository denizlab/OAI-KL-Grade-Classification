#!bin/bash
echo "Run All Experiment ..."
#bsub < submit_baseline.lsf
#bsub < submit_baseline_unsup.lsf
#bsub < submit_baselinefl_unsup.lsf
bsub < submit_baseline_fl.lsf
#bsub < submit_cbam.lsf
#bsub < submit_cbam_unsup.lsf
bsub < submit_cbamfl.lsf
#bsub < submit_cbamfl_unsup.lsf
#bsub < submit_default.lsf
#bsub < submit_default_cbam.lsf
#bsub < submit_default_fl.lsf
#bsub < submit_default_cbamfl.lsf


