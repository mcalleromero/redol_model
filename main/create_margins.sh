#!/bin/bash

# "zeros0.2_unos0.8_ringnorm_normal_2dimm" "zeros0.3_unos0.7_ringnorm_normal_2dimm" "zeros0.4_unos0.6_ringnorm_normal_2dimm"
# "zeros0.5_unos0.5_ringnorm_normal_2dimm" "zeros0.6_unos0.4_ringnorm_normal_2dimm" "zeros0.7_unos0.3_ringnorm_normal_2dimm"
# 
# arr=( "zeros0.8_unos0.2_ringnorm_normal_2dimm" "zeros0.9_unos0.1_ringnorm_normal_2dimm" )

arr=( "australian" "diabetes" "german" "heart" "ionosphere" "iris" "magic04" "new-thyroid" "ringnorm" "threenorm" "tic-tac-toe" "twonorm" "waveform" "wdbc" "wine" )

for i in "${arr[@]}"
do
	python3 get_margins.py $i
done
