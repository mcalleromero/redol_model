datasets=("australian" "diabetes" "german" "heart" "ionosphere" "magic04" "new-thyroid" "ringnorm" "segment" "threenorm" "tic-tac-toe" "twonorm" "waveform" "wdbc" "wine")
# datasets=("biodeg" "flags" "pageblocks" "sobar")

for DATASET in "${datasets[@]}"
do
	python method_parameter_comparison.py $DATASET
done