datasets=("australian" "diabetes" "german" "heart" "ionosphere" "magic04" "new-thyroid" "ringnorm" "segment" "threenorm" "tic-tac-toe" "twonorm" "waveform" "wdbc" "wine" "biodeg" "flags" "pageblocks" "sobar")

for DATASET in "${datasets[@]}"
do
	python main_npy_generator.py $DATASET
done