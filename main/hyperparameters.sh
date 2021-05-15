datasets=("australian" "diabetes" "german" "heart" "ionosphere" "magic04")
#datasets=("new-thyroid" "ringnorm" "segment" "threenorm")
#datasets=("tic-tac-toe" "twonorm" "waveform" "wdbc" "wine")

for DATASET in "${datasets[@]}"
do
	python main_to_check_hyperparameters.py $DATASET
done