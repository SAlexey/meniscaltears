listBestModels(){
	for each in $( find $1 -iname best_*model.json )
	do
		echo $each "$(cat $each)" 
	done 
}
