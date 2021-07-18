set -e
backends=( filesystem aws azure gcloud hosted )
frameworks=( catboost fastai gensim keras lightgbm pytorch pytorch-lightning sklearn tensorflow transformers xgboost)

for framework in "${frameworks[@]}"
do
	for backend in "${backends[@]}"
	do
		echo "\n 🔵  Running a $framework example in a $backend modelstore."
		python main.py --modelstore-in $backend --ml-framework $framework
		echo "\n ✅  Finished running $framework in $backend."
	done
done

