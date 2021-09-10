set -e
backends=( filesystem aws azure gcloud hosted )
frameworks=( annoy catboost fastai gensim keras lightgbm file pytorch pytorch-lightning sklearn tensorflow transformers xgboost )

for framework in "${frameworks[@]}"
do
	for backend in "${backends[@]}"
	do
		echo "\n 🔵  Running the $framework example in a $backend modelstore."
		python main.py --modelstore-in $backend --ml-framework $framework
		echo "\n ✅  Finished running the $framework example in $backend."
	done
done

