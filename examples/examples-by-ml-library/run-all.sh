set -e
backends=( filesystem aws azure gcloud hosted )
frameworks=( catboost fastai gensim keras lightgbm pytorch pytorch-lightning sklearn tensorflow transformers xgboost)

for framework in "${frameworks[@]}"
do
	for backend in "${backends[@]}"
	do
		python main.py --modelstore-in $backend --ml-framework $framework
		echo "\n"
	done
done

