set -e
backends=( filesystem aws azure gcloud )
frameworks=( annoy catboost fastai file gensim keras lightgbm \
	mxnet onnx-sklearn onnx-lightgbm prophet pytorch pytorch-lightning sklearn \
	sklearn-with-explainer sklearn-with-extras skorch \
	tensorflow transformers xgboost xgboost-booster )

for framework in "${frameworks[@]}"
do
	for backend in "${backends[@]}"
	do
		echo "\n 🔵  Running the $framework example in a $backend modelstore."
		python main.py --modelstore-in $backend --ml-framework $framework
		echo "\n ✅  Finished running the $framework example in $backend."
	done
done
