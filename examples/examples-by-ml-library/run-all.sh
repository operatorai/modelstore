set -e
backends=( filesystem aws-s3 google-cloud-storage azure-container minio )
frameworks=( annoy catboost causalml fastai file gensim keras lightgbm \
	onnx-sklearn onnx-lightgbm prophet pyspark pytorch pytorch-lightning \
	sklearn sklearn-with-explainer sklearn-with-extras skorch xgboost xgboost-booster \
	tensorflow hf-distilbert hf-gpt2-pt hf-gpt2-tf segment-anything yolov5 )

for framework in "${frameworks[@]}"
do
	for backend in "${backends[@]}"
	do
		echo -e "\n 🔵  Running the $framework example in a $backend modelstore."
		uv run python main.py --modelstore-in $backend --ml-framework $framework
		echo -e "\n ✅  Finished running the $framework example in $backend."
	done
done
