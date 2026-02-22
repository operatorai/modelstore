set -e

DOCKER_IMAGE="${DOCKER_IMAGE:-modelstore-examples}"

echo -e "\n 🔵  Building Docker image: $DOCKER_IMAGE"
docker build -f Dockerfile -t "$DOCKER_IMAGE" .
echo -e "\n ✅  Docker image built."

frameworks=( annoy catboost causalml fastai file gensim keras lightgbm \
	onnx-sklearn onnx-lightgbm prophet pyspark pytorch pytorch-lightning \
	sklearn sklearn-with-explainer sklearn-with-extras skorch xgboost xgboost-booster \
	tensorflow hf-distilbert hf-gpt2-pt hf-gpt2-tf segment-anything yolov5 )

for framework in "${frameworks[@]}"
do
	echo -e "\n 🔵  Running the $framework example in a filesystem modelstore."
	docker run --rm \
		-e MODEL_STORE_ROOT_PREFIX=/tmp/modelstore \
		$DOCKER_IMAGE --modelstore-in filesystem --ml-framework $framework
	echo -e "\n ✅  Finished running the $framework example in filesystem."

	echo -e "\n 🔵  Running the $framework example in an aws-s3 modelstore."
	docker run --rm \
		-e MODEL_STORE_AWS_BUCKET=$MODEL_STORE_AWS_BUCKET \
		-e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
		-e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
		$DOCKER_IMAGE --modelstore-in aws-s3 --ml-framework $framework
	echo -e "\n ✅  Finished running the $framework example in aws-s3."

	echo -e "\n 🔵  Running the $framework example in a google-cloud-storage modelstore."
	docker run --rm \
		-e MODEL_STORE_GCP_PROJECT=$MODEL_STORE_GCP_PROJECT \
		-e MODEL_STORE_GCP_BUCKET=$MODEL_STORE_GCP_BUCKET \
		-e GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcloud-key.json \
		-v $GOOGLE_APPLICATION_CREDENTIALS:/tmp/gcloud-key.json:ro \
		$DOCKER_IMAGE --modelstore-in google-cloud-storage --ml-framework $framework
	echo -e "\n ✅  Finished running the $framework example in google-cloud-storage."

	echo -e "\n 🔵  Running the $framework example in an azure-container modelstore."
	docker run --rm \
		-e MODEL_STORE_AZURE_CONTAINER=$MODEL_STORE_AZURE_CONTAINER \
		-e AZURE_STORAGE_CONNECTION_STRING=$AZURE_STORAGE_CONNECTION_STRING \
		$DOCKER_IMAGE --modelstore-in azure-container --ml-framework $framework
	echo -e "\n ✅  Finished running the $framework example in azure-container."

	echo -e "\n 🔵  Running the $framework example in a minio modelstore."
	docker run --rm \
		-e MODEL_STORE_AWS_BUCKET=$MODEL_STORE_AWS_BUCKET \
		-e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
		-e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
		$DOCKER_IMAGE --modelstore-in minio --ml-framework $framework
	echo -e "\n ✅  Finished running the $framework example in minio."
done
