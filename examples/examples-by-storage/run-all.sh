set -e
backends=( filesystem aws azure gcloud )

for backend in "${backends[@]}"
do
	echo "\n 🔵  Running the $backend example."
	python main.py --modelstore-in $backend
	echo "\n ✅  Finished running the $backend example."
done

