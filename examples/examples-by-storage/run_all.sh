function run {
	cd $1
	make pyenv-local
	make run
	cd ..
}

run aws
run filesystem
run gcloud

