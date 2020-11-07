function run {
	cd $1
	make pyenv-test
	make run
	cd ..
}

run aws
run filesystem
run gcloud

