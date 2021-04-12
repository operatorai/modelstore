function run {
	cd $1
	make pyenv-test
	make run
	cd ..
}

set -e
run aws
run azure
run filesystem
run gcloud
run hosted

