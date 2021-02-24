function run {
	cd $1
	make pyenv-test
	make run
	cd ..
}

set -e
run catboost
run fastai
run keras
run pytorch
run sklearn
run tensorflow
run transformers
run xgboost
