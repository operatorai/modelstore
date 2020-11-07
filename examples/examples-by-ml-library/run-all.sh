function run {
	cd $1
	make pyenv-test
	make run
	cd ..
}

run catboost
run keras
run pytorch
run sklearn
run tensorflow
run transformers
run xgboost
