function run {
	cd $1
	make pyenv-local
	make run
	cd ..
}

run catboost
run keras
run pytorch
run sklearn
run transformers
run xgboost

