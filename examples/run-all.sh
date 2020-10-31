set -e
set -x

root=$(pwd)

for dir in examples-by-ml-library/*/
do
    cd $root/$dir
    make refresh
    make run
done

for dir in examples-by-storage/*/
do
    cd $root/$dir
    make refresh
    make run
done

cd $root

