#/bin/bash

for i in {1..27..2}
do
	j=`expr $i + 1`
	echo $i $j
	./build/run_cpp_sample -c ./config.json -i ./palm_test/$i.png ./palm_test/$j.png
done
