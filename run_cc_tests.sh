#! /bin/bash

#run the generated datasets
search_dir=new_models
for entry in "$search_dir"/*.txt
do
	echo "reserve ./main benchspeed 21           9               1               1             1           0     1           1 nocleary $entry"
	reserve ./main benchspeed 21           9               1               1             100           0     1           1 nocleary "$entry"
done
#Run the random dataset
reserve ./main benchspeed 21           9               1               1             100           0     1           1 nocleary