#!/bin/bash

VAL_SEQ_FILE=/srv/glusterfs/kmaninis/Databases/Boundary_Detection/DAVIS/gt_sets/val_categories.txt
for i in $(cat $VAL_SEQ_FILE)
	do
		qsub -N PyOSVOS_$i script.sh $i $1
	done
