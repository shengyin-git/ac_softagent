#!/bin/bash

time=$(date "+%Y%m%d%H%M%S")
echo $time

for((i=1;i<=1;i++));
do
echo "i = $i";
python run_cem_rope_flattening.py --test_num $i --log_dir "data/simp_action/cem/$time/";
done

