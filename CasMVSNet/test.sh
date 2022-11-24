#!/usr/bin/env bash
TESTPATH="/data3/zyf/dataset/dtu"
TESTLIST="lists/dtu/test.txt"
CKPT_FILE=$1
python test.py --dataset=general_eval --batch_size=1 --testpath=$TESTPATH  --testlist=$TESTLIST --loadckpt $CKPT_FILE ${@:2}
