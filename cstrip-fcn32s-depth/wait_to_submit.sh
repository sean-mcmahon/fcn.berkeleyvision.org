#!/bin/bash -l
timeatstart=`date +%Y-%m-%d_%H-%M-%S`
echo "waiting to start training hha FCN at $timeatstart..."
sleep 4h
submit_date=`date +%Y-%m-%d_%H-%M-%S`
echo "submitting job at $submit_date"
qsub train_bash
