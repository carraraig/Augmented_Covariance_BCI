#!/usr/bin/env bash

oarsub -S /home/icarrara/Documents/Project/reduced_dataset/classification_Full_nested/Cluster/MOABB_Server_STATE_ART.sh \
    -l "nodes=1/core=16,walltime=10:00:00" \
    -p "cluster='dellc6420'" \
    -t besteffort \
    --notify "[END, ERROR]mail:${EMAIL_ADDRESS}" \
    --array-param-file subject.txt