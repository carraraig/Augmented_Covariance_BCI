#!/usr/bin/env bash

oarsub -S /home/icarrara/Documents/Project/reduced_dataset/classification_Full_nested/Cluster/MOABB_Server_STATE_ART.sh \
    -l "/nodes=1/gpunum=1, walltime=10:00:00" \
    -p "gpu='YES' and gpucapability>='5.0'" \
    -t besteffort \
    --notify "[END, ERROR]mail:${EMAIL_ADDRESS}" \
    --array-param-file subject_DL.txt