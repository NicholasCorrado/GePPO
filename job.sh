#!/bin/bash

# have job exit if any command returns with non-zero exit status (aka failure)
set -e

CODENAME=GePPO
CODEDIR=$CODENAME
ENVDIR=$CODEDIR/$ENVNAME
export PATH

cp /staging/ncorrado/${CODENAME}.tar.gz .
tar -xzf ${CODENAME}.tar.gz -C .
rm ${CODENAME}.tar.gz # remove code tarball
cd $CODENAME

# install code
source install.sh

pid=${1}
step=${2}
command_fragment=`tr '*' ' ' <<< $3`
echo $command

#pip install -e custom-envs
#pip install wandb
#pip install tensorboard
#export WANDB_CONFIG_DIR=$(pwd)/.config/wandb

cd geppo
python3 -u ${command_fragment} --run-id ${step} --seed ${step}
#$($command --seed $step --run-id $step)
#$($command)

tar -czvf results_${pid}.tar.gz results/*
mv results_${pid}.tar.gz ../..

cd ../..
rm -rf $CODENAME
