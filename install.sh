#!/bin/bash

ENVNAME=ppo-ros
ENVDIR=${ENVNAME}_env
cp /staging/ncorrado/${ENVNAME}.tar.gz .

mkdir $ENVDIR
tar -xzf ${ENVNAME}.tar.gz -C $ENVDIR
rm ${ENVNAME}.tar.gz # remove env tarball
source $ENVDIR/bin/activate

pip install seaborn tensorflow scipy gurobipy