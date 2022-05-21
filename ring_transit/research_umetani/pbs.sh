#! /bin/sh
#PBS -r y
#PBS -m abe
#PBS -q mid
#PBS -o Log.out
#PBS -e Log.err
#PBS -N MyPBSjob
#PBS -M umetani-tsubasa@ed.tmu.ac.jp

# Go to this job's working directory
cd $PBS_O_WORKDIR

# Run your executable
export PATH=$PATH:/home/umetanitb/anaconda3/bin
python3 ring_planet.py
