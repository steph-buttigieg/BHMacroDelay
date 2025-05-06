#!/bin/bash -l 

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=28
#SBATCH -J generate_files
#SBATCH -o ../slurm_output/delays.out
#SBATCH -e ../slurm_output/delays.err
#SBATCH -p cosma7
#SBATCH -A dp379
#SBATCH -t 3:00:00
#SBATCH --mail-type=BEGIN,END,FAIL # notifications for job done & fail
#SBATCH --mail-user=sb2583@cam.ac.uk


module purge
#load the modules used to build your program.
module load python/3.12.4
module load gnu_comp/14.1.0
module load openmpi

export OMPI_MCA_btl =^uct

source /cosma/apps/dp012/dc-butt3/astroenv/bin/activate

#Run the program, note $SLURM_ARRAY_TASK_ID is set to the array number.

mpiexec -n $SLURM_NTASKS python ../src/macrophysical_delays.py

