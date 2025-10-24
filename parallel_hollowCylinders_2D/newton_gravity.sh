#!/bin/bash
#SBATCH --job-name=newt_2D
#SBATCH --time=3-00:00
#SBATCH --mem=120G
#SBATCH --qos=c3_long_opa
#SBATCH --output=%j.spheres_2D_newton.out
#SBATCH --profile=all

echo "Starting Newton calculation..."
srun python /scratchm/mdellava/microsim/parallel_hollowCylinders_2D/newton_gravity.py

echo "Launching Yukawa calculation script and exiting."
sbatch /scratchm/mdellava/microsim/parallel_hollowCylinders_2D/yukawa_gravity.sh
