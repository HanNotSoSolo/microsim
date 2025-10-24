#!/bin/bash
#SBATCH --job-name=yuka_2D
#SBATCH --time=3-00:00
#SBATCH --mem=150G
#SBATCH --qos=c3_long_opa
#SBATCH --output=%j.spheres_2D_yukawa.out
#SBATCH --profile=all

echo "Starting Yukawa calculation..."
srun python /scratchm/mdellava/microsim/parallel_hollowCylinders_2D/yukawa_gravity.py
