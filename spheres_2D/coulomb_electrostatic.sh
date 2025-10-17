#!/bin/bash
#SBATCH --job-name=culom_2D
#SBATCH --time=2-00:00
#SBATCH --mem=150G
#SBATCH --qos=c3_long_opa
#SBATCH --output=%j.spheres_2D_coulomb.out
#SBATCH --profile=all

echo "Starting Coulomb calculation..."
srun python /scratchm/mdellava/microsim/spheres_2D/coulomb_electrostatic.py
wait
