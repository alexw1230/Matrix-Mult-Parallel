#!/bin/bash
#SBATCH --job-name=matrix_mult
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH -p short-96core
#SBATCH --output=mm_%j.out

set -euo pipefail

module purge
module load gcc/13.2.0
module load openmpi/gcc13.2/4.1.6

cd "$SLURM_SUBMIT_DIR"

make clean && make

mkdir -p results
echo "host: $(hostname)"   | tee    results/env_${SLURM_JOB_ID}.txt
module list 2>&1           | tee -a results/env_${SLURM_JOB_ID}.txt
mpicc --version 2>&1       | tee -a results/env_${SLURM_JOB_ID}.txt
lscpu                      >        results/lscpu_${SLURM_JOB_ID}.txt

MPIRUN="mpirun --bind-to core --map-by core"

for P in 1 4 8 16; do
  for N in 16 64 256; do
    $MPIRUN -np $P ./mm -n $N -r 3 -e -o results/correctness.csv
  done
done

$MPIRUN -np 4 ./mm -n 8 -r 1 -e -p -o results/correctness.csv > results/example_n8_p4.txt

for N in 64 256 1024 4096; do
  for P in 1 4 8 16; do
    $MPIRUN -np $P ./mm -n $N -o results/results.csv -w results/raw_times.csv
  done
done

echo "done"
