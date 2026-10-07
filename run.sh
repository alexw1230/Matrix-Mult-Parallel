#!/bin/bash
#SBATCH --job-name=matrix_mult
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH -p short-96core
#SBATCH --output=mm_%j.out

module purge
module load gcc/13.2.0
module load openmpi/gcc13.2/4.1.6

set -euo pipefail

export OMPI_MCA_coll_hcoll_enable=0

cd "$SLURM_SUBMIT_DIR"

make clean && make

RES=results/${SLURM_JOB_ID}
mkdir -p "$RES"
echo "host: $(hostname)"   | tee    "$RES/env.txt"
module list 2>&1           | tee -a "$RES/env.txt"
mpicc --version 2>&1       | tee -a "$RES/env.txt"
lscpu                      >        "$RES/lscpu.txt"

MPIRUN="mpirun --bind-to core --map-by core"

reps_for() {
  case "$1" in
    1024) echo 20 ;;
    4096) echo 5  ;;
    *)    echo 0  ;;
  esac
}

for P in 1 4 8 16; do
  for N in 16 64 256; do
    $MPIRUN -np $P ./mm -n $N -r 3 -e -o "$RES/correctness.csv"
  done
done
$MPIRUN -np 4 ./mm -n 8 -r 1 -e -p -o "$RES/correctness.csv" > "$RES/example_n8_p4.txt"

for N in 64 256 1024 4096; do
  for P in 1 4 8 16; do
    $MPIRUN -np $P ./mm -n $N -r "$(reps_for $N)" \
            -o "$RES/results.csv" -w "$RES/raw_times.csv"
  done
done

for i in 1 2 3 4 5; do
  $MPIRUN -np 1 ./mm -n 256 -o "$RES/n256_repeat.csv"
done

echo "done"