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

# Silence the harmless "unable to open mca_coll_hcoll" warnings
export OMPI_MCA_coll_hcoll_enable=0

cd "$SLURM_SUBMIT_DIR"

make clean && make

# One folder per job, so reruns never append to old results
RES=results/${SLURM_JOB_ID}
mkdir -p "$RES"
echo "host: $(hostname)"   | tee    "$RES/env.txt"
module list 2>&1           | tee -a "$RES/env.txt"
mpicc --version 2>&1       | tee -a "$RES/env.txt"
lscpu                      >        "$RES/lscpu.txt"
lscpu -e=CPU,CORE,SOCKET,NODE,CACHE > "$RES/cpu_map.txt"
# CPU frequency scaling setup (helps explain the ~2 s slow start of each run)
{ echo "cpufreq driver:   $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver 2>/dev/null || echo n/a)"
  echo "cpufreq governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
} | tee -a "$RES/env.txt"

MPIRUN="mpirun --bind-to core --map-by core"

# Timed reps per N (0 = program default: 100 for N=64, 30 for N=256)
reps_for() {
  case "$1" in
    1024) echo 20 ;;
    4096) echo 5  ;;
    *)    echo 0  ;;
  esac
}

# ---- 1. Exact-arithmetic correctness checks ----
for P in 1 4 8 16; do
  for N in 16 64 256; do
    $MPIRUN -np $P ./mm -n $N -r 3 -e -W 0 -o "$RES/correctness.csv"
  done
done
$MPIRUN -np 4 ./mm -n 8 -r 1 -e -p -W 0 -o "$RES/correctness.csv" > "$RES/example_n8_p4.txt"

# ---- 2. Timing sweep ----
for N in 64 256 1024 4096; do
  for P in 1 4 8 16; do
    $MPIRUN -np $P ./mm -n $N -r "$(reps_for $N)" \
            -o "$RES/results.csv" -w "$RES/raw_times.csv" -d "$RES/ranks.csv"
  done
done

# ---- 3. N = 256 baseline check: with the warm-up these should now agree ----
for i in 1 2 3 4 5; do
  $MPIRUN -np 1 ./mm -n 256 -o "$RES/n256_repeat.csv"
done

echo "done"