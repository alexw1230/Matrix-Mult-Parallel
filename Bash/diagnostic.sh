#!/bin/bash
#SBATCH --job-name=mm_diag
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH -p short-96core
#SBATCH --output=mm_diag_%j.out

module purge
module load gcc/13.2.0
module load openmpi/gcc13.2/4.1.6

set -euo pipefail
export OMPI_MCA_coll_hcoll_enable=0

cd "$SLURM_SUBMIT_DIR"
make -f Make/Makefile clean && make -f Make/Makefile

RES=results/diag_${SLURM_JOB_ID}
mkdir -p "$RES"
echo "host: $(hostname)" | tee "$RES/env.txt"
lscpu                    > "$RES/lscpu.txt"
# One line per CPU with its core, socket, NUMA node and cache IDs.
# The last number in the CACHE column is the L3 (chiplet) ID.
lscpu -e=CPU,CORE,SOCKET,NODE,CACHE > "$RES/cpu_map.txt"

# ---- 1. Per-rank timings with the same placement as the main sweep ----
MPIRUN="mpirun --bind-to core --map-by core --report-bindings"
for N in 1024 4096; do
  if [ "$N" -eq 1024 ]; then R=20; else R=5; fi
  for P in 8 16; do
    $MPIRUN -np $P ./mm -n $N -r $R \
            -o "$RES/summary_core.csv" -d "$RES/ranks_core.csv" \
            2> "$RES/bindings_core_N${N}_P${P}.txt"
  done
done

# ---- 2. N = 1024, P = 16 under two other placements ----
# l3cache: round-robin ranks across L3 chiplets (fewer ranks share each L3)
# socket : alternate ranks between the two sockets
for MAP in l3cache socket; do
  mpirun --bind-to core --map-by $MAP --report-bindings -np 16 ./mm -n 1024 -r 20 \
         -o "$RES/summary_${MAP}.csv" -d "$RES/ranks_${MAP}.csv" \
         2> "$RES/bindings_${MAP}.txt"
done

echo "done"