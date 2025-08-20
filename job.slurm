#!/bin/bash -l
#SBATCH -D ./

#SBATCH -o out/2_%j.out
#SBATCH -e out/2_%j.out

#SBATCH -J template_reconstruction

#SBATCH --ntasks=1
#SBATCH --constraint="gpu"

# #SBATCH --gres=gpu:a100:4
# #SBATCH --cpus-per-task=72
# #SBATCH --mem=500000

#SBATCH --mail-type=none
#SBATCH --mail-user=jthomas@mpe.mpg.de
#SBATCH --time=24:00:00


###### Variables ######
CONTAINER="nv-pytorch.sif"

###### PyTorch distributed settings ######
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export NUM_PROCESSES=$(( $SLURM_NNODES * 4 ))

###### Environment ######
module purge
module load apptainer/1.4.1
module list

###### Logs ######
echo -e "Nodes: ${SLURM_JOB_NUM_NODES} \t NTASK: ${SLURM_NTASKS}"
echo "${SLURM_NODELIST}"
echo
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE
echo "NUM_PROCESSES="$NUM_PROCESSES
apptainer exec ${CONTAINER} \
  python3 -c "import torch; \
    print(torch.__config__.show()); \ "

###### Run the program:
srun apptainer exec --nv -B .:/workspace/ ${CONTAINER}\
  bash -c """
    export LOCAL_RANK=\${SLURM_LOCALID}
    export RANK=\${SLURM_PROCID}
    echo \$SLURM_PROCID : \$(hostname) - \${SLURM_LOCALID} CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES}

    python3 ./trecon.py
  """
