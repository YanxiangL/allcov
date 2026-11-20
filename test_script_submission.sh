#!/bin/bash
#SBATCH --account=desi    
#SBATCH --constraint=cpu                
#SBATCH --qos=shared                  
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=40 
#SBATCH --mem=20GB
#SBATCH --job-name=pv_cov_g   
#SBATCH --output=pv_cov_g_%j.out

module load conda
conda activate /global/common/software/m1727/lgong/pv_cov
source /global/u1/l/lgong/allcov/.venv/bin/activate

echo "Job started on $(date)"
python -u test_script_pv.py
echo "Job completed on $(date)"