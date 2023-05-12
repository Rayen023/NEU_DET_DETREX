#!/bin/bash
#SBATCH --account=def-selouani
#SBATCH --job-name=yolo2json
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.log
#SBATCH --time=1-00:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem=20G


#module load gcc opencv cuda cudnn python/3.10 cmake
#source /home/zhor/projects/def-selouani/zhor/pyEnv3-10/bin/activate

root_dataset="/home/rayen/projects/NET_DET_DETREX/Yolo-to-COCO-format-converter/data"
stage=0

if [ $stage -le 0 ]; then
	for name in train val; do
		python main.py --path /home/rayen/projects/NET_DET_DETREX/Yolo-to-COCO-format-converter/data/${name} --output ${name}.json
	done
fi
