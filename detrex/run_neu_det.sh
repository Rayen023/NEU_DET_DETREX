#!/bin/bash


configs="/home/rayen/projects/NET_DET_DETREX/detrex/projects/dino/configs/dino-swin/dino_swin_small_224_4scale_12ep.py"
testDir_root="/home/rayen/projects/NET_DET_DETREX/detrex/test/crazing_10.jpg"

stage=${1:-0} # start from 0 if not specified in command line

#Finetuning step (training detectron2 model with our datasets on top of coco-based weights)
if [ $stage -eq 0 ]; then
	 python tools/train_net.py \
    --config-file $configs \
    --num-gpus 1 \
	--resume
	 

fi 

if [ $stage -eq 1 ]; then

	#Testing step (testing detectron2 model with our datasets on top of coco-based weights)
	python tools/train_net.py \
	--config-file $configs \
	--num-gpus 1 \
	--eval-only MODEL.WEIGHTS="/home/rayen/projects/NET_DET_DETREX/detrex/output/dino_swin_small_224_4scale_12ep/model_final.pth"
	OUTPUT_DIR=./output/output_eval/dino_swin_small_224_4scale_12ep 

fi 

if [ $stage -eq 2 ]; then

	#Testing step (testing detectron2 model with our datasets on top of coco-based weights)
	python demo/demo.py \
	--config-file $configs \
	--input $testDir_root \
	--min_size_test 224 \
	--metadata_dataset my_dataset_test \
	--opts MODEL.WEIGHTS="/home/rayen/projects/NET_DET_DETREX/detrex/output/dino_swin_small_224_4scale_12ep/model_final.pth" 
	OUTPUT_DIR=./output/output_eval/dino_swin_small_224_4scale_12ep

fi 