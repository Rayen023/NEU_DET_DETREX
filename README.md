# NET_DET_DETREX
- detrex is an open-source toolbox that provides state-of-the-art Transformer-based detection algorithms. It is built on top of Detectron2.


=============================================================

### Setup Virtual environment

=============================================================

1. Installing miniconda3 : 
    1. Download the Miniconda installer to your Home directory.
`wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh`
    2. Install Miniconda quietly, accepting defaults, to your Home directory.
`bash ~/miniconda.sh -b -p`
    3. Remove the Miniconda installer from your Home directory.
`rm ~/miniconda.sh`
----------------------------------------------

2. Activating environment :
    1. activate conda for this session
`source $HOME/miniconda3/bin/activate`

    2. Creating environment : #Skip if env already created
`conda create -n detrex python=3.7 -y`

    3. Activating the environment : 
`conda activate detrex`

----------------------------------------------

3. Usefull Commands :
- for deactivating :
`conda deactivate`

- to list created environments :
`conda env list`

=============================================================

### Setting Up detrex : 

=============================================================

1. Installing detrex
```
git clone https://github.com/IDEA-Research/detrex.git
cd detrex
git submodule init
git submodule update

```

2. Installing detectron2
```
python -m pip install -e detectron2
pip install -e .
```

=============================================================

### COnverting from YOLO format to COCO format : 

=============================================================

1. Using the Github REPO :Yolo-to-COCO-format-converter to convert dataset to COCO format:
Detailed Steps here : [link](https://github.com/Taeyoung96/Yolo-to-COCO-format-converter/blob/master/README.md)

2. Move COCO format Dataset under detrex/datasets
Directory Structure :

<pre>
detrex/
    datasets/
        surface/
            annotations/
                train.json
                val.json
            train/
                Photo_train_00001.jpg
                Photo_train_00002.jpg
            val/
                Photo_val_00001.jpg
                Photo_val_00002.jpg
</pre>

3. Register Dataset to be used with detectron2
    1. `cd detrex/configs/common/data/`
    2. Edit custom.py file by adding corresponding lines : 
    ```
    register_coco_instances("my_dataset_train", {}, "/home/rayen/projects/NET_DET_DETREX/detrex/datasets/surface/annotations/train.json","/home/rayen/projects/NET_DET_DETREX/detrex/datasets/surface/train/" )
    register_coco_instances("my_dataset_test", {},"/home/rayen/projects/NET_DET_DETREX/detrex/datasets/surface/annotations/val.json" ,"/home/rayen/projects/NET_DET_DETREX/detrex/datasets/surface/val/" )
    # Should also modify theese params :
    total_batch_size=1,
    num_workers=16,

    ```


=============================================================

### Configuration : 

=============================================================


1. Pick a model and its config file from projects, for example, dino_swin_tiny_224_4scale_12ep.py :

    1. Download the pretrained weights from [Model Zoo](https://detrex.readthedocs.io/en/latest/tutorials/Model_Zoo.html) or the [projects page](https://github.com/IDEA-Research/detrex/tree/main/projects/dino#pretrained-models)

    2. `cd detrex/projects/dino/configs/dino-swin/`


    3. Pick model config and edit as follow : 
        ```
        # 
        dataloader = get_config("common/data/custom.py").dataloader

        # modify training config
        train.init_checkpoint = "/home/rayen/projects/NET_DET_DETREX/detrex/swin_tiny_patch4_window7_224_22k.pth"
        train.output_dir = "./output/dino_swin_tiny_224_4scale_12ep_22kto1k_finetune"

        # max training iterations
        train.max_iter = 90000
        train.eval_period = 5000
        train.log_period = 20
        train.checkpointer.period = 5000

        # modify dataloader config
        dataloader.train.num_workers = 16

        # please notice that this is total batch size.
        # surpose you're using 4 gpus for training and the batch size for
        # each gpu is 16/4 = 4
        dataloader.train.total_batch_size = 1


        ```

=============================================================

### Using detrex : 

=============================================================

1. Edit ./run_neu_det.sh : 
```
#!/bin/bash

# Config file modified

configs="/home/rayen/projects/NET_DET_DETREX/detrex/projects/dino/configs/dino-swin/dino_swin_tiny_224_4scale_12ep.py"
testDir_root=[test/img.jpg,test/img2.jpg,test/img3.jpg,test/img.png]

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
	--eval-only MODEL.WEIGHTS=output/dino_r50_4scale_12ep/model_final.pth\
	OUTPUT_DIR=./output/output1/dino_r50_4scale_12ep 

fi 

if [ $stage -eq 2 ]; then

	#Testing step (testing detectron2 model with our datasets on top of coco-based weights)
	python demo/demo.py \
	--config-file $configs \
	--input $testDir_root \
	--opts MODEL.WEIGHTS=output/dino_r50_4scale_12ep/model_0035499.pth\
	OUTPUT_DIR=./output/output_test/dino_r50_4scale_12ep 

fi 
```

2. Add executable permissions to Shell script :
`sudo chmod +x ./run_neu_det.sh` 

3. Training & Evaluation in Command Line :

For training  :
    `run ./run_neu_det.sh 0`

For Evaluation :
    `run ./run_neu_det.sh 1`

For Inference :
    `run ./run_neu_det.sh 2`





