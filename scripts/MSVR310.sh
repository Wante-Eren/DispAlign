#!/bin/bash
conda activate IDEA
cd /root/cloud-ssd/IDEA-main
python train.py --config_file /root/cloud-ssd/IDEA-main/configs/MSVR310/IDEA.yml
