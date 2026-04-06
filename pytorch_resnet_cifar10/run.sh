#!/bin/bash

for model in resnet20 resnet32 resnet44 resnet56 
do
    echo "python -u trainer.py --arch=$model --save-dir=save_$model"
    python -u trainer.py --arch=$model --save-dir=save_$model 2>&1 | tee -a log_$model
done