for model in resnet18 resnet34
do
    echo "python -u vision.py --arch=$model --save-dir=save_$model"
    python -u vision.py --arch=$model --save-dir=save_$model 2>&1 | tee -a log_$model
done