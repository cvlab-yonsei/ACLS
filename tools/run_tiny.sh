GPUS=0

CUDA_VISIBLE_DEVICES=${GPUS} python tools/train.py log_period=100 data=tiny_imagenet model=resnet50_tiny model.num_classes=200 loss=acls optim=sgd scheduler=multi_step train.max_epoch=100 