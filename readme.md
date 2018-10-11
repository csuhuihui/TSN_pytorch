1. 环境配置参考https://github.com/milkcat0904/temporal-segment-network-pytorch
2. 替换.py文件
3. train: python3.5 main.py ucf101 Flow ../../action_dataset/action5.0/train_opencv_flow/labels_txt ../../action_dataset/action5.0/test_opencv_flow/labels_txt --arch BNInception --num_segments 3 --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 -b 128 -j 8 --dropout 0.8 --snapshot_pref action5.0__opencv__flow_model_best.pth.tar --gpus 0 1
4. test： python3.5 test_models.py ucf101 RGB ../../action_dataset/action5.0/test_opencv_flow/labels_txt action5.0__opencv_rgb_model_best.pth.tar --arch BNInception --save_scores action5.0_test_rgb.npz --gpu 1 -j 1
