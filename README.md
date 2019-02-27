# Spatiotemporal_Heterogeneous_Two-stream_Network
We follow the guidance provided by TSN to prepare the data. Please refer to the TSN repository for guidance. Here we only provide the additional details of our Spatiotemporal_Heterogeneous_Two-stream_Network.
 <br>
## Preparation<br>
1. Install opencv2.4.13. <br>
2. Install caffe：We have attached a modified version of [caffe](https://github.com/yjxiong/caffe) <br>
   Please compile our modified version of caffe-action/ with cmake and openmpi according to [TSN](https://github.com/yjxiong/temporal-segment-networks). <br>
3. Download UCF101 and/or HMDB51 datasets. <br>
4. Follow the guidance of [dense_flow](https://github.com/yjxiong/dense_flow) to extract optical flow. <br>
5. To build the file lists by running :python build_file_list.py ucf101 PATH(the path of RGB and optical flow image) --shuffle <br>
## Running the code <br>
1. Generate the pre-training model for  temporal network：python our_pretrain.py or python tsn_pretrain.py. <br>
2. Start training: Once all necessities ready, we can start training. For example, if we want to train on UCF101 with its weights initialized by ImageNet pretraining,we can run: <br>
    bash heterogeneous_train.sh <br>
3. Start testing: To test on the UCF101 and HMDB51 dataset, we can use the scripts test.py and fuse_scores.py.For example: <br>
    python test.py ucf101 1 rgb PATH spatiotemporal_heterogeneous/spatial_ResNet_101_deploy.prototxt \   <br>
    models/ucf101_res101_rgb_split1_iter_36000.caffemodel --num_worker 4 --save_scores NPZ_NAME <br>
where PATH is the path of RGB and optical flow image and NPZ_NAME is the filename of the scores. <br>

## Reference <br>
Wang L, Xiong Y, Wang Z, et al. Temporal segment networks: Towards good practices for deep action recognition[C]//European conference on computer vision. Springer, Cham, 2016: 20-36. <br>
[https://github.com/yjxiong/temporal-segment-networks](https://github.com/yjxiong/temporal-segment-networks)
