# Tiny-Face-Recognition

Extract an complete process from insightface_tf to do face recognition and verification using pure TensorFlow.

### Dataset

Training and Testing Dataset Download Website: [Baidu](https://pan.baidu.com/s/1S6LJZGdqcZRle1vlcMzHOQ)

Contains:

* train.rec/train.idx   : Main training data
* \*.bin  : varification data

### Examples

Make TFRecords File:

```
$ python3 mx2tfrecords.py --bin_path '/data/ChuyuanXiong/up/face/faces_emore/train.rec' --idx_path '/data/ChuyuanXiong/up/face/faces_emore/train.idx' --tfrecords_file_path '/data/ChuyuanXiong/up/face/tfrecords'
```


Train:

```
$ python3 train.py --tfrecords '/data/ChuyuanXiong/up/face/tfrecords/tran.tfrecords' --batch_size 64 --num_classes 85742 --lr [0.001, 0.0005, 0.0003, 0.0001] --ckpt_save_dir '/data/ChuyuanXiong/backup/face_real403_ckpt' --epoch 10000
```

Test:

```
$ python3 eval_veri.py --datasets '/data/ChuyuanXiong/up/face/faces_emore/cfp_fp.bin' --dataset_name 'cfp_fp' --num_classes 85742 --ckpt_restore_dir '/data/ChuyuanXiong/backup/face_ckpt/Face_vox_iter_78900.ckpt'
```


# Results

Datasets|backbone| loss|steps|batch_size|acc
-------|--------|-----|---|-----------|----|
lfw    | resnet50 | ArcFace | 78900 | 64 | 0.9741
cfp_ff | resnet50 | ArcFace | 78900 | 64 | 0.9400
cfp_fp | resnet50 | ArcFace | 78900 | 64 | 0.7713
agedb_30| resnet50 | ArcFace | 78900|64 | 0.6852s

Limited by the training time, so I just release the half-epoch training results temporarily. The model will be optimized later.






# References

1. [InsightFace_TF](https://github.com/auroua/InsightFace_TF)
2. [InsightFace_MxNet](https://github.com/deepinsight/insightface)

```
@inproceedings{deng2018arcface,
title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
booktitle={CVPR},
year={2019}
}
```

