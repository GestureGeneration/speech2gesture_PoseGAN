# speech2gesture_PoseGAN

- A model which predict gestures from speech
  - This repository is based on [text2gesture][1]
  - [original paper][2]


# Procedure
## 1. Download raw data

[See "Download raw data" in "Speech_driven_gesture_generation_with_autoencoder" repository](https://github.com/GestureGeneration/Speech_driven_gesture_generation_with_autoencoder#1-download-raw-data)

## 2. Split dataset

[See "Split dataset" in "Speech_driven_gesture_generation_with_autoencoder"](https://github.com/GestureGeneration/Speech_driven_gesture_generation_with_autoencoder#2-split-dataset)

## 3. Convert the dataset into vectors

```
python create_vector.py DATA_DIR
```

- Dataset is created by separating 64 frames each (both speech and motion)
- Shape
	- Speech: (block of frames, 26, 64)
	- Motion: (block of frames, 192, 64)
- The mean and standard deviation parameters obtained when standardizing the training data are located in `. /norm/`.

## 4. train

```
python train.py [--batch_size] [--epochs] [--lr] [--weight_decay] [--embedding_dimension]
                [--outdir_path] [--device] [--gpu_num] [--speech_path] [--pose_path] [--generator]
                [--gan] [--discriminator] [--lambda_d]
```

- See ["Usage" in "text2gesture"](https://github.com/GestureGeneration/text2gesture#usage) for details.

## 5. predict

```
python predict.py [--modelpath] [--inputpath] [--outpath]
```

- The argument of `--modelpath` is set to specifies the folder where the generator model is located
	- model is output by `train.py` and located in `./out/datetime/generator_datetime_weights.pth`

## 6. reshape
```
python reshape-predict.py [--denorm] [--denormpath] [--datatype] [--npypath] [--outpath]
```

- If you want to undo the normalized data, set the argument of `--denorm` to 1. In this case, `--denormpath` and `--datatype` should be set. (`--datatype` defaults to train.)
    - `--denormpath` and `--datatype` are arguments to specify the directory where mean and standard deviation parameters obtained when standardizing the training data are located (Same as `/norm/` output path in chapter 3.)
- `--npypath` is set to the folder where the test data is located

[1]:https://github.com/GestureGeneration/text2gesture
[2]:https://people.eecs.berkeley.edu/~shiry/projects/speech2gesture/index.html
[3]:https://github.com/GestureGeneration/Speech_driven_gesture_generation_with_autoencoder