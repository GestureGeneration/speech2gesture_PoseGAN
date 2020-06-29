# speech2gesture_PoseGAN

- 音声からジェスチャを推測するモデル
  - [text2gesture][1]を参考に作成
  - [元論文][2]

[English translation](/README.md)

# Procedure
## 1. Download raw data

[Speech_driven_gesture_generation_with_autoencoderのDownload raw dataを参照](https://github.com/GestureGeneration/Speech_driven_gesture_generation_with_autoencoder#1-download-raw-data)

## 2. Split dataset

[Speech_driven_gesture_generation_with_autoencoderのSplit datasetを参照](https://github.com/GestureGeneration/Speech_driven_gesture_generation_with_autoencoder#2-split-dataset)

## 3. Convert the dataset into vectors

```
python create_vector.py DATA_DIR
```

- センテンスごとに64frame区切るデータセットを作成
- shape
	- speech: (block of frames, 26, 64)
	- motion: (block of frames, 192, 64)
- 標準化時の平均値と標準偏差のパラメータは`./norm/`に保存される

## 4. train

```
python train.py [--batch_size] [--epochs] [--lr] [--weight_decay] [--embedding_dimension]
                [--outdir_path] [--device] [--gpu_num] [--speech_path] [--pose_path] [--generator]
                [--gan] [--discriminator] [--lambda_d]
```

- 詳細は[text2gestureのUsage](https://github.com/GestureGeneration/text2gesture#usage)を参照

## 5. predict

```
python predict.py [--modelpath] [--inputpath] [--outpath]
```

- --modelpathにはgeneratorモデルがあるフォルダを指定
	- train.pyで出力された`./out/日付/generator_日付_weights.pth`を指定

## 6. reshape
```
python reshape-predict.py [--denorm] [--denormpath] [--datatype] [--npypath] [--outpath]
```

- 正規化されたデータを元に戻す場合、--denormを1。その場合、--denormpathと--datatypeを指定する。(--datatypeのデフォルトはtrain)
    - --denormpathと--datatypeは3章の`./norm/`内の平気位置と標準偏差のパラメータを指定
- --npypathにはテストデータがあるフォルダを指定。

[1]:https://github.com/GestureGeneration/text2gesture
[2]:https://people.eecs.berkeley.edu/~shiry/projects/speech2gesture/index.html
[3]:https://github.com/GestureGeneration/Speech_driven_gesture_generation_with_autoencoder