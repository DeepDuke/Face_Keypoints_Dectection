### Face Keypoints Detection

|      |Train-loss|Validate-loss|Test-loss|Optimizer| Loss-function| Epochs|Batch-size|Learning-rate|
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| model1 | $3.23$ | $3.11$ | $3.14$ | Adam | L1loss | $100$ | $128$ | $0.001$ |
| model2 | $22$ | $21$ | $20.01$ | Adam | MSEloss | $100$ | $128$| $0.001$ |
| model3 |      |      |      |      |      |      |      ||

#### Description
I trained three models: 

Model1 and model2 both used MyNet with different loss functions. Model3 used a more complicated model. All of these models achieved almost same performance.

#### Result
Green pointes represent for true landmarks, while blue ones stand for predictions

##### On $112*112$ image

<center>

![](F:\Pycharm\Face_Keypoints_Dectection\Result\0.jpg)![](F:\Pycharm\Face_Keypoints_Dectection\Result\1.jpg)![](F:\Pycharm\Face_Keypoints_Dectection\Result\2.jpg)

![](F:\Pycharm\Face_Keypoints_Dectection\Result\49.jpg)![](F:\Pycharm\Face_Keypoints_Dectection\Result\55.jpg)![](F:\Pycharm\Face_Keypoints_Dectection\Result\56.jpg)

![](F:\Pycharm\Face_Keypoints_Dectection\Result\114.jpg)![](F:\Pycharm\Face_Keypoints_Dectection\Result\115.jpg)![](F:\Pycharm\Face_Keypoints_Dectection\Result\122.jpg)

</center>

##### On original size image

![](F:\Pycharm\Face_Keypoints_Dectection\Result\origin182.jpg)

![](F:\Pycharm\Face_Keypoints_Dectection\Result\origin114.jpg)

![](F:\Pycharm\Face_Keypoints_Dectection\Result\origin19.jpg)






