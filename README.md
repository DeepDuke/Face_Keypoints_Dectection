### Face Keypoints Detection

|      |Train-loss|Validate-loss|Test-loss|Optimizer| Loss-function| Epochs|Batch-size|Learning-rate|
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| model1 | 3.23 | 3.11 | 3.14 | Adam | L1loss | 100| 128 | 0.001 |
| model2 | 22 | 21 | 20.01 | Adam | MSEloss | 100 | 128| 0.001 |
| model3 |      |      |      |      |      |      |      ||

#### Description
I trained three models: 

Model1 and model2 both used MyNet with different loss functions. Model3 used a more complicated model. All of these models achieved almost same performance.

#### Result
Green pointes represent for true landmarks, while blue ones stand for predictions

##### On 112*112 image

<p float="left">
  <img src="http://wx2.sinaimg.cn/large/ysply1g6z78k2ntsj3034034wed.jpg" width="100" />
  <img src="http://wx4.sinaimg.cn/largeU5ysply1g6z78nnd9wj3034034jra.jpg" width="100" /> 
  <img src="http://wx2.sinaimg.cn/largeU5ysply1g6z78k2ntsj3034034wed.jpg" width="100" />
</p>

##### On original size image

![](F:\Pycharm\Face_Keypoints_Dectection\Result\origin182.jpg)

![](F:\Pycharm\Face_Keypoints_Dectection\Result\origin114.jpg)

![](F:\Pycharm\Face_Keypoints_Dectection\Result\origin19.jpg)






