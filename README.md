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

![](https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/0.jpg)![](https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/1.jpg)![](https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/2.jpg)

![](https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/14.jpg)![](https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/15.jpg)![](https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/16.jpg)

![]
(https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/50.jpg)![](https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/56.jpg)![](https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/76.jpg)


#####  On original size image

![](https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/origin182.jpg)

![](https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/origin114.jpg)

![](https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/origin19.jpg)






