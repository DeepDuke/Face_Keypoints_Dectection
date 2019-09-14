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
Green points represent for true landmarks, while blue ones stand for predictions

##### Show results On 112*112 image



![](https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/0.jpg)![](https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/1.jpg)![](https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/2.jpg)

![](https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/14.jpg)![](https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/15.jpg)![](https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/16.jpg)

![](https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/50.jpg)![](https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/56.jpg)![](https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/76.jpg)


#####  Show results On original size image



![origin image 1](https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/origin182.jpg)

![origin image 2](https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/origin114.jpg)

![origin image 3](https://github.com/DeepDuke/Face_Keypoints_Dectection/raw/master/Figures/origin19.jpg)






