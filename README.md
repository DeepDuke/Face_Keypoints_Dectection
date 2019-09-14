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

<center>

![](\\Figures\\0.jpg)![](\\Figures\\1.jpg)![](\\Figures\\2.jpg)

![](\\Figures\\14.jpg)![](\\Figures\\15.jpg)![](\\Figures\\16.jpg)

![](\\Figures\\50.jpg)![](\\Figures\\56.jpg)![](\\Figures\\76.jpg)

</center>

##### On original size image

![](\\Figures\\origin182.jpg)

![](\\Figures\\origin114.jpg)

![](\\Figures\\origin19.jpg)






