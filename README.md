# MADNet
## "Detection of potential anxiety in social media based on multimodal fusion with deep learning methods" has received by BIBM-2023 as regular paper.
This is the Pytorch implementation of MADNet mentioned by paper.

## MADNet Architecture
Three modalities input : text, image, behavior
![MADNet Architecture](MADNet.png)

## MAI Fusion Method
Fusing textual features and non-textual features
![MAI Fusion Method](MAI.png)

## For Visualization
We use Eigen-grad-cam method to plot the attention of model on image.
Redder the area, more attention the model pay.
![Visualization](CAM.png)

## Result
![Compare experiment](Exp1.png)
![Ablation experiment](Exp2.png)





