# LogiPose
LogiPose是一款在疫情遠距教學下誕生的產物，許多肢體動作教學的課程如舞蹈、瑜珈、健身等，需要從旁糾正動作，如果學生數量一多，老師無法從小小的鏡頭中觀察每一位學生的情況與姿勢正確度，也無法邊做動作邊糾正學生，因此LogiPose將視訊教學結合OpenPose，讓AI自動判斷學生動作與老師動作的正確度，並將錯誤較高的學生鏡頭放大，方便老師糾正，並且在CPU環境下達到Real-Time等級的即時回應。

![](https://github.com/Coolshanlan/LogiPose/blob/master/image/demo_gif.gif?raw=true)

## Lightweight OpenPose Technolegy
Fork from [this repo](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch)

## Requirement
```
torch>=0.4.1
torchvision>=0.2.1
pycocotools==2.0.0
opencv-python>=3.4.0.14
numpy>=1.14.0
```
