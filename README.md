# Credit_Card_OCR
Apply OCR to recognize the sixteen digits on the credit card.

# Usage
## dependencies
* OpenCV = 3.2.0
```bash
g++ -o main main.cpp `pkg-config --cflags --libs opencv`
./main "sample.jpg"
```
The output should be
```bash
>> Account Number: 4000-1234-5678-9123
```
# Details
Building credit card OCR can be accomplished in the following steps:
* Detect and use the edges in the image to find the contour representing the card. Then apply a perspective transform to obtain the top-down view of the card.

<img src=https://github.com/zhangchicheng/Credit_Card_OCR/blob/master/sample.jpg height="250"> <img src=https://github.com/zhangchicheng/Credit_Card_OCR/blob/master/images/warpedCard.jpg height="250"> 

* Localize the four groupings of four digits on a credit card by applying a series of operations.
<img src=https://github.com/zhangchicheng/Credit_Card_OCR/blob/master/images/morphology.jpg width="400"> <img src=https://github.com/zhangchicheng/Credit_Card_OCR/blob/master/images/blocks.jpg width="400">

* Extract each of these four groupings followed by segmenting each of the sixteen numbers individually. Recognize each of the sixteen credit card digits by using template matching.

<img src=https://github.com/zhangchicheng/Credit_Card_OCR/blob/master/images/bounding.jpg width="400"> <img src=https://github.com/zhangchicheng/Credit_Card_OCR/blob/master/images/digitContours.jpg> <img src=https://github.com/zhangchicheng/Credit_Card_OCR/blob/master/images/digitBlock.jpg>
