---
layout: post
---

# Upstage talk - NFL 1st and Future Impact Detection

> NFL 헬멧 충돌 감지 문제에 대한 Upstage talk 솔루션 정리



헬멧 디텍션, 그리고 해당 헬멧에 선수 라벨을 assign하는 대회



1. Object Detection

   동영상 데이터 분석 - 이미지 주변의 이미지들을 temporal features 로 활용하기로 함. 

   Video Object Detection에는 여러 모델이 있음. 여러가지를 써보았지만 여기서는 실패함.

   선수들이 뭉쳐있는 경우 패드와 헬멧 오인

   흐린 경우, False Negative 가 증가해 Recall 이 낮아진다.

   Yolov5 의 Mosaic이 가장 효과적이었음.

   Yolov5 로 수행. 

   1등 솔루션 : pretraining + finetuning

   

   

2. number?

   위치와 각 선수 헬멧 번호를 mapping 해주어야 함.

   영상과 csv 의 frame 이 다르기 때문에 동기화 시켜주어야 하는데, Ball snap 이 발생하는 시점 정보가 있기 때문에 이를 활용하여 동기화 한다

   

   

   

   

   

3. 솔루션 

   SuperGlue variant

   데이터 증강에 가장 효과적이었던 것은 Remove 이다. 

   

- 전체 그림

![img](/study/images/helmet1.png)![img](/study/images/helmet2.png)

