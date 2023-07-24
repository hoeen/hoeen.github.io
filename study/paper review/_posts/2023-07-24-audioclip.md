---
tags:       [NLP]
---

# AudioCLIP: Extending CLIP to Image, Text and Audio

AudioCLIP 논문 리뷰

[link](https://arxiv.org/pdf/2106.13043.pdf)

## Abstract
CLIP의 확장 버전으로써 텍스트, 이미지 뿐 아니라 오디오를 다룰 수 있는 AudioCLIP을 제안하는데, 오디오 모델인 ESResNeXt를 CLIP 프레임워크에 통합시켰다. 데이터셋은 AudioSet을 이용하였다. 이를 통해 bimodal, unimodal 분류 및 querying을 가능케 하면서 CLIP의 일반성을 살려 zero-shot 추론이 가능하게끔 하였다. 이를 통해 AudioCLIP은 ESC라고도 불리는 Environmental Sound Classification에서 SOTA를 달성하였고, ESC-task에서 새로운 zero-shot을 이용한 baseline을 정확도 68%로 설정하였다.  

## Introduction
오디오 분야에서 그동안의 연구는 오디오 그 자체의 모달리티 (정보 전달)에만 집중했는데, 최근에는 멀티모달 형태 (여러 정보 전달)로 오디오 관련 ai task가 이루어지고 있다. 이는 주로 text 혹은 visual 정보를 함께 전달해주는 방식으로 이루어진다. 


하지만 오디오에서 2개 이상의 모달리티를 결합하는 것은 많이 이루어지지 않았다.  