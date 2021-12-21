---
layout: post
---

# xgboost Apple M1 Mac 에 설치할 때 유의사항

xgboost 는 현재 시점(2021.12) 에 scipy 와의 호환성 문제 때문에 pip install 로만 설치할 수 없다.  

conda 를 사용하는 경우 conda 를 이용하여 lightgbm을 설치한다.

> conda install lightgbm

그리고 xgboost를 pip로 해당 환경에서 설치해 주면 호환성 문제를 해결하고 성공적인 설치가 이루어진다.

> pip install xgboost