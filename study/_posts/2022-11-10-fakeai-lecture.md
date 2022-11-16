# 가짜연구소 - Kaggle 대회 특강
## Tensorflow - Help Protect the Great Barrier Reef
## 대회 접근 방법
1. Train data 조사
2. Test data 조사
3. Metric 조사
4. Validation 전략
5. 적합한 Deep Learning 네트워크 선택
6. Ensemble

이 방식을 주로 이용하고, 캐글에 익숙해지면 일주일 정도에 1~5번까지 전체 조사는 어느정도 끝난다고 함. 앙상블 6번은 시간 많이 소요.

### Train/Test 데이터

### Metric
- F2 score


### Validation Strategy

### Ensemble Strategy
- Yolov5m, yolov5s 로 코드 컴페티션 9시간을 모두 활용함.
앙상블 방법 : Weighted Boxes Fusion


## Mayo Clinic Strip AI
- CE 인지 LAA인지 이미지 분류 대회
### Train data
- 데이터 사이즈가 매우 큼 (40000x40000 해상도)

### Metric
- logloss
    - 딥러닝 모델이 예측한 값이 0에 가까운데 그게 오답이라면 loss는 치솟는다 -> 그래서 0에 가까운 값을 최대한 예측하지 않도록 하는게 핵심!
    - 그래프로 시각화하니 한눈에 볼 수 있었다.

### DL Network 탐색
- EfficientNet 썼으나 
- 3D CNN 을 활용하여 2 classification - 매우 CV 성능이 좋았다고 함.

### Ensemble Strategy
- 전처리 작업이 많아 모델 하나를 제출하는 데만 6시간이 소요됨. 


## 은메달 수준으로 올라간 방법?
- baseline 코드를 내가 손수 짜봄
