---
layout: post


---

# 코딩테스트 - 자물쇠와 열쇠

자물쇠와 열쇠 문제 풀이 - 시간 초과 해결 방법

###### 문제 설명

고고학자인 **"튜브"**는 고대 유적지에서 보물과 유적이 가득할 것으로 추정되는 비밀의 문을 발견하였습니다. 그런데 문을 열려고 살펴보니 특이한 형태의 **자물쇠**로 잠겨 있었고 문 앞에는 특이한 형태의 **열쇠**와 함께 자물쇠를 푸는 방법에 대해 다음과 같이 설명해 주는 종이가 발견되었습니다.

잠겨있는 자물쇠는 격자 한 칸의 크기가 **`1 x 1`**인 **`N x N`** 크기의 정사각 격자 형태이고 특이한 모양의 열쇠는 **`M x M`** 크기인 정사각 격자 형태로 되어 있습니다.

자물쇠에는 홈이 파여 있고 열쇠 또한 홈과 돌기 부분이 있습니다. 열쇠는 회전과 이동이 가능하며 열쇠의 돌기 부분을 자물쇠의 홈 부분에 딱 맞게 채우면 자물쇠가 열리게 되는 구조입니다. 자물쇠 영역을 벗어난 부분에 있는 열쇠의 홈과 돌기는 자물쇠를 여는 데 영향을 주지 않지만, 자물쇠 영역 내에서는 열쇠의 돌기 부분과 자물쇠의 홈 부분이 정확히 일치해야 하며 열쇠의 돌기와 자물쇠의 돌기가 만나서는 안됩니다. 또한 자물쇠의 모든 홈을 채워 비어있는 곳이 없어야 자물쇠를 열 수 있습니다.

열쇠를 나타내는 2차원 배열 key와 자물쇠를 나타내는 2차원 배열 lock이 매개변수로 주어질 때, 열쇠로 자물쇠를 열수 있으면 true를, 열 수 없으면 false를 return 하도록 solution 함수를 완성해주세요.

### 제한사항

- key는 M x M(3 ≤ M ≤ 20, M은 자연수)크기 2차원 배열입니다.
- lock은 N x N(3 ≤ N ≤ 20, N은 자연수)크기 2차원 배열입니다.
- M은 항상 N 이하입니다.
- key와 lock의 원소는 0 또는 1로 이루어져 있습니다.
  - 0은 홈 부분, 1은 돌기 부분을 나타냅니다.

------

### 입출력 예

| key                               | lock                              | result |
| --------------------------------- | --------------------------------- | ------ |
| [[0, 0, 0], [1, 0, 0], [0, 1, 1]] | [[1, 1, 1], [1, 1, 0], [1, 0, 1]] | true   |

### 입출력 예에 대한 설명

![자물쇠.jpg](https://grepp-programmers.s3.amazonaws.com/files/production/469703690b/79f2f473-5d13-47b9-96e0-a10e17b7d49a.jpg)

key를 시계 방향으로 90도 회전하고, 오른쪽으로 한 칸, 아래로 한 칸 이동하면 lock의 홈 부분을 정확히 모두 채울 수 있습니다.

------

## 풀이

열심히 구현하여 문제를 풀었는데, 시간초과가 나왔다.

[기존 풀이 코드]

```python
def solution(key, lock):
    
    n = len(lock)
    m = len(key)

    # key 회전
    # 90
    key_90 = [[0 for _ in range(len(key))] for _ in range(len(key))]
    for x in range(len(key)):
        for y in range(len(key)):
            key_90[y][len(key)-(1+x)] = key[x][y]

    # 180
    key_180 = [[0 for _ in range(len(key))] for _ in range(len(key))]
    for x in range(len(key)):
        for y in range(len(key)):
            key_180[y][len(key)-(1+x)] = key_90[x][y]

    # 270
    key_270 = [[0 for _ in range(len(key))] for _ in range(len(key))]
    for x in range(len(key)):
        for y in range(len(key)):
            key_270[y][len(key)-(1+x)] = key_180[x][y]

    key_rot = [key, key_90, key_180, key_270]



    # key 이동 함수 구현
    def move(dx, dy):    
        # dx, dy 만큼 옮겨서 값 넣기
        for x in range(len(key)): # x는 아래쪽부터 이동
            for y in range(len(key)): # y는 오른쪽부터 이동
                key_map[x+dx][y+dy] = key_rot[i][x][y]




    def check(dx,dy):
        # x, y 겹치는지 확인 - lock에 빈데가 있으면 다시 진행
        for x in range(dx, len(key)+dx):
            for y in range(dy, len(key)+dy):
                pos = (x,y)
                for lx in range(m-1, m-1+len(lock)):
                    for ly in range(m-1, m-1+len(lock)):
                        # lpos = (lx, ly)
                        # 겹치는 좌표 찾으면 - 0,1 반대되는지 확인
                        if (x,y) == (lx, ly):
                            lock_map[lx][ly] += key_map[x][y]
                            if lock_map[lx][ly] != 1:
                                return False
                
        # lock이 1로 다채워졌는지 확인
        for lx in range(m-1, m-1+len(lock)):
            for ly in range(m-1, m-1+len(lock)):
                if lock_map[lx][ly] != 1:
                    return False

        return True



    for i in range(len(key_rot)):

        # key x로, y로 range(m+n-1)만큼 움직임
        for dx in range(m+n-1):
            for dy in range(m+n-1):

                # lock 재배치
                lock_map = [[-1 for _ in range(2*m+n-2)] for _ in range(2*m+n-2)]
                for x in range(len(lock)):
                    for y in range(len(lock)):
                        lock_map[x+m-1][y+m-1] = lock[x][y]

                # key map 초기화
                key_map = [[-1 for _ in range(2*m+n-2)] for _ in range(2*m+n-2)]

                # key 이동
                move(dx, dy)
                # check 
                result = check(dx,dy)


                if result:
                    return True

    return False
```



여기에서 결국 답안 확인 결과, check 함수 부분에서 for 구문을 필요없이 많이 썼다는 것을 알게 되었다.

가장 중요한 포인트는, **자물쇠와 열쇠의 맞물리는부분을 일일이 확인할 필요가 없다는 것이다. 단지 열쇠를 옮겨가며 자물쇠 map에 값을 더해서, 매 시도 후 자물쇠의 값이 1로 채워지는지만 확인하면 된다.**



아래에서 check 부분의 pos = (x,y) 부터 return False까지를 모두 지우고 이를 한 줄로 대체할 수 있다.

[기존 코드]

```python
 def check(dx,dy):
        # x, y 겹치는지 확인 - lock에 빈데가 있으면 다시 진행
        for x in range(dx, len(key)+dx):
            for y in range(dy, len(key)+dy):
                
                ### 여기부터 return False까지 삭제 ###
                pos = (x,y)
                for lx in range(m-1, m-1+len(lock)):
                    for ly in range(m-1, m-1+len(lock)):
                        # lpos = (lx, ly)
                        # 겹치는 좌표 찾으면 - 0,1 반대되는지 확인
                        if (x,y) == (lx, ly):
                            lock_map[lx][ly] += key_map[x][y]
                            if lock_map[lx][ly] != 1:
                                return False
                ##################################
                
        # lock이 1로 다채워졌는지 확인
        for lx in range(m-1, m-1+len(lock)):
            for ly in range(m-1, m-1+len(lock)):
                if lock_map[lx][ly] != 1:
                    return False

        return True
```



[변경 코드]

```python
 def check(dx,dy):
        # x, y 겹치는지 확인 - lock에 빈데가 있으면 다시 진행
        for x in range(dx, len(key)+dx):
            for y in range(dy, len(key)+dy):
                lock_map[x][y] += key_map[x][y]
                
        # lock이 1로 다채워졌는지 확인
        for lx in range(m-1, m-1+len(lock)):
            for ly in range(m-1, m-1+len(lock)):
                if lock_map[lx][ly] != 1:
                    return False

        return True
```

이렇게 하여 시간 초과 없이 모든 케이스를 통과하였다. 



[최종 풀이 코드]

```python
def solution(key, lock):
    
    n = len(lock)
    m = len(key)

    # key 회전
    # 90
    key_90 = [[0 for _ in range(len(key))] for _ in range(len(key))]
    for x in range(len(key)):
        for y in range(len(key)):
            key_90[y][len(key)-(1+x)] = key[x][y]

    # 180
    key_180 = [[0 for _ in range(len(key))] for _ in range(len(key))]
    for x in range(len(key)):
        for y in range(len(key)):
            key_180[y][len(key)-(1+x)] = key_90[x][y]

    # 270
    key_270 = [[0 for _ in range(len(key))] for _ in range(len(key))]
    for x in range(len(key)):
        for y in range(len(key)):
            key_270[y][len(key)-(1+x)] = key_180[x][y]

    key_rot = [key, key_90, key_180, key_270]



    # key 이동 함수 구현
    def move(dx, dy):    
        # dx, dy 만큼 옮겨서 값 넣기
        for x in range(len(key)): # x는 아래쪽부터 이동
            for y in range(len(key)): # y는 오른쪽부터 이동
                key_map[x+dx][y+dy] = key_rot[i][x][y]




    def check(dx,dy):
        # x, y 겹치는지 확인 - lock에 빈데가 있으면 다시 진행
        for x in range(dx, len(key)+dx):
            for y in range(dy, len(key)+dy):
                lock_map[x][y] += key_map[x][y]
        # lock이 1로 다채워졌는지 확인
        for lx in range(m-1, m-1+len(lock)):
            for ly in range(m-1, m-1+len(lock)):
                if lock_map[lx][ly] != 1:
                    return False

        return True



    for i in range(len(key_rot)):

        # key x로, y로 range(m+n-1)만큼 움직임
        for dx in range(m+n-1):
            for dy in range(m+n-1):

                # lock 재배치
                lock_map = [[-1 for _ in range(2*m+n-2)] for _ in range(2*m+n-2)]
                for x in range(len(lock)):
                    for y in range(len(lock)):
                        lock_map[x+m-1][y+m-1] = lock[x][y]

                # key map 초기화
                key_map = [[-1 for _ in range(2*m+n-2)] for _ in range(2*m+n-2)]

                # key 이동
                move(dx, dy)
                # check 
                result = check(dx,dy)


                if result:
                    return True

    return False
```

