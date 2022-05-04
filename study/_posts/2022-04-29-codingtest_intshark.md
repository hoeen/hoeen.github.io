---
layout: post


---

# 코딩테스트 - 청소년 상어

청소년 상어 문제 풀이 - 리스트 차원에 따른 깊은복사, 얕은복사의 차이 

## 문제 설명

[링크](https://www.acmicpc.net/problem/19236)



## 문제 풀이

얕은 복사와 깊은 복사에 대해 확실히 알고 있지 못했는데, 이번 문제를 풀면서 이를 확실히 알게 되었다.

list comprehension이 보통 copy.deepcopy() 방식보다 빠르지만, 전자를 이용했을 시 3차원 이상의 다차원 배열을 복사할 때는 명시한 list 보다 더 내부는 얕은 복사가 이루어진다.

```python
# slicing 이용
list_a = [[[i for i in range(1000)]] for _ in range(1000)]
list_b = [item[:] for item in list_a]
id(list_a[0][0][0]) == id(list_b[0][0][0])
>>> True
```

어찌보면 당연한 것일 수도 있지만, 그냥 list comprehension식 복사방법으로 [i[:] for i in item] 이면 깊은복사가 완전히 이루어질 것이라 단정지은 잘못이 컸다.

어쨌든 이번 문제로 하루종일 디버깅하면서 이 사실을 알아내어 다행이다.



[문제 풀이 코드]

```python
import copy
fish = [[] for _ in range(4)]


for j in range(4):
    board = list(map(int, input().split()))
    for i in range(4):
        fish[j].append([board[i*2], board[i*2+1]-1]) # 방향은 0~7


for j in range(4):
    for i in range(4):
        fish[j].append([board[j][i*2], board[j][i*2+1]-1])
    

dx = (-1, -1, 0, 1, 1, 1, 0, -1)
dy = (0, -1, -1, -1, 0, 1, 1, 1)

belly_max = []
def dfs(fish, sx, sy, belly):
    # 상어가 처음에 물고기를 먹는다.
    belly_after = belly + fish[sx][sy][0]
    fish[sx][sy][0] = 0

    # 물고기 순서대로 찾고, 하나씩 이동하기
    for num in range(1,17):
        fx, fy = -1, -1
        for x in range(4):
            for y in range(4):
                if fish[x][y][0] == num:
                    fx, fy = x, y
        if (fx, fy) != (-1,-1):
            # 물고기 이동
            # 방향 맞을때까지 위치 바꿈
            fd = fish[fx][fy][1]
            for _ in range(8):
                nx = fx + dx[fd]
                ny = fy + dy[fd]
                if nx < 0 or ny < 0 or nx >= 4 or ny >= 4 or \
                    (nx,ny) == (sx,sy):
                    fd += 1
                    if fd == 8:
                        fd = 0
                else:
                    fish[fx][fy][1] = fd
                    fish[fx][fy], fish[nx][ny] = fish[nx][ny], fish[fx][fy]
                    break
                  
                  
    sd = fish[sx][sy][1]
    
    moved = False
    for l in range(1,4):
        snx, sny = sx+l*dx[sd], sy+l*dy[sd]
        
        if (0 <= snx < 4) and (0 <= sny < 4) and fish[snx][sny][0] > 0:
            moved = True
            dfs(copy.deepcopy(fish), snx, sny, belly_after)
            
    if not moved:
        belly_max.append(belly_after)
             

dfs(fish, 0, 0, 0)
print(max(belly_max))

```

