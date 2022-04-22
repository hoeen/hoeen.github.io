---

layout: post



---

# 코딩테스트 - 연구소

연구소 문제 풀이 - DFS/BFS



이번 문제의 풀이에서는 조합을 구현하지 않고 for 구문을 이용해 모든 위치에 벽을 쌓아가면서 재귀적으로 dfs를 수행한 부분이 인상깊었다. 배우고 적용할 수 있어야 겠다.



코딩 과정에서의 나의 문제점은 다음과 같다:

- 자꾸 전역변수를 계속 바꾸어 가면서 문제를 풀려고 한다. 메서드를 불러와 동작시키기 위해 deepcopy를 이용한다.
  deepcopy 는 되도록 이용하지 말고, 전역변수를 효과적으로 이용하는 방법에 대해 계속 생각해야 한다.
- 얕은 복사가 이루어져 고정되어야 할 전역변수가 바뀌는 문제점이 생긴다. 이는 아마도 함수 인자에다가 전역변수를 넣어버려서 생기는 문제인 것 같다. 고정되어야 할 변수가 있으면 인자에는 넣지 말고, 리셋해야 할 시 다른 방법을 생각해 보자.



[나의 풀이]

```python
'''
0 빈칸 / 1 벽 / 2 바이러스
벽은 무조건 세개만 . 
이때 0의 개수의 최댓값을 구하시오.
'''
# 1을 임의로 세 곳에 세웠을때, 모든 장소에 바이러스 퍼뜨리고, 남은곳을 0으로 해서 카운트하자.
# 조합 이용 - 최대 64C3
# 1. 0 중에 세곳 골라서 1로 만든다
# 2. BFS로 2를 마구 퍼뜨린다!
# 3. 남은 0 개수를 센다
# 4. 0의 최댓값을 출력한다.
import copy 

def comb(input_list, n):
    ret = []
    # 1. exit
    if len(input_list) < n:
        return ret
    # 2.
    elif n == 1:
        for i in input_list:
            ret.append([i])

    # 3.
    else:
        temp = [k for k in input_list]
        for j in range(len(input_list)):
            temp.remove(input_list[j])
            for c in comb(temp, n-1):
                ret.append([input_list[j]] + c)

    return ret

def find(sci):
    zero = []
    virus = []
    for row in range(n):
        for col in range(m):
            if sci[row][col] == 0:
                zero.append((row,col))
            elif sci[row][col] == 2:
                virus.append((row,col))  
    return zero, virus #0의위치 찾아서 리스트로 반환

def execute(sci, case):
    
    temp = copy.deepcopy(sci)

    def bfs(row,col):
        dx = (-1,0,1,0)
        dy = (0,-1,0,1)
    
        if temp[row][col] == 0:
            temp[row][col] = 2
            # print('infected', (row,col))
        
        for i in range(4):
            if row+dx[i] >= 0 and row+dx[i] < n and col+dy[i] >= 0 and col+dy[i] < m:
                if temp[row+dx[i]][col+dy[i]] == 0:
                    # print('move', dx[i], dy[i])
                    bfs(row+dx[i],col+dy[i])
            
        return 
    
    
    
 
    for i in range(3):
        temp[case[i][0]][case[i][1]] = 1 # 벽 세우기 
    
    for vx, vy in virus:
        bfs(vx,vy)
    # count 0
    zero_found, _ = find(temp)
    
    
    return len(zero_found)
    
n,m = map(int, input().split())  # 세로, 가로


lab = []
for _ in range(n):
    lab.append(list(map(int, input().split())))

zero, virus = find(lab)

zero_comb = comb(zero, 3)  # 3개 고른 리스트

cand_list = []

for case in zero_comb: # [(1, 4), (3, 0), (3, 3)]= case
    cand_list.append(execute(lab, case))
    
    
print(max(cand_list))






```





[모범적 풀이 - 책]

```python
n, m = map(int, input().split())
data = [] # 초기 맵 리스트
temp = [[0] * m for _ in range(n)] # 벽을 설치한 뒤의 맵 리스트

for _ in range(n):
    data.append(list(map(int, input().split())))


# 4가지 이동방향
dx = [-1,0,1,0]
dy = [0,-1,0,1]

result = 0

# 깊이 우선 탐색(DFS)를 이용해 각 바이러스가 사방으로 퍼지도록 하기
def virus(x, y):
    for i in range(4):
        nx = x + dx[i]
        ny = x + dy[i]
        # 상하좌우 중에서 바이러스가 퍼질 수 있는 경우 
        if nx >= 0 and nx < n and ny >= 0 and ny < m:
            if temp[nx][ny] == 0:
                # 해당 위치에 바이러스 배치, 다시 재귀적으로 수행
                temp[nx][ny] = 2
                virus(nx,ny)
                # 재귀때는, input값을 같게 하자.

# 현재 맵에서 안전 영역의 크기 계산하는 메서드
def get_score():
    score = 0
    for i in range(n):
        for j in range(m):
            if temp[i][j] == 0:
                score += 1
    return score

# 깊이 우선 탐색(DFS)를 이용하여 울타리를 설치하면서, 매번 안전 영역의 크기 계산
def dfs(count):
    global result
    if count == 3:
        for i in range(n):
            for j in range(m):
                temp[i][j] = data[i][j]

        # 각 바이러스의 위치에서 전파 진행
        for i in range(n):
            for j in range(m):
                if temp[i][j] == 2:
                    virus(i, j)
        
        # 안전영역의 최댓값 계산
        result = max(result, get_score())
        return
    
    # 빈 공간에 울타리 설치
    for i in range(n):
        for j in range(m):
            if data[i][j] == 0:
                data[i][j] = 1
                count += 1
                dfs(count)
                data[i][j] = 0 # dfs 이후에 울타리 뺌
                count -= 1

dfs(0)
print(result)

```

