---
layout: post
---

# [DFS, BFS] 게임 맵 최단거리 - 프로그래머스

완전탐색 중 미로찾기 + 최단경로 유형의 문제. 방문처리를 True, False 대신 거리값으로 치환하는 풀이로 접근하자.



## 문제 설명 - [링크](https://school.programmers.co.kr/learn/courses/30/lessons/1844)

ROR 게임은 두 팀으로 나누어서 진행하며, 상대 팀 진영을 먼저 파괴하면 이기는 게임입니다. 따라서, 각 팀은 상대 팀 진영에 최대한 빨리 도착하는 것이 유리합니다. 

지금부터 당신은 한 팀의 팀원이 되어 게임을 진행하려고 합니다. 다음은 5 x 5 크기의 맵에, 당신의 캐릭터가 (행: 1, 열: 1) 위치에 있고, 상대 팀 진영은 (행: 5, 열: 5) 위치에 있는 경우의 예시입니다.

![최단거리1_sxuruo.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/dc3a1b49-13d3-4047-b6f8-6cc40b2702a7/%E1%84%8E%E1%85%AC%E1%84%83%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A5%E1%84%85%E1%85%B51_sxuruo.png)

위 그림에서 검은색 부분은 벽으로 막혀있어 갈 수 없는 길이며, 흰색 부분은 갈 수 있는 길입니다. 캐릭터가 움직일 때는 동, 서, 남, 북 방향으로 한 칸씩 이동하며, 게임 맵을 벗어난 길은 갈 수 없습니다.
아래 예시는 캐릭터가 상대 팀 진영으로 가는 두 가지 방법을 나타내고 있습니다.

- 첫 번째 방법은 11개의 칸을 지나서 상대 팀 진영에 도착했습니다.

![최단거리2_hnjd3b.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/9d909e5a-ca95-4088-9df9-d84cb804b2b0/%E1%84%8E%E1%85%AC%E1%84%83%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A5%E1%84%85%E1%85%B52_hnjd3b.png)

- 두 번째 방법은 15개의 칸을 지나서 상대팀 진영에 도착했습니다.

![최단거리3_ntxygd.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/4b7cd629-a3c2-4e02-b748-a707211131de/%E1%84%8E%E1%85%AC%E1%84%83%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A5%E1%84%85%E1%85%B53_ntxygd.png)

위 예시에서는 첫 번째 방법보다 더 빠르게 상대팀 진영에 도착하는 방법은 없으므로, 이 방법이 상대 팀 진영으로 가는 가장 빠른 방법입니다.

만약, 상대 팀이 자신의 팀 진영 주위에 벽을 세워두었다면 상대 팀 진영에 도착하지 못할 수도 있습니다. 예를 들어, 다음과 같은 경우에 당신의 캐릭터는 상대 팀 진영에 도착할 수 없습니다.

![최단거리4_of9xfg.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/d963b4bd-12e5-45da-9ca7-549e453d58a9/%E1%84%8E%E1%85%AC%E1%84%83%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A5%E1%84%85%E1%85%B54_of9xfg.png)

게임 맵의 상태 maps가 매개변수로 주어질 때, 캐릭터가 상대 팀 진영에 도착하기 위해서 지나가야 하는 칸의 개수의 **최솟값**을 return 하도록 solution 함수를 완성해주세요. 단, 상대 팀 진영에 도착할 수 없을 때는 -1을 return 해주세요.

##### 제한사항

- maps는 n x m 크기의 게임 맵의 상태가 들어있는 2차원 배열로, n과 m은 각각 1 이상 100 이하의 자연수입니다.
  - n과 m은 서로 같을 수도, 다를 수도 있지만, n과 m이 모두 1인 경우는 입력으로 주어지지 않습니다.
- maps는 0과 1로만 이루어져 있으며, 0은 벽이 있는 자리, 1은 벽이 없는 자리를 나타냅니다.
- 처음에 캐릭터는 게임 맵의 좌측 상단인 (1, 1) 위치에 있으며, 상대방 진영은 게임 맵의 우측 하단인 (n, m) 위치에 있습니다.

------

##### 입출력 예

| maps                                                         | answer |
| ------------------------------------------------------------ | ------ |
| [[1,0,1,1,1],[1,0,1,0,1],[1,0,1,1,1],[1,1,1,0,1],[0,0,0,0,1]] | 11     |
| [[1,0,1,1,1],[1,0,1,0,1],[1,0,1,1,1],[1,1,1,0,0],[0,0,0,0,1]] | -1     |

##### 입출력 예 설명

입출력 예 #1
주어진 데이터는 다음과 같습니다.

![최단거리6_lgjvrb.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/6db71f7f-58d3-4623-9fab-7cd99fa863a5/%E1%84%8E%E1%85%AC%E1%84%83%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A5%E1%84%85%E1%85%B56_lgjvrb.png)

캐릭터가 적 팀의 진영까지 이동하는 가장 빠른 길은 다음 그림과 같습니다.

![최단거리2_hnjd3b (1).png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/d223d017-b3e2-4772-9045-a565133d45ff/%E1%84%8E%E1%85%AC%E1%84%83%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A5%E1%84%85%E1%85%B52_hnjd3b%20%281%29.png)

따라서 총 11칸을 캐릭터가 지나갔으므로 11을 return 하면 됩니다.

입출력 예 #2
문제의 예시와 같으며, 상대 팀 진영에 도달할 방법이 없습니다. 따라서 -1을 return 합니다.



## 문제 풀이

도착지 자체를 찾아내는, 미로 찾기 형태의 문제에서 BFS를 이용한다면 모든 주위 탐색 후 전진하므로 도착 지점이 나왔을 시점이 바로 최단 거리라고 할 수 있다.  
DFS는 정해진 방향으로 끝까지 도달한 뒤에 다시 이를 반복하는데, 어차피 도착지 자체를 찾아야 하는 완전탐색의 경우 BFS/DFS 두 방법으로 풀 수 있다.

하지만 이 문제는 출구는 늘 끝점으로 정해져 있지만, **최단 거리**를 구해야 한다. 그래서 한 번만 탐색을 하면 끝나는 것이 아니라 최단 경로를 찾을 때까지 같은 곳을 계속 방문할 수 있어야 한다. 그래서 visited (방문 처리) 의 값을 True/False로 넣는다면 재탐색 시 방문 히스토리가 갱신되지 못한다.

처음 접근은 queue/stack 자체에 True/False로 이루어진 visited를 계속 넣어주어 경우에 따라 visited가 달라질 수 있도록 하는 것이다. 이 방법으로 정확성 테스트는 통과했지만, 연산 시간이 초과되고 말았다.

시간 초과를 극복하지 못하고 다른 사람의 풀이를 참고했는데, 최단 거리의 유형의 핵심은 **visited 의 값을 boolean이 아닌 출발지로부터의 거리**로 두고 탐색하며 갱신하는 것이다. 이렇게 visited 값을 완전탐색으로 갱신하면 고정된 도착점의 경우 해당 지점의 visited 숫자값을 정답으로 제출하면 된다. 

결과는 시간 초과 없이 pass 했다.

## 풀이 코드

```python
from collections import deque

dx = (-1, 1, 0, 0)
dy = (0, 0, -1, 1)

def solution(maps):
    n = len(maps)
    m = len(maps[0])
    q = deque([])
    
    dist = [[-1]*m for _ in range(n)]
    q.append((0,0))  # x, y starting point
    dist[0][0] = 1
    while q:
        x, y = q.popleft()
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if 0 <= nx < n and 0 <= ny < m:
                if maps[nx][ny] == 1:
                    if dist[nx][ny] == -1 or dist[x][y] + 1 < dist[nx][ny]:
                        dist[nx][ny] = dist[x][y] + 1
                        q.append((nx,ny))
                    
                    
    return dist[n-1][m-1]
```

