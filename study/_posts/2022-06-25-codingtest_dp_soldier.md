---
layout: post

description: 병사 배치하기 - 다이나믹 프로그래밍 풀이

---

# [DP] 코딩테스트 문제풀이 - 병사 배치하기

## 문제 설명 - [링크](https://www.acmicpc.net/problem/18353)



## 문제 풀이

이 문제는 점화식을 떠올리기가 매우 어려웠다. 병사를 내림차순으로 배치시킬 때 최대한 적게 열외시키는 것이 목적인데 일반적인 F(k), F(k-1), F(k+1) 로 점화식을 만드는 방식으로는 문제의 핵심 점화식을 구현할 수 없었다.

핵심 점화식은 i < k 인 모든 i에 대하여 F(i) < F(k) 인 조건일 때 다음과 같다.   


$$
F(k) = max(F(k), F(i)+1))[i < k, F(i) < F(k)]  
$$




쉽게 이야기해서 F(k)로 가기 위한 항들을 **단순히 앞, 뒤에서 찾는 것이 아니라 이전 모든 F(i)를 이용하여 F(k)를 계산해 낸다**는 뜻이다.   
또한 최소 하나의 병사는 있어야 하므로 tabular를 이용한 bottom-up 식 DP 풀이에서 table의 요소는 1로 넣어야 한다. 

이러한 방식의 점화식 유형도 있다는 것을 익혀두어야 겠다.



## 풀이 코드

```python
# 내림차순으로 배치되며 최대한 적게 열외시키자.
n = int(input())
solds = list(map(int, input().split()))

# f(k) = max(f(k), f(i)+1) for i in range(1~k) if f(i) < f(k)

tab = [1]*n
solds.reverse()

for k in range(n): #0~n-1
    for i in range(k): #0~k-1
        if solds[i] < solds[k]:
            tab[k] = max(tab[k], tab[i]+1)

# print(tab)
print(n - max(tab))
```



