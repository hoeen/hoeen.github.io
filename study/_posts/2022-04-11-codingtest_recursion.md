---
layout: post

---

# 코딩테스트 - 문자열 압축

2020 Kakao Blind Recruitment. 재귀를 이용한 풀이 그리고 setrecursionlimit 설정

[문제 링크](https://programmers.co.kr/learn/courses/30/lessons/60057#qna)

## 문제 설명

데이터 처리 전문가가 되고 싶은 **"어피치"**는 문자열을 압축하는 방법에 대해 공부를 하고 있습니다. 최근에 대량의 데이터 처리를 위한 간단한 비손실 압축 방법에 대해 공부를 하고 있는데, 문자열에서 같은 값이 연속해서 나타나는 것을 그 문자의 개수와 반복되는 값으로 표현하여 더 짧은 문자열로 줄여서 표현하는 알고리즘을 공부하고 있습니다.
간단한 예로 "aabbaccc"의 경우 "2a2ba3c"(문자가 반복되지 않아 한번만 나타난 경우 1은 생략함)와 같이 표현할 수 있는데, 이러한 방식은 반복되는 문자가 적은 경우 압축률이 낮다는 단점이 있습니다. 예를 들면, "abcabcdede"와 같은 문자열은 전혀 압축되지 않습니다. "어피치"는 이러한 단점을 해결하기 위해 문자열을 1개 이상의 단위로 잘라서 압축하여 더 짧은 문자열로 표현할 수 있는지 방법을 찾아보려고 합니다.

예를 들어, "ababcdcdababcdcd"의 경우 문자를 1개 단위로 자르면 전혀 압축되지 않지만, 2개 단위로 잘라서 압축한다면 "2ab2cd2ab2cd"로 표현할 수 있습니다. 다른 방법으로 8개 단위로 잘라서 압축한다면 "2ababcdcd"로 표현할 수 있으며, 이때가 가장 짧게 압축하여 표현할 수 있는 방법입니다.

다른 예로, "abcabcdede"와 같은 경우, 문자를 2개 단위로 잘라서 압축하면 "abcabc2de"가 되지만, 3개 단위로 자른다면 "2abcdede"가 되어 3개 단위가 가장 짧은 압축 방법이 됩니다. 이때 3개 단위로 자르고 마지막에 남는 문자열은 그대로 붙여주면 됩니다.

압축할 문자열 s가 매개변수로 주어질 때, 위에 설명한 방법으로 1개 이상 단위로 문자열을 잘라 압축하여 표현한 문자열 중 가장 짧은 것의 길이를 return 하도록 solution 함수를 완성해주세요.

### 제한사항

- s의 길이는 1 이상 1,000 이하입니다.
- s는 알파벳 소문자로만 이루어져 있습니다.

##### 입출력 예

| s                            | result |
| ---------------------------- | ------ |
| `"aabbaccc"`                 | 7      |
| `"ababcdcdababcdcd"`         | 9      |
| `"abcabcdede"`               | 8      |
| `"abcabcabcabcdededededede"` | 14     |
| `"xababcdcdababcdcd"`        | 17     |

### 입출력 예에 대한 설명

**입출력 예 #1**

문자열을 1개 단위로 잘라 압축했을 때 가장 짧습니다.

**입출력 예 #2**

문자열을 8개 단위로 잘라 압축했을 때 가장 짧습니다.

**입출력 예 #3**

문자열을 3개 단위로 잘라 압축했을 때 가장 짧습니다.

**입출력 예 #4**

문자열을 2개 단위로 자르면 "abcabcabcabc6de" 가 됩니다.
문자열을 3개 단위로 자르면 "4abcdededededede" 가 됩니다.
문자열을 4개 단위로 자르면 "abcabcabcabc3dede" 가 됩니다.
문자열을 6개 단위로 자를 경우 "2abcabc2dedede"가 되며, 이때의 길이가 14로 가장 짧습니다.

**입출력 예 #5**

문자열은 제일 앞부터 정해진 길이만큼 잘라야 합니다.
따라서 주어진 문자열을 x / ababcdcd / ababcdcd 로 자르는 것은 불가능 합니다.
이 경우 어떻게 문자열을 잘라도 압축되지 않으므로 가장 짧은 길이는 17이 됩니다.



---

## 문제 풀이

이 문제를 나는 재귀로 접근하였다.

n개의 문자열로 탐색한다고 할때, 처음 n개와 그다음 n개가 같은지 탐색한 뒤 [n:]부터 다시 함수를 수행하는 방식이다.

재귀를 구현해서 처음으로 문제를 풀었기 때문에 나 자신이 조금 더 발전했구나 라는 보람을 느꼈다.

테스트 케이스 4개가 끝까지 골치아프게 했는데, 엣지케이스를 살펴보았음에도 불구하고 해결방법을 찾지 못하다가   

"Recursion limit 때문은 아닐까" 생각이 들었고 이것이 원인이어서 수정 후 모든 테스트 케이스를 통과할 수 있었다.



파이썬에서 재귀를 구현할 때 recursion limit이 존재하며, 이를 넘으면 에러가 발생한다. 따라서 최대 재귀 횟수를 조절할 수 있는데, 이는 다음과 같이 설정할 수 있다.

```python
import sys
sys.setrecursionlimit(10**9) # 재귀 제한을 늘려서 런타임 에러를 방지
```



이렇게 적용한 풀이는 다음과 같다.

```python
import sys
sys.setrecursionlimit(10**9)

def solution(input_str):
    
    if len(input_str) == 1:
        return 1
    global output_str, temp
    output_str = ''
    temp = 1

    def find_nword(in_str, n):
        global output_str, temp
        if len(in_str) < n*2:
            if temp != 1:
                output_str += (str(temp)+in_str)
            else:
                output_str += in_str

            return 

        start = in_str[:n]
        find = in_str[n:n*2]

        if start != find:
            # print('dif')
            if temp == 1:
                output_str += start

            else:
                output_str += (str(temp)+start)
                temp = 1

            # 다시 n번째 뒤부터 탐색
            find_nword(in_str[n:], n)


        else: # 같은 경우
            # print('same')
            # 등장 횟수를 업데이트
            temp += 1
            # n개 다음부터 다시 탐색
            find_nword(in_str[n:], n)


    

    for str_len in range(1, len(input_str)//2+1):
        find_nword(input_str, str_len)
        res = len(output_str)
        if str_len == 1:
            before = len(output_str)
        if res < before:
            before = res
        # print('str_len:', str_len, 'output:', output_str)
        output_str = ''
        temp = 1

    return before
```



