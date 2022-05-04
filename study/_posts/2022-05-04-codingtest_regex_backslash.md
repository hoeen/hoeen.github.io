---
layout: post

description: 정규식에 등장하는 \숫자 의 의미와 용도를 설명함.

---

# 정규식 문제 : 창영이의 일기장 - '\ (숫자)' 의 의미

[문제 링크](https://www.acmicpc.net/problem/2954)

오늘부터 정규식 문제를 조금씩 풀어보기 시작했다.

창영이의 일기장 문제의 핵심은 '(모음 aeiou 중 하나)p(앞의 모음)'을 찾아서 '모음 aeiou중 하나'로 바꾸는 것이고, 나의 답안 코드는 이러했다.

```python
import re

text = input()
pat = 'aeiou'
for i in range(5):
    text = re.sub(pat[i]+'p'+pat[i],pat[i],text)
print(text)
```

물론 이대로 해도 정답이 나온다. 하지만 모음 앞에것 찾은것을 어떻게 뒤의 조건에도 똑같이 적용할지를 몰라서 pat[i]를 이용했고, 이런 점이 좀 마음에 들지 않았다. 

백준 답안들을 살펴보니, 이런 답안이 존재했다.

```python
import re
print(re.sub(r'([aeiou])p\1', r'\1', input()))
```

여기서 **\1 의 의미를 찾아보니, 매치되는 조건들 중 첫번째 조건 (여기서는 [aeiou]) 을 인용하는 방법이었다.** 물론 백슬래시의 사용을 위해 raw string을 이용하였다.

백슬래시 숫자를 이용하면 여러모로 정규식 문제를 풀 때 많은 도움이 될 것 같다. 