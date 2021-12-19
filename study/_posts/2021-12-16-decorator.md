---
layout: post
---

# Python class 관련 중요 개념

> python 클래스의 데코레이터 등 중요한 개념에 대해서 설명합니다.



## 클래스에서 언더바 (_) 의 사용법  

---



파이썬에서 (_) 를 사용하는 경우에는

1. 마지막 값을 저장하고 싶을 때
2. 값을 무시하고 싶을 때

등이 있습니다. 

여기에선 python 클래스에서 (_) 를 사용하여 변수 및 메서드를 사용하는 방식에 대하여 설명합니다.

### 1. 언더바가 앞에 하나 붙은 경우 (ex: _variable)

이 경우는 **모듈 내에서만** 해당 변수/함수를 사용하겠다는 의미입니다. 하지만 완전히 private하지는 않기 때문에 여전히 접근하거나 사용할 수 있습니다.

하지만 외부에서 해당 모듈을 import 하는 경우에는 앞에 언더바 (_) 가 붙은 변수나 함수는 import 하지 않습니다.





```python
# 외부에서 아래 두 함수로 이루어진 모듈을 import 했을 경우
def name(input_name):
    return input_name

def _hidden_name(input_name):
    return 'hidden_'+input_name
  
---  
>>> _hidden_name
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name '_hidden_name' is not defined
```

``` python
# 직접 함수를 import 한 경우에는 사용 가능
>>> from test import _hidden_name
>>> print(_hidden_name('name'))
'hidden_name'
```



### 2. 언더바가 앞에 둘 붙은 경우 (ex: __variable)

네임 맹글링을 적용하는 경우입니다. Mangling의 뜻 그대로 이름을 '짓이겨' 바꾸어서 본연의 이름으로 접근 불가하게 만듭니다.

``` python
# 맹글링 적용된 변수를 직접 호출 시 AttributeError 발생
class testname:
    def __init__(self, input):
        self.name = input
        self.__hidden_name = input+" hidden"
        
---
>>> a = testname('car')
>>> a.name
'car'
>>> a.__hidden_name
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'testname' object has no attribute '__hidden_name'   
```



### 3. 앞 뒤로 언더바가 2개씩 붙은 경우 (ex: __variable__)

이 경우로 직접 함수를 만드는 경우는 없고, 거의 오버라이딩 할 때 많이 사용됩니다.

이 룰이 적용된 함수는 **매직 메소드**나 **던더 메소드**라고 부릅니다.

<br>

<br>



- __ repr __ 와  __ str __ 의 차이

repr : 해당 객체를 설명해줄 수 있는, 그리고 화면에 출력될 수 있는 문자열 표현을 반환하는 함수.



""" object란?

​        클래스 계층의 기본 클래스입니다.

​        호출되면 인수를 허용하지 않으며 

​        인스턴스 특성이 없고 값을 지정할 수 없는 새로운 기능 없는 인스턴스를 반환합니다.

​        """



refs : https://www.daleseo.com/python-property/

https://docs.python.org/3/library/functions.html#property

https://bluese05.tistory.com/30



언더바 : https://tibetsandfox.tistory.com/20 을 참고하여 작성하였습니다.