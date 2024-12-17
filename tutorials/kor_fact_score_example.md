# LLM reliability: FactScore Example - KorFactScore

최종 업데이트: 2024-12-04

이 문서는 `pnpxai`를 활용하여 LLM reliability 알고리즘 중 하나인 FactScore를 산출하는 방법을 소개합니다.

## FactScore

FactScore는 평가 대상 LLM의 생성결과(generation)를
- 원소사실생성기(atomic fact generator)를 통해 다수의 원소사실(atomic fact)로 분해한 후,
- 판별기(scorer)를 통해 원소사실별로 ground-truth 데이터(knowledge source)와 비교하여 사실여부를 판단하여,
- 최종적으로, 원소사실별로 판단된 사실성여부를 합산(aggregate function)하여 LLM의 사실성을 평가
  
하는 알고리즘입니다. `pnpxai` 기본 사용법은 다음과 같습니다.

```python
from pnpxai.llm import FactScore

fs = FactScore(
    atomic_fact_generator=atomic_fact_generator,
    knowledge_source=knowledge_source,
    scorer=scorer,
    aggregate_fn=aggregate_fn,
)

PROMPT = 'My prompt of {topic}'
generation = my_llm(PROMPT.format(topic=topic))
outputs = fs.evaluate(topic, generation)
```

**References**
- FactScore (https://arxiv.org/abs/2305.14251)
- FactScore github (https://github.com/shmsw25/FActScore)

## KorFactScore

이 튜토리얼에서는 한국어 기반 FactScore 알고리즘 "KorFactScore"를 이용해 `gpt-3.5-turbo-0125`의 한국어 생성결과에 대한 사실성 점수를 산출합니다. 튜토리얼 실행 전 다음 단계의 준비가 필요합니다:
- [KorFactScore 깃헙 페이지](https://github.com/ETRI-XAINLP/KorFactScore)를 참고하여 KorFactScore를 설치해주세요
- [KorFactScore 깃헙 페이지](https://github.com/ETRI-XAINLP/KorFactScore)를 참고하여 적절한 위치에 knowledge source의 데이터베이스를 생성해주세요 (약 8시간 소요, 소요시간은 구동환경에 따라 다를 수 있습니다)
- `KorFactScore/api.keys`에 openai api key를 입력해주세요

본 시스템은 외부지식 검색기로써 BM25가 적용되었으며, ETRI에서 개발된 검색기를 적용할 경우 성능이 개선
됩니다. 성능 개선 수치는 https://github.com/ETRI-XAINLP/KorFactScore 의 "결과 (1) System
의 사실판단 성능" 절을 확인해 주세요.

**References**
- KorFactScore github (https://github.com/ETRI-XAINLP/KorFactScore)

### Prepare KorFactScore

**Register the package**


```python
kfs_dir = '../../KorFactScore/'

import sys

sys.path.insert(0, kfs_dir)
```

#### KorFactScore 원소사실생성기 (atomic fact generator)

원소사실생성기로 KorFactScore에 구현된 원소사실생성기 클래스 `AtomicFactGenerator`의 `run` method를 활용합니다(`kfs_af_generator`). 해당 generator는 LLM의 생성결과로부터 (문장, 원소사실) 튜플의 리스트를 출력하는 반면, `pnpxai`의 `FactScore`는 원소사실의 리스트를 출력하는 generator를 입력으로 요구하므로 (`Callable[[str], List[str]]`), 이를 위해 간단한 wrapper(`convert_kfs_af_generator_to_pnpxai_af_generator`) 또한 정의합니다.


```python
from typing import Callable, List
import os
from factscore.atomic_facts import AtomicFactGenerator


data_dir = '.cache/factscore/'
cache_dir = '.cache/factscore/'
api_keys_filename = 'api.keys'

# create af_generator
kfs_af_generator = AtomicFactGenerator(
    key_path=os.path.join(kfs_dir, api_keys_filename),
    demon_dir=os.path.join(kfs_dir, data_dir, 'demos'),
    gpt3_cache_file=os.path.join(kfs_dir, cache_dir, '2_af_cache.pkl'),
    af_model_name='gpt-3.5-turbo-0125',
    demon_fn='k_demons_v1.json',
).run

# define a wrapper to use it in pnpxai FactScore
def convert_kfs_af_generator_to_pnpxai_af_generator(func):
    def wrapper(*args, **kwargs):
        curr_afs, _ = func(*args, **kwargs)
        curr_afs = [fact for _, facts in curr_afs for fact in facts]
        return curr_afs
    return wrapper

# wrap the method
my_af_generator: Callable[
    [str], List[str] # Returns a list of atomic facts from generation
] = convert_kfs_af_generator_to_pnpxai_af_generator(kfs_af_generator)
```

    /home/geonhyeong/store/miniconda3/envs/pnpxai-kfs/lib/python3.9/site-packages/torch/cuda/__init__.py:129: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 11040). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.)
      return torch._C._cuda_getDeviceCount() > 0


    @AtomicFactGenerator
    Model loading... gpt-3.5-turbo-0125


**Atomic Fact Generator 출력 결과 예시**


```python
example = '유관순은 조선시대의 왕으로, 독립운동을 이끌었다.'
atomic_facts_example = my_af_generator(example)
print('Atomic facts:', atomic_facts_example)
```

    [Kss]: Because there's no supported C++ morpheme analyzer, Kss will take pecab as a backend. :D
    For your information, Kss also supports mecab backend.
    We recommend you to install mecab or konlpy.tag.Mecab for faster execution of Kss.
    Please refer to following web sites for details:
    - mecab: https://github.com/hyunwoongko/python-mecab-kor
    - konlpy.tag.Mecab: https://konlpy.org/en/latest/api/konlpy.tag/#mecab-class
    
    Generating AFs >> : 100%|███████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 3251.40it/s]

    ... Model loading... gpt-3.5-turbo-0125


    


    Atomic facts: ['유관순은 조선시대의 왕이었다.', '유관순은 독립운동을 이끌었다.']


#### KorFactScore knowledge source: kowiki-20240301

`pnpxai`의 `FactScore`는 (1) 문장생성에 사용된 topic과 (2) 하나의 원소사실 입력으로부터 ground truth 지식을 출력하는 함수 `knowledge_source: Callable[[str, str], List[Dict[str, str]]]`를 입력으로 요구합니다. 다음과 같이 KorFactScore의 데이터베이스 클래스 `DocDB` 및 `Retrieval`을 활용하여 knowledge source를 구성합니다.


```python
from typing import Dict
import functools
from factscore.retrieval import DocDB, Retrieval


KNOWLEDGE_SOURCE_NAME = 'kowiki-20240301'
RETRIEVAL_CACHE_FILENAME = '3_retrieval-{retrieval_type}-k{retrv_k}'

# create knowledge source
db = Retrieval(
    db=DocDB(
        db_path=os.path.join(kfs_dir, data_dir, f'{KNOWLEDGE_SOURCE_NAME}.db'),
        data_path=os.path.join(kfs_dir, data_dir, f'{KNOWLEDGE_SOURCE_NAME}.jsonl'),
        tokenizer_path=None,
    ),
    cache_path=os.path.join(kfs_dir, cache_dir, f'{RETRIEVAL_CACHE_FILENAME}.json'),
    embed_cache_path=os.path.join(kfs_dir, cache_dir, f'{RETRIEVAL_CACHE_FILENAME}.pkl'),
    batch_size=256,
    ckp_path=None,
    retrieval_type='bm25',
)

my_knowledge_source: Callable[
    [str, str], # (a topic, "an" atomic fact)
    List[Dict[str, str]] # A list of knowledges (passages)
] = functools.partial(db.get_passages, k=20)
```

**Knowledge Source 출력 결과 예시**


```python
knowledges_example = my_knowledge_source('유관순', atomic_facts_example[0])
knowledges_example
```




    [{'title': '유관순',
      'text': '��상한다.\n취미.\n유관순의 취미는 태극기 만들기이다\n유관순은 우리나라를 사랑한다</s>'},
     {'title': '유관순',
      'text': '� 소요죄 및 《보안법》 위반죄로 징역 5년을 선고받은 유관순은 이에 불복해 항소하였고, 같은 해 6월 30일 경성복심법원에서 징역 3년을 선고받은 후 상고를 포기하였다. 유관순은 경성복심법원 재판 당시 일제의 한국점령'},
     {'title': '유관순',
      'text': '독교운동가이자 종교권력감시시민연대 대표인 김상구는 유관순이 사후 박인덕, 전영택, 일부 기독교인들의 선전도구로 이용되었다는 주장을 펼쳤다. 그는 서대문형무소의 유관순 기록과 당시 언론 보도 등을 근거�'},
     {'title': '유관순',
      'text': '��하지 않았다.\n투옥과 사인.\n재판.\n유관순은 천안경찰서 일본헌병대에 투옥되었다가 곧 공주경찰서 감옥으로 이감되었고, 공주지방법원에서 구속 상태로 재판을 받았다. 1919년 5월 9일 공주지방법원의 1심재판에�'},
     {'title': '유관순',
      'text': '��려 이화학당이 폐교하자 3월 8일 열차편으로 천안으로 돌아왔다.\n만세 운동.\n고향으로 돌아온 유관순은 교회와 청신학교(靑新學校)를 찾아다니며, 서울에서의 독립 시위운동 상황을 설명하고 천안에서도 만세시위를 전�'},
     {'title': '유관순',
      'text': '��고자 하는 선각자들이었다.\n1919년 3·1 운동이 일어나자, 이화학당 고등과 1년생이었던 유관순은 만세시위에 참가하였고, 연이어 3월 5일의 서울 만세시위에도 참가하였다. 그 뒤로부터는 총독부 학무국에서 임시휴교령을 �'},
     {'title': '유관순',
      'text': '��서 독립만세시위를 벌였다. 아우내 만세시위 주동자로 일제 헌병에 붙잡힌 유관순은 미성년자인 점을 감안하여 범죄를 인정하고 수사에 협조하면 선처하겠다는 제안을 거절하였고, 이후 협력자와 시위 가담자를 발�'},
     {'title': '유관순',
      'text': '��씨로 추정되고 있는데, 유중권과 이씨를 포함해 20명이 같은 장소·날짜·상황에서 순국했다는 자료의 내용은 1987년에 작성된 독립유공자 공훈록에서 유관순 열사의 아버지 유중권과 어머니 "이씨(李氏)" 등 열 아홉명이 현장�'},
     {'title': '유관순',
      'text': '�순 열사의 애국애족 정신을 기리고 그 얼을 오늘에 되살려 진취적이고 미래지향적인 사고로 국가와 지역사회발전에 이바지한 여성(여학생) 또는 단체를 전국에서 선발한다. 유관순 횃불상은 고등학교 1학년 여학생에게 �'},
     {'title': '유관순',
      'text': ' 1년 6개월로 감형되었다. 그러나, 유관순은 서대문형무소 복역 중에도 옥안에서 독립만세를 고창하여, 고문을 당했으며 1920년 9월 28일 오전 8시 20분, 출소를 1일 남기고 서대문형무소에서 방광이 파열하여 옥사하였다.\n유�'},
     {'title': '유관순',
      'text': '�한 일부 개신교 세력에 대해 폭로하고 유관순은 개신교계의 친일 전력을 덮어주는 동시에 선교 전략에 활용되는 ‘시대의 아이콘’으로 이용되었다는 의혹을 제기하였다.\n이화학당 출신 인사의 은폐 의혹.\n광복 직후인 1940년대'},
     {'title': '유관순',
      'text': "��시 동남구 병천면 용두리의 생가가 복원되어 1991년에 사적 제230호로 지정되었다. 천안 유관순 열사 유적과 천안종합운동장 내 '유관순체육관'은 유관순의 이름을 딴 것이다. 해방 후 박인덕 등에 의해 기념사업이 추진되었는�"},
     {'title': '유관순',
      'text': "심 선고형은 '징역 5년'임이 확인되었다.\n정치적, 종교적 목적의 악용 논란.\n유관순 사후 그를 정치적으로 이용할 목적 또는 자신들의 친일행위를 덮기 위한 일부 기독교인들에 의해 과도하게 띄워졌다는 견해도 있다. 반기"},
     {'title': '유관순',
      'text': '��, 이 때문에 박인덕 등이 자신들의 친일 의혹을 덮기 위한 불순한 의도로 이화학당 학생이었던 유관순 열사를 부각시켰다는 의혹도 제기되고 있다.\n성명.\n본관은 고흥 유씨이다. 두음법칙과 관련하여 성명 표기에 대해 과거�'},
     {'title': '유관순',
      'text': '�� 유관순 열사는 박인덕 등 친일 경력자들이 해방 후 자신의 전력을 덮고 개신교 선교 전략에 이용하는 도구로 만들어낸 영웅이라고 주장하면서, 2011년에 &lt;믿음이 왜 돈이 되는가&gt;(해피스토리, 2011)라는 책을 통해 유관순을 악�'},
     {'title': '유관순',
      'text': '합하기 위한 연락원의 한 사람이 되어 다른 연락원들과 함께 인근 지역을 돌아다니며 주민들을 상대로 시위운동 참여를 설득했다.\n4월 1일 수천 명의 군중이 모인 가운데 조인원의 선도로 시위가 시작되자 유관순은 시위대 선두�'},
     {'title': '유관순',
      'text': '�순 부모의 사인.\n이 자료에는 아우내 만세 운동 당일의 시위자도 기재되어 있는데, 유관순 열사의 부친인 유중권 열사의 기록이 가장 먼저 나온다. 일시는 기미년(己未年·1919년) 3월 1일, 장소는 천안군 병천면 병천리이다. 이는 1987�'},
     {'title': '유관순',
      'text': ' 온전한 몸에다 수의를 입혔다"고 밝혔다. 열사의 사인은 구타 등 고문 후유증이었다.\n유관순 우상화 배경 관련 의혹.\n그들은 유관순을 실제 이상의 영웅으로 신화화하는 데에 열을 올렸다. 박인덕과 최초로 유관순의 전기를 쓴 �'},
     {'title': '유관순',
      'text': '��종성(李鍾成) 등의 주동으로 3.1 만세 운동에 호응하는 만세 시위운동을 계획했으나 사전에 구금당해 실행하지 못했다. 유관순은 부친 유중권의 주선으로 3월 9일 밤 교회 예배가 끝난 뒤 마을 속장 조인원(趙仁元), 지역 유지 이'},
     {'title': '유관순',
      'text': '��제 받고, 졸업 후에 교사로 일하는 학생이었다. 파랑새어린이에서 출판한 유관순 전기에 따르면, 이화학당 학생들은 기숙사 생활을 했고, 김장, 빨래 등은 여럿이 함께 해야 하는 일이므로 학생들이 함께 일을 함을 뜻하는 �'}]



#### Scorer

`pnpxai`의 `FactScore`는 (1) 문장생성에 사용된 topic, (2) 원소사실생성기로부터 생성된 하나의 원소사실, (3) knowledge source로부터 query된 ground-truth 지식들(knowledges)로부터 판별 결과를 출력하는 `scorer: Callable[[str, str, List[Dict[str, str]]], Any]`를 입력으로 요구합니다. 이 튜토리얼에서는 llm prompting을 통해 원소사실의 참/거짓 여부를 판단하는 판별기를 다음과 같이 정의합니다.


```python
from typing import Any
from openai import OpenAI


SCORING_DEFINITION = 'Answer the question about {topic} based on the given context.\n\n'
SCORING_CONTEXT = 'Title: {title}\nText: {text}\n\n'
SCORING_PROMPT = '{definition}\n\nInput: {atomic_fact} True or False?\nAnswer:'

def llm(prompt: str, model='gpt-3.5-turbo-0125') -> str:
    client = OpenAI()
    res = client.chat.completions.create(
        model=model,
        messages=[{
            'role': 'user',
            'content': prompt,
        }],
    )
    return res.choices[0].message.content

def load_openai_api_key(key_path):
    assert os.path.exists(key_path), f"Please place your OpenAI APT Key in {key_path}."
    with open(key_path, 'r') as f:
        keys = dict(line.strip().split('=') for line in f if line.strip())
    # OpenAI API 키 설정
    api_key = keys.get('openai', None)
    os.environ['OPENAI_API_KEY'] = api_key

def kfs_scorer(topic, atomic_fact, knowledges, scoring_model=None):
    definition = SCORING_DEFINITION.format(topic=topic)
    context = ''
    for knowledge in knowledges:
        context += SCORING_CONTEXT.format(
            title=knowledge['title'],
            text=knowledge['text'].replace('<s>', '').replace('</s>', '')
        )
    definition += context.strip()
    prompt = SCORING_PROMPT.format(
        definition=definition.strip(),
        atomic_fact=atomic_fact.strip(),
    )
    output = scoring_model(prompt),
    ans = output[0].lower()
    if "true" in ans or "false" in ans:
        if "true" in ans and "false" not in ans:
            is_supported = True
        elif "false" in ans and "true" not in ans:
            is_supported = False
        else:
            is_supported = ans.index("true") > ans.index("false")
    else:
        is_supported = all([keyword not in ans.lower().translate(
            str.maketrans("", "", string.punctuation)).split() for keyword in
                            ["not", "cannot", "unknown", "information"]])
    return is_supported


# create scorer
load_openai_api_key(os.path.join(kfs_dir, api_keys_filename))
scoring_model = functools.partial(llm, model='gpt-3.5-turbo-0125')
my_scorer: Callable[
    [str, str, List[Dict[str, str]]], # (a topic, an atomic fact, knowledges (passages))
    Any # Any type of score. In this case, bool.
] = functools.partial(kfs_scorer, scoring_model=scoring_model)
```

**Scorer 출력 예시**


```python
scores_example = [my_scorer('유관순', af, knowledges_example) for af in atomic_facts_example]
print(list(zip(atomic_facts_example, scores_example)))
```

    [('유관순은 조선시대의 왕이었다.', False), ('유관순은 독립운동을 이끌었다.', True)]


#### Aggregate Function

`pnpxai`의 `FactScore`는 `scorer`부터 출력된 원소사실별 점수를 합산하는 `aggregate_fn`을 입력으로 요구합니다. 미입력시 합산 점수는 출력되지 않습니다. 이 튜토리얼에서는 다음과 같이 원소사실별 참/거짓 판단여부 중 참의 비율로 합산하는 함수를 사용합니다.


```python
import numpy as np

# define score aggregation function
my_aggregate_fn: Callable[
    [List[Any]], Any # Returns any type of scalar from the list of scores. In this case, float.
] = lambda scores: np.array(scores).mean()
```

**Aggregate Function 출력 예시**


```python
my_aggregate_fn(scores_example)
```




    np.float64(0.5)



### Play with `pnpxai`

위 섹션에서 준비한 원소사실생성기(`my_af_generator`), knowledge source(`my_knowledge_source`), 판별기(`my_scorer`), 합산함수(`my_aggregate_fn`)를 이용해 다음과 같이 `FactScore`를 구성합니다.


```python
from pnpxai.llm import FactScore

fs = FactScore(
    atomic_fact_generator=my_af_generator,
    knowledge_source=my_knowledge_source,
    scorer=my_scorer,
    aggregate_fn=my_aggregate_fn,
)
```

평가할 LLM결과를 생성하고, fact score를 산출합니다.


```python
PROMPT = '{topic}의 약력에 대해 알려줘'
topic = '유관순'

generated = llm(PROMPT.format(topic=topic), model='gpt-3.5-turbo-0125')
print(generated)

outputs = fs.evaluate(topic, generated)
```

`FactScore.evaluate`은 `FactScoreOutput`을 출력하며, `FactScoreOutput`는 다음과 같이 원소사실을 나타내는 `FactScoreOutput.atomic_facts`, 판별결과를 나타내는 `FactScoreOutput.scores`, 마지막으로 합산점수를 나타내는 `FactScoreOutput.aggregated_score`로 구성되어있습니다.


```python
for af, s in zip(outputs.atomic_facts, outputs.scores):
    print(af, s)
```

    유관순은 대한독립운동가이다. True
    유관순은 대한민국의 독립운동사에 큰 영향을 끼친 인물 중 하나이다. True
    유관순은 1902년 평안북도 안동에서 태어났다. False
    유관순은 조선여족 유가족의 아이로 태어났다. True
    그녀는 조선 여성 최초로 조선일보에 글을 발표했다. False
    그녀는 19세에 독일로 유학을 떠났다. False
    그녀는 독립운동을 경험했다. True
    1919년 3.1 운동이 일어났다. True
    유관순은 일본 식민지에서 국민학교에서 일본의 고문에 저항하기 위해 평양에서 학생 항일운동을 조직했다. True
    그녀는 1919년 3월 1일 대한민국 임시 정부의 독립군으로 전환하였다. False
    그녀는 독립운동가 윤봉길을 돕으러 항일운동을 진행했다. False
    그녀는 1920년 일본 경찰에 체포되었다. True
    그녀는 올해 19세의 나이로 아사히. False
    1920년 8월 1일 유관순은 형사상 시선 과민으로 의식을 잃었다. False
    1920년 8월 1일 유관순은 무법시장에 돌아가 프랜차이즈를 시작했다. False
    유관순은 1920년 8월 11일 민족비례 수상병에 처해졌다. False
    유관순은 혹독한 고문과 고통을 받았다. True
    유관순은 1920년 9월 1일 19세의 나이로 감옥에서 사망했다. False
    유관순은 대한민국의 역사에 기록된 여인 중 한 명이다. True
    유관순은 대한민국 역사상 가장 영웅적인 여성 중 하나로 기억되고 있다. True



```python
outputs.aggregated_score
```




    np.float64(0.5)


