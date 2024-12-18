'''
이 스크립트는 "KorFactScore"를 기반으로 `pnpxai`의 FactScore 사용법을 소개하기 위해 작성되었습니
다.

스크립트 실행 전 다음 단계의 준비가 필요합니다:
- Clone the repo: git clone https://github.com/ETRI-XAINLP/KorFactScore
- Create database for the knowledge source 'kowiki-20240301' in a right location

* 본 시스템은 외부지식 검색기로써 BM25가 적용되었으며, ETRI에서 개발된 검색기를 적용할 경우 성능이 개선
됩니다. 성능 개선 수치는 https://github.com/ETRI-XAINLP/KorFactScore 의 "결과 (1) System
의 사실판단 성능" 절을 확인해 주세요.

References
- KorFactScore (https://github.com/ETRI-XAINLP/KorFactScore)
'''

from typing import Callable, List, Dict, Any, Optional

import argparse
import os
import functools

from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

from pnpxai.llm import FactScore


parser = argparse.ArgumentParser()
parser.add_argument('--kfs_dir', type=str, required=True)
parser.add_argument('--data_dir', type=str, default='.cache/factscore/')
parser.add_argument('--cache_dir', type=str, default='.cache/factscore/')
parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo-0125')
parser.add_argument('--af_model_name', type=str, default='gpt-3.5-turbo-0125')
parser.add_argument('--scoring_model_name', type=str, default='gpt-3.5-turbo-0125')
parser.add_argument('--retrieval_type', type=str, default='bm25')
parser.add_argument('--retrieval_k', type=int, default=5)
parser.add_argument('--topic', type=str, default='장기하')
parser.add_argument('--api_keys', type=str, default='api.keys')


KNOWLEDGE_SOURCE_NAME = 'kowiki-20240301'
RETRIEVAL_CACHE_FILENAME = '3_retrieval-{retrieval_type}-k{retrv_k}'

SCORING_DEFINITION = 'Answer the question about {topic} based on the given context.\n\n'
SCORING_CONTEXT = 'Title: {title}\nText: {text}\n\n'
SCORING_PROMPT = '{definition}\n\nInput: {atomic_fact} True or False?\nAnswer:'

PROMPT = '{topic}의 약력에 대해 말해줘'


def register_korfactscore(kfs_dir):
    import sys
    sys.path.insert(0, kfs_dir)


def load_openai_api_key(key_path):
    assert os.path.exists(key_path), f"Please place your OpenAI APT Key in {key_path}."
    with open(key_path, 'r') as f:
        keys = dict(line.strip().split('=') for line in f if line.strip())
        # api_key = f.readline()
    # OpenAI API 키 설정
    api_key = keys.get('openai', None)
    os.environ['OPENAI_API_KEY'] = api_key


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


def convert_kfs_af_generator_to_pnpxai_af_generator(func):
    def wrapper(*args, **kwargs):
        curr_afs, _ = func(*args, **kwargs)
        curr_afs = [fact for _, facts in curr_afs for fact in facts]
        return curr_afs
    return wrapper


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


def main(args):
    register_korfactscore(args.kfs_dir)
    from factscore.atomic_facts import AtomicFactGenerator
    from factscore.retrieval import DocDB, Retrieval

    load_openai_api_key(os.path.join(
        args.kfs_dir, args.api_keys,
    ))

    # create af_generator
    kfs_af_generator = AtomicFactGenerator(
        key_path=os.path.join(args.kfs_dir, 'api.keys'),
        demon_dir=os.path.join(args.kfs_dir, args.data_dir, 'demos'),
        gpt3_cache_file=os.path.join(args.kfs_dir, args.cache_dir, '2_af_cache.pkl'),
        af_model_name='gpt-3.5-turbo-0125',
        demon_fn='k_demons_v1.json',
    ).run
    af_generator: Callable[
        [str], List[str] # Returns a list of atomic facts from generation
    ] = convert_kfs_af_generator_to_pnpxai_af_generator(kfs_af_generator)

    # create knowledge source
    db = Retrieval(
        db=DocDB(
            db_path=os.path.join(args.kfs_dir, args.data_dir, f'{KNOWLEDGE_SOURCE_NAME}.db'),
            data_path=os.path.join(args.kfs_dir, args.data_dir, f'{KNOWLEDGE_SOURCE_NAME}.jsonl'),
            tokenizer_path=None,
        ),
        cache_path=os.path.join(args.kfs_dir, args.cache_dir, f'{RETRIEVAL_CACHE_FILENAME}.json'),
        embed_cache_path=os.path.join(args.kfs_dir, args.cache_dir, f'{RETRIEVAL_CACHE_FILENAME}.pkl'),
        batch_size=256,
        ckp_path=None,
        retrieval_type='bm25',
    )
    knowledge_source: Callable[
        [str, str], # (a topic, "an" atomic fact)
        List[Dict[str, str]] # A list of knowledges (passages)
    ] = functools.partial(db.get_passages, k=args.retrieval_k)

    # create scorer
    scoring_model = functools.partial(llm, model=args.scoring_model_name)
    scorer: Callable[
        [str, str, List[Dict[str, str]]], # (a topic, an atomic fact, knowledges (passages))
        Any # Any type of score. In this case, bool.
    ] = functools.partial(kfs_scorer, scoring_model=scoring_model)

    # define score aggregation function
    aggregate_fn: Callable[
        [List[Any]], Any # Returns any type of scalar from the list of scores. In this case, float.
    ] = lambda scores: np.array(scores).mean()

    # create korfact scorer
    fs_kr = FactScore(
        atomic_fact_generator=af_generator,
        knowledge_source=knowledge_source,
        scorer=scorer,
        aggregate_fn=aggregate_fn, # if None, evaluate method returns None for aggregated score
    )

    # generate input
    generation = llm(
        prompt=PROMPT.format(topic=args.topic),
        model=args.model_name,
    )

    # evaluate
    output = fs_kr.evaluate(
        topic=args.topic,
        generation=generation,
    )

    # print output
    print('atomic_facts', dict(zip(output.atomic_facts, output.scores)))
    print('aggregated', output.aggregated_score)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)