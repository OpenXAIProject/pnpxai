# 명령어 모음
## build
cd pnp/pnpxai/visualizer/frontend && docker run --rm --name build-container -v "$(pwd):/project" my-react-app-build


inputGallery, experimentResult, input, explainer, metric를 state에 보관하기
1. 처음에는 global에서 state를 가져온다.(디폴트로 선택되어 있을 값을 global state에서 지정)
2. UI를 통해 값이 handle되면 dispatch를 통해 값을 설정한다.

colorMap 바뀌면 state에서 데이터 꺼내고 preprocess만 다시하기
1. useEffect colorMap을 설정한다.
2. 만약 cache에 experimentResult가 존재하는 경우, 레이아웃마다 colorScale만 변경하는 함수를 만든다.

Component 이름 refactoring

질문
ProjectId가 바뀔 때 experiment를 fetch하는 것이 어떤 의미?
RAM에 어디까지 저장해둘 것인지 고민(일단 다 저장해두자)

