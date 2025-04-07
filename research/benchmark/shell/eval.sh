#!/bin/bash

# 설정값 정의
SEED=42
# DATASETS=("Adult" "Bank Marketing" "Statlog (German Credit Data)" "Wine Quality")
DATASETS=("Wine Quality")
MODELS=("tab_resnet" "xgb")
FRAMEWORKS=("pnpxai" "omnixai" "openxai")
EXPLAINERS=("lime" "shap" "ig" "grad" "sg" "itg" "vg" "lrp")
METRIC=("abpc" "cmpx" "cmpd")

# 로그 저장 디렉토리
LOG_DIR="logs"
mkdir -p $LOG_DIR

# 가능한 모든 조합에 대해 실행
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for framework in "${FRAMEWORKS[@]}"; do
            for explainer in "${EXPLAINERS[@]}"; do
                for metric in "${METRIC[@]}"; do
                    # ✅ 잘못된 조합 필터링
                    if [[ "$framework" == "omnixai" && "$model" != "xgb" ]]; then
                        # echo "❌ Skipping invalid combination: $framework with $model"
                        continue
                    fi

                    if [[ "$framework" == "omnixai" && ! "$explainer" =~ ^(lime|shap)$ ]]; then
                        # echo "❌ Skipping invalid combination: $framework with $explainer"
                        continue
                    fi

                    if [[ "$framework" == "openxai" && "$explainer" =~ ^(lrp|vg)$ ]]; then
                        # echo "❌ Skipping invalid combination: $framework with $explainer"
                        continue
                    fi

                    if [[ "$framework" == "openxai" && "$model" == "xgb" ]]; then
                        # echo "❌ Skipping invalid combination: $framework does not support $model"
                        continue
                    fi

                    if [[ "$framework" == "pnpxai" && "$model" == "xgb" && ! "$explainer" =~ ^(shap|lime)$ ]]; then
                        # echo "❌ Skipping invalid combination: $framework does not support $model with $explainer"
                        continue
                    fi

                    # ✅ 결과 파일 존재 여부 확인
                    RESULT_PATH="results/${dataset}/${model}/${framework}/${explainer}/${metric}.npy"
                    if [[ -f "$RESULT_PATH" ]]; then
                        # echo "🚫 Skipping: Result already exists at $RESULT_PATH"
                        continue
                    fi

                    # ✅ 로그 파일 생성
                    LOG_FILE="$LOG_DIR/evaluate.log"
                    echo "🚀 Running: dataset=$dataset, model=$model, framework=$framework, explainer=$explainer, metric=$metric"
                    
                    # ✅ Python 실행 명령어 (evaluate.py 호출)
                    python script/evaluate.py \
                        --seed $SEED \
                        --dataset "$dataset" \
                        --model "$model" \
                        --framework "$framework" \
                        --explainer "$explainer" \
                        --metric "$metric" >> "$LOG_FILE" 2>&1

                    # ✅ 성공/실패 메시지 출력
                    if [[ $? -eq 0 ]]; then
                        echo "✅ Success: $dataset, $model, $framework, $explainer"
                    else
                        echo "❌ Failed: $dataset, $model, $framework, $explainer"
                    fi

                    # ✅ 결과 파일이 저장되었는지 확인
                    RESULT_PATH="results/${dataset}/${model}/${framework}/${explainer}/${metric}.npy"
                    if [[ -f "$RESULT_PATH" ]]; then
                        echo "🎯 Result saved to: $RESULT_PATH"
                    else
                        echo "⚠️ Result not found: $RESULT_PATH"
                    fi
                done
            done
        done
    done
done

echo "🔥 All tasks completed."