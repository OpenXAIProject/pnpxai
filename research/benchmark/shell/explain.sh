#!/bin/bash

# 설정값 정의
SEED=42
# DATASETS=("Adult" "Bank Marketing" "Statlog (German Credit Data)" "Wine Quality")
DATASETS=("Wine Quality")
# MODELS=("tab_resnet" "xgb")
MODELS=("xgb" "tab_resnet")
# MODELS=("tab_resnet")
# FRAMEWORKS=("captum" "autoxai"  "pnpxai" "omnixai" "openxai")
# FRAMEWORKS=("captum")
FRAMEWORKS=("pnpxai")
# EXPLAINERS=("ig" "grad" "sg" "itg" "vg" "lrp" "lime" "shap")
# EXPLAINERS=("lime" "shap")

# 배치 크기 설정
BATCH_SIZE=32

# 로그 저장 디렉토리
LOG_DIR="logs"
mkdir -p $LOG_DIR

source ~/miniconda3/etc/profile.d/conda.sh

# 가능한 모든 조합에 대해 실행
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for framework in "${FRAMEWORKS[@]}"; do
            if [[ "$framework" == "autoxai" ]]; then
                conda activate autoxai
            else
                conda activate pnpenv
            fi

            if [[ "$framework" == "pnpxai" ]]; then

                # # ✅ 결과 파일 존재 여부 확인
                # RESULT_PATH="results/${dataset}/${model}/${framework}/${explainer}/explanations.npy"
                # if [[ -f "$RESULT_PATH" ]]; then
                #     echo "🚫 Skipping: Result already exists at $RESULT_PATH"
                #     continue
                # fi

                LOG_FILE="$LOG_DIR/explain.log"
                echo "🚀 Running: dataset=$dataset, model=$model, framework=$framework"
                echo "python script/explain.py --seed $SEED --dataset \"$dataset\" --model \"$model\" --framework \"$framework\" --batch_size $BATCH_SIZE >> \"$LOG_FILE\" 2>&1"

                python script/explain.py \
                    --seed $SEED \
                    --dataset "$dataset" \
                    --model "$model" \
                    --framework "$framework" \
                    --batch_size $BATCH_SIZE >> "$LOG_FILE" 2>&1

                # ✅ 성공/실패 메시지 출력
                if [[ $? -eq 0 ]]; then
                    echo "✅ Success: $dataset, $model, $framework, $explainer"
                else
                    echo "❌ Failed: $dataset, $model, $framework, $explainer"
                fi

                # ✅ 결과 파일이 저장되었는지 확인
                RESULT_PATH="results/${dataset}/${model}/${framework}/evaluation.csv"
                if [[ -f "$RESULT_PATH" ]]; then
                    echo "🎯 Result saved to: $RESULT_PATH"
                else
                    echo "⚠️ Result not found: $RESULT_PATH"
                fi

                continue
            fi

            for explainer in "${EXPLAINERS[@]}"; do
                
                # ✅ 잘못된 조합 필터링
                if [[ "$framework" == "omnixai" && "$model" != "xgb" ]]; then
                    # echo "❌ Skipping invalid combination: $framework with $model"
                    continue
                fi

                if [[ "$framework" == "omnixai" && ! "$explainer" =~ ^(lime|shap)$ ]]; then
                    # echo "❌ Skipping invalid combination: $framework with $explainer"
                    continue
                fi

                if [[ "$framework" == "autoxai" && ! "$explainer" =~ ^(lime|shap)$ ]]; then
                    # echo "❌ Skipping invalid combination: $framework with $explainer"
                    continue
                fi

                if [[ "$framework" == "openxai" && "$explainer" =~ ^(lrp|vg)$ ]]; then
                    # echo "❌ Skipping invalid combination: $framework with $explainer"
                    continue
                fi

                if [[ "$framework" == "captum" && "$explainer" =~ ^(vg)$ ]]; then
                    # echo "❌ Skipping invalid combination: $framework with $explainer"
                    continue
                fi

                if [[ "$framework" == "captum" && "$model" == "xgb" && ! "$explainer" =~ ^(lime|shap)$ ]]; then
                    # echo "❌ Skipping invalid combination: $framework with $explainer"
                    continue
                fi

                if [[ "$framework" == "openxai" && "$model" == "xgb" ]]; then
                    # echo "❌ Skipping invalid combination: $framework does not support $model"
                    continue
                fi

                # # ✅ 결과 파일 존재 여부 확인
                # RESULT_PATH="results/${dataset}/${model}/${framework}/${explainer}/explanations.npy"
                # if [[ -f "$RESULT_PATH" ]]; then
                #     echo "🚫 Skipping: Result already exists at $RESULT_PATH"
                #     continue
                # fi

                # ✅ 로그 파일 생성
                # LOG_FILE="$LOG_DIR/${dataset}_${model}_${framework}_${explainer}.log"
                LOG_FILE="$LOG_DIR/explain.log"
                echo "🚀 Running: dataset=$dataset, model=$model, framework=$framework, explainer=$explainer"
                
                echo "python script/explain.py --seed $SEED --dataset \"$dataset\" --model \"$model\" --framework \"$framework\" --explainer \"$explainer\" --batch_size $BATCH_SIZE >> \"$LOG_FILE\" 2>&1"

                python script/explain.py \
                    --seed $SEED \
                    --dataset "$dataset" \
                    --model "$model" \
                    --framework "$framework" \
                    --explainer "$explainer" \
                    --batch_size $BATCH_SIZE >> "$LOG_FILE" 2>&1

                # ✅ 성공/실패 메시지 출력
                if [[ $? -eq 0 ]]; then
                    echo "✅ Success: $dataset, $model, $framework, $explainer"
                else
                    echo "❌ Failed: $dataset, $model, $framework, $explainer"
                fi

                # ✅ 결과 파일이 저장되었는지 확인
                RESULT_PATH="results/${dataset}/${model}/${framework}/${explainer}/explanations.npy"
                if [[ -f "$RESULT_PATH" ]]; then
                    echo "🎯 Result saved to: $RESULT_PATH"
                else
                    echo "⚠️ Result not found: $RESULT_PATH"
                fi
            done
        done
    done
done

echo "🔥 All tasks completed."
