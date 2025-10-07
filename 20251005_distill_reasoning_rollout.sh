#!/bin/bash

# 多模型多分类模式并行测试脚本
# 支持自定义模型列表和分类模式列表进行批量测试

set -e  # 遇到错误时退出

# =============================================================================
# 配置区域 - 支持预定义配置和自定义配置
# =============================================================================

# 加载配置文件
CONFIG_FILE="test_config.sh"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
else
    echo -e "${RED}警告: 配置文件 $CONFIG_FILE 不存在，使用默认配置${NC}"
fi

# 处理命令行参数
CONFIG_TYPE=${1:-"custom"}

# 通用配置（不随配置类型改变）
BATCH_SIZE=512                # 批处理大小
DEBUG_MODE=true             # 是否启用调试模式
SAVE_FULL_RESPONSE=true     # 是否保存完整响应
SAVE_JSON_LOGS=true         # 是否保存JSON日志
TOP_LOGPROBS=8             # logprobs数量

# Rollout 配置
TOTAL_ROLLOUTS=5  # 总共执行的rollout次数
START_ROLLOUT_ID=1  # 起始rollout ID

# 文件路径配置
method_name=MiroDiag-16K # MDD-5K, MiroDiag-16K
INPUT_FILE="./data/SMHC_${method_name}_train_data.xlsx"
SCRIPT_NAME="20251003_depression_anxiety_classifier_async_Qwen_four_tasks_openrouter.py"

# API配置
API_BASE="https://openrouter.ai/api/v1"
SITE_URL=""
SITE_NAME=""

# 如果提供了配置类型参数，使用预定义配置
if [ "$CONFIG_TYPE" != "custom" ] && declare -f select_test_config > /dev/null; then
    select_test_config "$CONFIG_TYPE"
else
    # 默认自定义配置 - 在这里修改你的测试配置
    MODEL_NAME_LIST=(
        # "qwen/qwen3-8b"
        # "qwen/qwen3-32b"
        # "openai/gpt-oss-120b"
        # "openai/o1"
        # "google/gemini-2.5-pro"
        # "x-ai/grok-4"
        # "deepseek/deepseek-r1-0528"
        # "deepseek/deepseek-v3.2-exp"
        # "deepseek/deepseek-r1-distill-qwen-32b"
        # "deepseek/deepseek-r1-0528-qwen3-8b"
        "moonshotai/kimi-k2-0905"
        # 添加更多模型...
    )

    # 分类模式列表 - 添加你想测试的分类模式
    CLASSIFICATION_MODE_LIST=(
        "recommendation"
        # "icd10"
        # "binary"
        # "multiclass"
        # "depression_symptom"
        # "anxiety_symptom"
        # "multilabel"
        # 添加更多分类模式...
    )

    # 基础配置
    LIMIT=100000                   # 每个测试的样本数量
    MAX_CONCURRENT=1           # 最大并发数（免费模型建议2，付费模型可以更高）
    PARALLEL_JOBS=7             # 同时运行的测试任务数量
fi

# =============================================================================
# 脚本逻辑区域 - 通常不需要修改
# =============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 生成全局时间戳
GLOBAL_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 执行任务的函数
run_task() {
    local task_id=$1
    local task_name=$2
    local cmd=$3
    
    echo -e "${CYAN}[$(date '+%H:%M:%S')] 启动任务 $((task_id+1))/${#TASK_LIST[@]}: $task_name${NC}"
    
    # 执行命令（不重定向输出，因为Python脚本会自己生成日志）
    if eval "$cmd" >/dev/null 2>&1; then
        echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓ 任务完成: $task_name${NC}"
        return 0
    else
        echo -e "${RED}[$(date '+%H:%M:%S')] ✗ 任务失败: $task_name${NC}"
        echo -e "${RED}    任务执行失败，请检查Python脚本生成的日志文件${NC}"
        return 1
    fi
}

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}    多模型多分类模式并行测试脚本${NC}"
echo -e "${BLUE}=========================================${NC}"
echo -e "${CYAN}当前配置: $CONFIG_TYPE${NC}"
echo -e "${CYAN}测试配置:${NC}"
echo -e "  模型数量: ${#MODEL_NAME_LIST[@]}"
echo -e "  分类模式数量: ${#CLASSIFICATION_MODE_LIST[@]}"
echo -e "  总测试组合: $((${#MODEL_NAME_LIST[@]} * ${#CLASSIFICATION_MODE_LIST[@]}))"
echo -e "  每组样本数: $LIMIT"
echo -e "  并行任务数: $PARALLEL_JOBS"
echo -e "  总Rollout次数: $TOTAL_ROLLOUTS"
echo -e "  全局时间戳: $GLOBAL_TIMESTAMP"
echo -e "${BLUE}=========================================${NC}"

# 显示使用说明
if [ "$CONFIG_TYPE" = "help" ] || [ "$CONFIG_TYPE" = "--help" ] || [ "$CONFIG_TYPE" = "-h" ]; then
    if declare -f show_usage > /dev/null; then
        show_usage
    else
        echo "使用方法:"
        echo "  $0 [配置类型]"
        echo ""
        echo "配置类型:"
        echo "  custom (默认)  - 使用脚本中的自定义配置"
        echo "  quick_test     - 快速测试配置"
        echo "  standard_test  - 标准测试配置"
        echo "  full_test      - 完整测试配置"
    fi
    exit 0
fi

# 检查前置条件
echo -e "${YELLOW}检查前置条件...${NC}"

# 检查输入文件
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}错误: 输入文件不存在: $INPUT_FILE${NC}"
    exit 1
fi

# 检查Python脚本
if [ ! -f "$SCRIPT_NAME" ]; then
    echo -e "${RED}错误: Python脚本不存在: $SCRIPT_NAME${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 前置条件检查通过${NC}"

# 显示测试计划
echo -e "\n${CYAN}测试计划:${NC}"
echo -e "${PURPLE}模型列表:${NC}"
for i in "${!MODEL_NAME_LIST[@]}"; do
    echo -e "  $((i+1)). ${MODEL_NAME_LIST[i]}"
done

echo -e "\n${PURPLE}分类模式列表:${NC}"
for i in "${!CLASSIFICATION_MODE_LIST[@]}"; do
    echo -e "  $((i+1)). ${CLASSIFICATION_MODE_LIST[i]}"
done

echo -e "\n${YELLOW}按Enter继续，Ctrl+C取消...${NC}"
read -r

# 开始Rollout循环
echo -e "\n${BLUE}=========================================${NC}"
echo -e "${BLUE}    开始执行 $TOTAL_ROLLOUTS 次 Rollout${NC}"
echo -e "${BLUE}=========================================${NC}"

for rollout_id in $(seq $START_ROLLOUT_ID $((START_ROLLOUT_ID + TOTAL_ROLLOUTS - 1))); do
    echo -e "\n${PURPLE}=========================================${NC}"
    echo -e "${PURPLE}    执行 Rollout $rollout_id/$((START_ROLLOUT_ID + TOTAL_ROLLOUTS - 1))${NC}"
    echo -e "${PURPLE}=========================================${NC}"
    
    # 为当前rollout设置目录
    OUTPUT_DIR="./distill/${method_name}/rollout-${rollout_id}/results"
    LOGS_DIR="./distill/${method_name}/rollout-${rollout_id}/logs"
    RESULTS_DIR="./distill/${method_name}/rollout-${rollout_id}/results"
    
    # 创建必要目录
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$LOGS_DIR"
    mkdir -p "$RESULTS_DIR"
    
    # 生成当前rollout的时间戳
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    
    echo -e "${CYAN}Rollout $rollout_id 开始时间: $(date)${NC}"
    echo -e "${CYAN}输出目录: $OUTPUT_DIR${NC}"
    
    # 创建任务列表
    declare -a TASK_LIST=()
    declare -a TASK_NAMES=()

    for model in "${MODEL_NAME_LIST[@]}"; do
        for mode in "${CLASSIFICATION_MODE_LIST[@]}"; do
            # 生成安全的文件名（替换特殊字符）
            safe_model=$(echo "$model" | sed 's/[^a-zA-Z0-9._-]/_/g')
            safe_mode=$(echo "$mode" | sed 's/[^a-zA-Z0-9._-]/_/g')
            
            task_name="${safe_model}_${safe_mode}"
            output_file="${OUTPUT_DIR}/${task_name}_${TIMESTAMP}.xlsx"
            
            # 构建命令
            cmd="python \"$SCRIPT_NAME\" \
                --api \"$API_BASE\" \
                --input \"$INPUT_FILE\" \
                --output \"${task_name}_${TIMESTAMP}.xlsx\" \
                --logs_dir \"$LOGS_DIR\" \
                --results_dir \"$RESULTS_DIR\" \
                --model \"$model\" \
                --classification_mode \"$mode\" \
                --limit $LIMIT \
                --max_concurrent $MAX_CONCURRENT \
                --batch_size $BATCH_SIZE"
            
            # 添加可选参数
            if [ "$DEBUG_MODE" = true ]; then
                cmd="$cmd --debug"
            fi
            
            if [ "$SAVE_FULL_RESPONSE" != true ]; then
                cmd="$cmd --save_full_response"
            fi
            
            if [ "$SAVE_JSON_LOGS" = true ]; then
                cmd="$cmd --save_json_logs"
            fi
            
            cmd="$cmd --evaluate \
                --top_logprobs $TOP_LOGPROBS \
                --site_url \"$SITE_URL\" \
                --site_name \"$SITE_NAME\""
            
            TASK_LIST+=("$cmd")
            TASK_NAMES+=("$task_name")
        done
    done

    echo -e "\n${BLUE}开始并行执行 ${#TASK_LIST[@]} 个测试任务...${NC}"
    echo -e "开始时间: $(date)"

    # 创建任务状态跟踪
    declare -a TASK_PIDS=()
    declare -a TASK_STATUS=()
    declare -a TASK_START_TIME=()

    # 并行执行任务
    for i in "${!TASK_LIST[@]}"; do
        # 等待空闲槽位
        while [ ${#TASK_PIDS[@]} -ge $PARALLEL_JOBS ]; do
            # 检查已完成的任务
            for j in "${!TASK_PIDS[@]}"; do
                if ! kill -0 "${TASK_PIDS[j]}" 2>/dev/null; then
                    # 任务已完成，移除PID
                    unset TASK_PIDS[j]
                    TASK_PIDS=("${TASK_PIDS[@]}")  # 重新索引数组
                    break
                fi
            done
            sleep 1
        done
        
        # 启动新任务
        run_task "$i" "${TASK_NAMES[i]}" "${TASK_LIST[i]}" &
        TASK_PIDS+=($!)
        TASK_START_TIME[i]=$(date +%s)
        
        # 短暂延迟避免同时启动过多任务
        sleep 2
    done

    # 等待所有任务完成
    echo -e "\n${YELLOW}等待所有任务完成...${NC}"
    for pid in "${TASK_PIDS[@]}"; do
        wait "$pid"
    done
    
    echo -e "\n${GREEN}✓ Rollout $rollout_id 完成！${NC}"
    echo -e "${CYAN}Rollout $rollout_id 结束时间: $(date)${NC}"
done

echo -e "\n${BLUE}=========================================${NC}"
echo -e "${GREEN}所有 $TOTAL_ROLLOUTS 次 Rollout 完成！${NC}"
echo -e "${CYAN}总体开始时间: $GLOBAL_TIMESTAMP${NC}"
echo -e "${CYAN}总体结束时间: $(date)${NC}"
echo -e "${CYAN}数据集: $method_name${NC}"
echo -e "${CYAN}结果保存在: ./distill/${method_name}/rollout-*/results/${NC}"
echo -e "${BLUE}=========================================${NC}"

echo -e "\n${GREEN}多轮Rollout脚本执行完成！${NC}"
