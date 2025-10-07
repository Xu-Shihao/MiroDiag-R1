import pandas as pd
import requests
import json
import numpy as np
import math
from tqdm import tqdm
import os
import argparse
import logging
import time
# 设置matplotlib为非交互式后端，防止在无GUI环境下出问题
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re  # 添加正则表达式库
import traceback  # 添加traceback模块用于详细错误信息
# 保留异步处理相关导入
import asyncio
from asyncio import Semaphore
# 添加评估指标相关导入
from sklearn.metrics import f1_score, balanced_accuracy_score, average_precision_score
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
from sklearn.preprocessing import LabelBinarizer
# 添加tokenizer相关导入
from transformers.models.auto.tokenization_auto import AutoTokenizer
# 添加OpenAI客户端导入
from openai import AsyncOpenAI
# 添加环境变量读取
from dotenv import load_dotenv
import os

# # 取消设置代理相关的环境变量
# proxy_vars = [
#     'http_proxy',
#     'https_proxy', 
#     'HTTP_PROXY',
#     'HTTPS_PROXY',
#     'all_proxy'
# ]

# for var in proxy_vars:
#     # 使用pop方法安全地删除环境变量，如果不存在也不会报错
#     removed_value = os.environ.pop(var, None)
#     if removed_value is not None:
#         print(f"已取消设置环境变量: {var} = {removed_value}")
#     else:
#         print(f"环境变量 {var} 不存在，跳过")

# print("所有代理环境变量已取消设置完成")

# 确保日志和结果目录存在
os.makedirs("./logs", exist_ok=True)
os.makedirs("./results", exist_ok=True)

# 获取当前日期作为前缀
current_date = datetime.now().strftime("%Y%m%d")
current_time = datetime.now().strftime("%H%M%S")

# 初始化logger，在main函数中重新配置
logger = logging.getLogger(__name__)

# JSON日志记录相关变量
json_log_data = []
json_log_lock = asyncio.Lock()

max_tokens = 32000
# Qwen3 best practice
temperature = 1
top_p = 0.95


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="基于LLM模型通过OpenRouter分析文本中的抑郁症和焦虑症倾向")
    parser.add_argument("--api", type=str, default="https://openrouter.ai/api/v1",  # http://localhost:9019/v1/
                        help="OpenRouter API base URL or local API base URL")
    parser.add_argument("--input", type=str, default="./data/SMHC_MDD-5K_validation_data.xlsx", 
                        help="输入Excel文件路径")
    parser.add_argument("--output", type=str, default="test.xlsx", 
                        help="输出Excel文件名（将保存在results_dir目录下）")
    parser.add_argument("--logs_dir", type=str, default="./logs", 
                        help="日志文件保存目录")
    parser.add_argument("--results_dir", type=str, default="./results", 
                        help="结果文件保存目录")
    parser.add_argument("--model", type=str, default="qwen/qwen3-8b", 
                        help="模型名称，OpenRouter格式（如openai/gpt-5）或本地vLLM模型路径")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="批处理大小")
    parser.add_argument("--debug", action="store_true", default=True, 
                        help="开启调试模式，显示更详细的信息")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="可视化结果并保存图表")
    parser.add_argument("--limit", type=int, default=10, 
                        help="限制处理的记录数量，用于快速测试")
    parser.add_argument("--save_full_response", action="store_true", default=True,
                        help="保存模型的完整响应内容")
    parser.add_argument("--max_concurrent", type=int, default=16, 
                        help="最大并发请求数")
    # 添加分类模式和评估相关参数
    parser.add_argument("--classification_mode", type=str, choices=["binary", "multiclass", "depression_symptom", "anxiety_symptom", "multilabel", "icd10", "recommendation"], default="recommendation",
                        help="分类模式：binary(二分类：抑郁vs焦虑) 或 multiclass(四分类：抑郁/焦虑/mix/others) 或 depression_symptom(抑郁症状检测) 或 anxiety_symptom(焦虑症状检测) 或 multilabel(多标签：同时检测抑郁和焦虑) 或 icd10(输出ICD-10诊断代码) 或 recommendation(推荐多个可能的ICD-10大类代码)")
    parser.add_argument("--evaluate", action="store_true", default=True,
                        help="如果数据中包含OverallDiagnosis列，则计算模型评估指标")
    parser.add_argument("--min-samples-per-class", type=int, default=5,
                        help="每个类别的最小样本数（用于平衡采样）")
    parser.add_argument("--evaluate_only", type=str, default=None,
                        help="仅对指定的结果文件进行评估，不运行模型推理")

    parser.add_argument("--site_url", type=str, default="",
                        help="站点URL，用于OpenRouter排名显示")
    parser.add_argument("--site_name", type=str, default="",
                        help="站点名称，用于OpenRouter排名显示")
    parser.add_argument("--top_logprobs", type=int, default=8,
                        help="返回每个token位置最可能的token数量（0-20）")
    parser.add_argument("--save_json_logs", action="store_true", default=True,
                        help="保存请求和响应的详细JSON日志到results文件夹")
    return parser.parse_args()

# 全局tokenizer变量
tokenizer = None

def load_tokenizer(model_path):
    """加载模型对应的tokenizer"""
    global tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info(f"成功加载tokenizer: {model_path}")
        return True
    except Exception as e:
        logger.error(f"加载tokenizer失败: {e}")
        return False

async def log_request_response(visit_number, messages, response, model_name, classification_mode, ground_truth=None, save_json_logs=True):
    """记录请求和响应到JSON日志"""
    if not save_json_logs:
        return
    
    global json_log_data, json_log_lock
    
    # 去除response中的logprobs信息
    if response and 'choices' in response and response['choices']:
        response['choices'][0].pop('logprobs', None)
    
    # 创建日志条目
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "visit_number": visit_number,
        "model_name": model_name,
        "classification_mode": classification_mode,
        "ground_truth": ground_truth,
        "request": {
            "messages": messages,
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "logprobs": True
        },
        "response": {
            "raw_response": response,
            "content": response['choices'][0].get('text', '') if response and 'choices' in response and response['choices'] else None,
            "reasoning_content": None,
        }
    }
    
    # 提取reasoning内容（如果有）
    if response and 'choices' in response and response['choices']:
        choice = response['choices'][0]
        if hasattr(choice, 'message') and choice.message and hasattr(choice.message, 'model_extra') and choice.message.model_extra:
            if "reasoning_content" in choice.message.model_extra:
                log_entry["response"]["reasoning_content"] = choice.message.model_extra["reasoning_content"]
            elif "reasoning" in choice.message.model_extra:
                log_entry["response"]["reasoning_content"] = choice.message.model_extra["reasoning"]
    
    # 线程安全地添加到日志列表
    async with json_log_lock:
        json_log_data.append(log_entry)

def save_json_logs_to_file(output_prefix, model_name, classification_mode, results_dir="./results"):
    """保存JSON日志到文件"""
    global json_log_data
    
    if not json_log_data:
        logger.info("没有JSON日志数据需要保存")
        return
    
    # 确保results目录存在
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成文件名
    model_basename = os.path.basename(model_name.rstrip('/'))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_prefix}_{model_basename}_{classification_mode}_{timestamp}_logs.json"
    filepath = os.path.join(results_dir, filename)
    
    # 创建完整的日志数据结构
    log_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "model_name": model_name,
            "classification_mode": classification_mode,
            "total_requests": len(json_log_data),
            "output_prefix": output_prefix
        },
        "logs": json_log_data
    }
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON日志已保存至: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"保存JSON日志失败: {e}")
        return None

def save_batch_json_logs_to_file(output_prefix, model_name, classification_mode, batch_logs, results_dir="./results"):
    """保存指定批次的JSON日志到文件"""
    if not batch_logs:
        logger.info("没有批次JSON日志数据需要保存")
        return
    
    # 确保results目录存在
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成文件名
    model_basename = os.path.basename(model_name.rstrip('/'))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_prefix}_{model_basename}_{classification_mode}_{timestamp}_logs.json"
    filepath = os.path.join(results_dir, filename)
    
    # 创建完整的日志数据结构
    log_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "model_name": model_name,
            "classification_mode": classification_mode,
            "total_requests": len(batch_logs),
            "output_prefix": output_prefix
        },
        "logs": batch_logs
    }
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        logger.info(f"批次JSON日志已保存至: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"保存批次JSON日志失败: {e}")
        return None

def find_existing_json_logs(output_prefix, model_name, classification_mode, results_dir="./results"):
    """查找已存在的JSON日志文件"""
    try:
        import glob
        
        model_basename = os.path.basename(model_name.rstrip('/'))
        pattern = f"{results_dir}/*_{model_basename}_{classification_mode}_*_logs.json"
        
        # 查找匹配的文件
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            return None, []
        
        # 按修改时间排序，取最新的
        matching_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_file = matching_files[0]
        
        # 加载JSON数据
        with open(latest_file, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
        
        # 提取logs部分
        if isinstance(file_data, dict) and 'logs' in file_data:
            previous_data = file_data['logs']
        else:
            # 兼容旧格式
            previous_data = file_data if isinstance(file_data, list) else []
        
        logger.info(f"找到已存在的JSON日志文件: {latest_file}")
        logger.info(f"包含 {len(previous_data)} 条记录")
        
        return latest_file, previous_data
        
    except Exception as e:
        logger.error(f"查找或加载已存在的JSON日志失败: {e}")
        return None, []

def find_and_merge_batch_json_logs(output_prefix, model_name, classification_mode, results_dir="./results"):
    """查找并合并所有批次的JSON日志文件"""
    try:
        import glob
        from datetime import datetime
        
        model_basename = model_name.split("/")[-1]
        
        # 查找批次文件模式：包含 _batch_ 的文件
        # 支持两种模式：新的批次文件格式和可能的其他格式
        batch_pattern1 = f"{results_dir}/*_batch_*_{model_basename}_{classification_mode}_*_logs.json"
        batch_pattern2 = f"{results_dir}/*_{model_basename}_{classification_mode}_*_batch_*_logs.json"
        batch_files = glob.glob(batch_pattern1) + glob.glob(batch_pattern2)
        # 去重
        batch_files = list(set(batch_files))
        
        # 查找完整文件模式：包含 _complete_ 的文件
        complete_pattern = f"{results_dir}/*_complete_{model_basename}_{classification_mode}_*_logs.json"
        complete_files = glob.glob(complete_pattern)
        
        # 查找普通文件模式：不包含 _batch_ 和 _complete_ 的文件
        all_pattern = f"{results_dir}/*_{model_basename}_{classification_mode}_*_logs.json"
        all_files = glob.glob(all_pattern)
        regular_files = [f for f in all_files if "_batch_" not in f and "_complete_" not in f]
        
        all_found_files = batch_files + complete_files + regular_files
        
        # 输出查询的pattern
        logger.info(f"查询的pattern: {batch_pattern1}, {batch_pattern2}, {complete_pattern}, {all_pattern}")
        
        if not all_found_files:
            logger.info("未找到任何已存在的JSON日志文件")
            return None, []
        
        logger.info(f"找到 {len(batch_files)} 个批次文件, {len(complete_files)} 个完整文件, {len(regular_files)} 个普通文件")
        
        if batch_files:
            logger.info("批次文件列表:")
            for bf in batch_files:
                logger.info(f"  - {os.path.basename(bf)}")
        
        if complete_files:
            logger.info("完整文件列表:")
            for cf in complete_files:
                logger.info(f"  - {os.path.basename(cf)}")
        
        if regular_files:
            logger.info("普通文件列表:")
            for rf in regular_files:
                logger.info(f"  - {os.path.basename(rf)}")
        
        # 优先使用批次文件，如果没有批次文件则使用完整文件或普通文件
        if batch_files:
            # 按文件名中的批次号排序
            def extract_batch_number(filename):
                import re
                # 尝试多种批次号模式
                patterns = [
                    r'_batch_(\d+)_',  # _batch_001_
                    r'batch_(\d+)',    # batch_001
                    r'_(\d+)_batch',   # _001_batch
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, filename)
                    if match:
                        return int(match.group(1))
                
                # 如果没有找到批次号，尝试从文件修改时间排序
                return 0
            
            batch_files.sort(key=extract_batch_number)
            
            merged_data = []
            processed_visit_numbers = set()
            
            logger.info(f"开始合并 {len(batch_files)} 个批次文件...")
            
            for batch_file in batch_files:
                try:
                    with open(batch_file, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                    
                    # 提取logs部分
                    if isinstance(file_data, dict) and 'logs' in file_data:
                        batch_logs = file_data['logs']
                    else:
                        batch_logs = file_data if isinstance(file_data, list) else []
                    
                    # 去重：只添加未处理过的visit_number
                    new_logs = []
                    for log_entry in batch_logs:
                        visit_num = log_entry.get('visit_number')
                        if visit_num not in processed_visit_numbers:
                            new_logs.append(log_entry)
                            processed_visit_numbers.add(visit_num)
                    
                    merged_data.extend(new_logs)
                    logger.info(f"批次文件 {os.path.basename(batch_file)}: 添加了 {len(new_logs)} 条新记录")
                    
                except Exception as e:
                    logger.warning(f"读取批次文件 {batch_file} 失败: {e}")
                    continue
            
            if merged_data:
                # 按时间戳排序
                merged_data.sort(key=lambda x: x.get('timestamp', ''))
                
                logger.info(f"成功合并批次文件，总共 {len(merged_data)} 条记录")
                return "merged_batch_files", merged_data
            else:
                logger.warning("批次文件合并后没有有效数据")
        
        # 如果没有批次文件或批次文件合并失败，使用最新的完整文件或普通文件
        fallback_files = complete_files + regular_files
        if fallback_files:
            fallback_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_file = fallback_files[0]
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
            
            if isinstance(file_data, dict) and 'logs' in file_data:
                previous_data = file_data['logs']
            else:
                previous_data = file_data if isinstance(file_data, list) else []
            
            logger.info(f"使用备用文件: {latest_file}")
            logger.info(f"包含 {len(previous_data)} 条记录")
            
            return latest_file, previous_data
        
        return None, []
        
    except Exception as e:
        logger.error(f"查找或合并批次JSON日志失败: {e}")
        return None, []

def extract_processed_visit_numbers(json_data):
    """从JSON数据中提取已处理的VisitNumber列表"""
    processed_visits = set()
    
    for entry in json_data:
        if 'visit_number' in entry:
            processed_visits.add(entry['visit_number'])
    
    return processed_visits

def load_previous_results_from_json(json_data, classification_mode):
    """从JSON数据中重建结果DataFrame"""
    results_list = []
    
    for entry in json_data:
        if 'visit_number' not in entry:
            continue
            
        visit_number = entry['visit_number']
        ground_truth = entry.get('ground_truth')
        
        # 从响应中提取结果
        response = entry.get('response', {}).get('raw_response')
        if not response:
            continue
        
        # 根据分类模式提取相应的结果
        try:
            if classification_mode == "recommendation":
                # 提取推荐代码
                recommended_codes = extract_classification_probs(response, classification_mode, debug=False, ground_truth=ground_truth)
                
                if recommended_codes and isinstance(recommended_codes, list):
                    top1_code = recommended_codes[0] if len(recommended_codes) > 0 else None
                    top3_codes = recommended_codes[:3] if len(recommended_codes) >= 3 else recommended_codes
                    num_recommended = len(recommended_codes)
                else:
                    top1_code = None
                    top3_codes = []
                    num_recommended = 0
                    recommended_codes = []
                
                result = {
                    "VisitNumber": visit_number,
                    "DiagnosisCode": ground_truth if not isinstance(ground_truth, list) else ground_truth,  # 添加DiagnosisCode列用于评估
                    "Ground_Truth_ICD10": ground_truth if not isinstance(ground_truth, list) else ground_truth,
                    "Recommended_ICD10_Codes": str(recommended_codes),
                    "Top1_Code": top1_code,
                    "Top3_Codes": str(top3_codes),
                    "Num_Recommended": num_recommended,
                    "Model_Response": entry.get('response', {}).get('content', '')
                }
                
            elif classification_mode == "icd10":
                # 提取ICD10代码
                predicted_icd10 = extract_icd10_code(response, debug=False)
                
                # 提取大类和小类
                predicted_major_class = None
                predicted_minor_class = None
                
                if predicted_icd10:
                    import re
                    major_match = re.match(r'(F\d+|Z71)', predicted_icd10)
                    if major_match:
                        predicted_major_class = major_match.group(1)
                    
                    minor_match = re.match(r'(F\d+\.\d+|Z71\.\d+)', predicted_icd10)
                    if minor_match:
                        predicted_minor_class = minor_match.group(1)
                
                result = {
                    "VisitNumber": visit_number,
                    "DiagnosisCode": ground_truth,  # 添加DiagnosisCode列用于评估
                    "Ground_Truth_ICD10": ground_truth,
                    "Predicted_ICD10": predicted_icd10,
                    "Predicted_Major_Class": predicted_major_class,
                    "Predicted_Minor_Class": predicted_minor_class,
                    "Model_Response": entry.get('response', {}).get('content', '')
                }
                
            else:
                # 其他分类模式的处理
                continue
                
            results_list.append(result)
            
        except Exception as e:
            logger.warning(f"处理VisitNumber {visit_number}的历史数据时出错: {e}")
            continue
    
    if results_list:
        results_df = pd.DataFrame(results_list)
        logger.info(f"从历史数据中重建了 {len(results_df)} 条结果记录")
        return results_df
    else:
        return pd.DataFrame()

# 创建prompt模板
def create_prompt(text, classification_mode="binary"):
    if classification_mode == "binary":
        system_prompt = rf"""你是一位经验丰富的精神科医生。请阅读以下病人来精神科问诊的对话记录，并判断病人更可能患有抑郁症还是焦虑症。

请用中文一步一步思考，之后将"抑郁"或者"焦虑"的结果放在<box>xxx</box>中输出。"""
        user_prompt = rf"""[问诊对话开始]
{text}
[问诊对话结束]"""

    elif classification_mode == "multiclass":
        system_prompt = rf"""你是一位经验丰富的精神科医生。请阅读以下病人来精神科问诊的对话记录，并判断病人的主要心理健康状况。抑郁症以**持续的情绪低落、兴趣减退和精力缺乏**为主，而焦虑症则以**过度担忧、紧张不安和对未来事件的恐惧**为主要特点。

请从以下四个选项中选择最合适的诊断：
- 抑郁：主要表现为抑郁症状, 满足ICD诊断要求。 
- 焦虑：主要表现为焦虑症状, 满足ICD诊断要求。 
- mix：同时表现出明显的抑郁和焦虑症状均满足ICD诊断要求，或者都没有满足单诊断为抑郁和焦虑的程度。
- others：其他心理健康问题（比如双向情感障碍，精神分裂症，等等）或正常状态。

请用中文一步一步思考，之后将结果（"抑郁"、"焦虑"、"mix"或"others"）放在<box>xxx</box>中输出。"""
        user_prompt = rf"""[问诊对话开始]
{text}
[问诊对话结束]"""

    elif classification_mode == "depression_symptom":
        system_prompt = rf"""你是一位经验丰富的精神科医生。请阅读以下精神科问诊的对话记录，并判断对话的人是否主要表现为抑郁症, 满足ICD诊断要求。

请用中文一步一步思考，之后判断问诊的人是否是抑郁症，将"是"或"否"放在<box>xxx</box>中输出。"""
        user_prompt = rf"""[问诊对话开始]
{text}
[问诊对话结束]"""


    elif classification_mode == "anxiety_symptom":
        system_prompt = rf"""你是一位经验丰富的精神科医生。请阅读以下精神科问诊的对话记录，并判断对话的人是否主要表现为焦虑症, 满足ICD诊断要求。

请用中文一步一步思考，之后判断病人是否有焦虑症，将"是"或"否"放在<box>xxx</box>中输出。"""
        user_prompt = rf"""[问诊对话开始]
{text}
[问诊对话结束]"""

    elif classification_mode == "multilabel":
        system_prompt = rf"""你是一位经验丰富的精神科医生。请阅读以下精神科问诊的对话记录，并根据ICD-10标准单独判断对话的人是否患有抑郁症和焦虑症。
        
请分别判断是否根据ICD-10标准，对话的人是否患有抑郁症和焦虑症，格式如下：
抑郁症：是/否
焦虑症：是/否

请用中文一步一步思考，之后将结果放在<box>xxx</box>中输出。"""
        user_prompt = rf"""[问诊对话开始]
{text}
[问诊对话结束]"""

    elif classification_mode == "icd10":
        system_prompt = rf"""你是一位经验丰富的精神科医生。请阅读以下精神科问诊的对话记录，并根据ICD-10国际疾病分类标准，给出最符合患者状况的具体诊断代码。

请仅从以下10种疾病中选择最符合的诊断：
    - F32 抑郁发作：情绪持续低落、兴趣/愉快感下降、精力不足；伴睡眠/食欲改变、自责/无价值感等；可轻/中/重度（重度可伴精神病性症状）；无既往躁狂/轻躁狂。
    - F41 其他焦虑障碍：恐慌发作或广泛性焦虑为主；过度担忧、紧张、心悸、胸闷、出汗、眩晕、濒死感/失控感；与特定情境无关或不成比例，造成显著痛苦/功能损害。
    - F39 未特指的心境（情感）障碍：存在心境障碍证据，但资料不足以明确归入抑郁或双相等具体亚型时选用。
    - F51 非器质性睡眠障碍：失眠、过度嗜睡、梦魇、昼夜节律紊乱等；非器质性原因；睡眠问题为主要主诉并致显著困扰/功能损害。
    - F98 其他儿童和青少年行为与情绪障碍：多见于儿童期起病（如遗尿/遗粪、口吃、抽动相关习惯性问题等），以发育期特异表现为主。
    - F42 强迫障碍：反复的强迫观念/行为，个体自知过度或不合理但难以抵抗，耗时或致显著困扰/损害。
    - F31 双相情感障碍：既往或目前存在躁狂/轻躁狂发作与抑郁发作的交替或混合；需有明确躁狂谱系证据。
    - F43 对严重应激反应和适应障碍：与明确应激事件有关；可为急性应激反应、PTSD或适应障碍；核心包含再体验、回避、警觉性增高或与应激源相关的情绪/行为改变。
    - F45 躯体形式障碍：反复或多样躯体症状为主（如疼痛、心悸、胃肠不适等），检查难以找到足以解释的器质性原因或与病因不相称，显著痛苦/就诊反复。
    - F20 精神分裂症：在知觉、思维、情感及行为等方面的广泛障碍；常见持续性妄想、幻听、思维松弛/破裂、情感淡漠、阴性症状，病程≥1月（或依本地标准）。

请用中文一步一步思考，之后直接给出最符合的ICD-10诊断代码，格式为"Fxx.x"，放在<box>xxx</box>中输出。"""
        user_prompt = rf"""[问诊对话开始]
{text}
[问诊对话结束]"""

    elif classification_mode == "recommendation":
        system_prompt = rf"""你是一位经验丰富的精神科医生。请阅读以下初次精神科门诊的问诊对话记录，并根据ICD-10国际疾病分类标准，仔细分析后输出患者诊断结束后的ICD-10诊断代码。

## 疾病分类说明
请仅从以下ICD-10标准中的10种疾病中选择最符合的诊断大类以及进一步细分的小类：
    - F32 抑郁发作：情绪持续低落、兴趣/愉快感下降、精力不足；伴睡眠/食欲改变、自责/无价值感等；可轻/中/重度（重度可伴精神病性症状）；无既往躁狂/轻躁狂。
        F32.0 轻度抑郁发作：症状轻，社会功能影响有限。
        F32.1 中度抑郁发作：症状更明显，日常活动受限。
        F32.2 重度抑郁发作，无精神病性症状：症状显著，丧失功能，但无妄想/幻觉。
        F32.3 重度抑郁发作，有精神病性症状：伴有抑郁性妄想、幻觉或木僵。
        F32.8 其他抑郁发作；F32.9 抑郁发作，未特指。
    - F41 其他焦虑障碍：恐慌发作或广泛性焦虑为主；过度担忧、紧张、心悸、胸闷、出汗、眩晕、濒死感/失控感；与特定情境无关或不成比例，造成显著痛苦/功能损害。
        F41.0 惊恐障碍：突发的强烈恐慌发作，常伴濒死感。
        F41.1 广泛性焦虑障碍：长期持续的过度担忧和紧张不安。
        F41.2 混合性焦虑与抑郁障碍：焦虑与抑郁并存但均不足以单独诊断。
        F41.3 其他混合性焦虑障碍：混合焦虑表现但未完全符合特定标准。
        F41.9 焦虑障碍，未特指：存在焦虑症状但资料不足以分类。
    - F39.x00 未特指的心境（情感）障碍：存在心境障碍证据，但资料不足以明确归入抑郁或双相等具体亚型时选用。
    - F51 非器质性睡眠障碍：失眠、过度嗜睡、梦魇、昼夜节律紊乱等；非器质性原因；睡眠问题为主要主诉并致显著困扰/功能损害。
        F51.0 非器质性失眠：入睡困难、易醒或睡眠不恢复精力。
        F51.1 非器质性嗜睡：过度睡眠或难以保持清醒。
        F51.2 非器质性睡眠-觉醒节律障碍：昼夜节律紊乱导致睡眠异常。
        F51.3 梦魇障碍：频繁恶梦导致醒后强烈不安。
        F51.4 睡眠惊恐（夜惊）：夜间突然惊恐醒来伴强烈焦虑反应。
        F51.5 梦游症：睡眠中出现起床或行走等复杂行为。
        F51.9 非器质性睡眠障碍，未特指：睡眠异常但无具体分类。
    - F98 其他儿童和青少年行为与情绪障碍：多见于儿童期起病（如遗尿/遗粪、口吃、抽动相关习惯性问题等），以发育期特异表现为主。
        F98.0 非器质性遗尿症：儿童在不适当年龄仍有排尿失控。
        F98.1 非器质性遗粪症：儿童在不适当情境排便。
        F98.2 婴儿期或儿童期进食障碍：儿童进食行为异常影响营养或发育。
        F98.3 异食癖：持续摄入非食物性物质。
        F98.4 刻板性运动障碍：重复、无目的的运动习惯。
        F98.5 口吃：言语流利性障碍，表现为言语阻塞或重复。
        F98.6 习惯性动作障碍：如咬甲、吮指等持续存在的习惯。
        F98.8 其他特指的儿童行为和情绪障碍：符合儿童期特异但不归入上述类。
        F98.9 未特指的儿童行为和情绪障碍：症状存在但缺乏分类依据。
    - F42 强迫障碍：反复的强迫观念/行为，个体自知过度或不合理但难以抵抗，耗时或致显著困扰/损害。
        F42.0 以强迫观念为主：反复出现难以摆脱的思想或冲动。
        F42.1 以强迫行为为主：反复、仪式化的动作难以控制。
        F42.2 强迫观念与强迫行为混合：思想和动作同时反复困扰。
        F42.9 强迫障碍，未特指：存在强迫症状但分类不详。
    - F31 双相情感障碍：既往或目前存在躁狂/轻躁狂发作与抑郁发作的交替或混合；需有明确躁狂谱系证据。
        F31.0 躁狂期，无精神病性症状：躁狂明显但无妄想或幻觉。
        F31.1 躁狂期，有精神病性症状：躁狂发作伴妄想或幻觉。
        F31.2 抑郁期，无精神病性症状：抑郁发作但无精神病性特征。
        F31.3 抑郁期，有精神病性症状：抑郁伴妄想或幻觉。
        F31.4 混合状态：躁狂与抑郁症状同时或快速交替出现。
        F31.5 缓解期：既往双相障碍，当前症状缓解。
        F31.6 其他状态：不符合典型躁狂/抑郁/混合的表现。
        F31.9 未特指：双相障碍，但无法进一步分类。
    - F43 对严重应激反应和适应障碍：与明确应激事件有关；可为急性应激反应、PTSD或适应障碍；核心包含再体验、回避、警觉性增高或与应激源相关的情绪/行为改变。
        F43.0 急性应激反应：暴露于重大应激后立即出现短暂严重反应。
        F43.1 创伤后应激障碍：经历创伤事件后持续出现再体验、回避和警觉性增高。
        F43.2 适应障碍：对生活变故反应过度，伴情绪或行为异常。
        F43.8 其他反应性障碍：与应激相关但不符合特定诊断。
        F43.9 未特指：应激反应存在，但资料不足以分类。
    - F45 躯体形式障碍：反复或多样躯体症状为主（如疼痛、心悸、胃肠不适等），检查难以找到足以解释的器质性原因或与病因不相称，显著痛苦/就诊反复。
        F45.0 躯体化障碍：反复多样的身体症状无器质性解释。
        F45.1 未分化的躯体形式障碍：躯体症状存在但未达到躯体化标准。
        F45.2 疑病障碍：持续担忧患严重疾病。
        F45.3 自主神经功能紊乱型：以心悸、胸闷等自主神经症状为主。
        F45.4 持续性躯体疼痛障碍：慢性疼痛为主要表现。
        F45.8 其他躯体形式障碍：特殊类型躯体症状但不归入上述类。
        F45.9 未特指：存在躯体症状但无法分类。
    - F20 精神分裂症：在知觉、思维、情感及行为等方面的广泛障碍；常见持续性妄想、幻听、思维松弛/破裂、情感淡漠、阴性症状，病程≥1月（或依本地标准）。
        F20.0 偏执型：以妄想和幻听为主。
        F20.1 紊乱型：思维、情感和行为紊乱显著。
        F20.2 紧张型：以木僵、紧张性兴奋为主要表现。
        F20.3 未分化型：符合精神分裂症但不属特定亚型。
        F20.4 残留状态：阴性症状为主，病程长期。
        F20.5 精神分裂症后抑郁：精神分裂症后出现显著抑郁。
        F20.6 单纯型：逐渐出现阴性症状，无显著阳性症状。
        F20.8 其他类型：特殊表现但不属于前述类别。
        F20.9 未特指：存在精神分裂症特征但资料不足。
    - Z71 咨询和医疗建议相关因素：包括心理咨询、健康教育、生活方式指导等，当患者主要需要咨询服务而非特定疾病治疗时使用。
        Z71.9 未特指的咨询：提供咨询，但缺乏具体分类。

## 注意：
1. 问诊对话为初次问诊，在症状严重程度和细节不可判断的时候，请推荐未特指的icd code。
2. 诊断结果可能包含1至2个icd-10诊断结果，大多只包含一个但不超过2个。
3. 用分号分隔不同的代码。
4. 需要严格根据icd-10标准来进行诊断的分析, 避免猜测和无根据的诊断，避免诊断错误。"""

        user_prompt = rf"""[问诊对话开始]
{text}
[问诊对话结束]

## 输出格式：
请用中文一步一步思考，并将思考过程放在<think>xxx</think>中输出，之后将最后诊断的ICD-10代码必须放在<box>xxx</box>中输出，用分号分隔，格式如：<think>xxx</think><box>Fxx.x;Fxx.x;Fxx.x</box>。"""


    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    return messages

# 检测API类型
def detect_api_type(base_url):
    """检测API类型：OpenRouter或本地vLLM"""
    if "openrouter.ai" in base_url:
        return "openrouter"
    elif "localhost" in base_url or "127.0.0.1" in base_url or base_url.startswith("http://0.0.0.0"):
        return "vllm"
    else:
        # 默认假设是vLLM本地部署
        return "vllm"

# 发送异步请求到API获取token概率（支持OpenRouter和本地vLLM）
async def get_token_probabilities_async(client, semaphore, messages, model_name, max_retries=3, top_logprobs=20, site_url="", site_name="", api_type="openrouter"):
    async with semaphore:  # 控制并发数量
        for attempt in range(max_retries):
            try:
                # 根据API类型构建不同的请求参数
                if api_type == "openrouter":
                    # OpenRouter特定的headers
                    extra_headers = {}
                    if site_url:
                        extra_headers["HTTP-Referer"] = site_url
                    if site_name:
                        extra_headers["X-Title"] = site_name
                    
                    # 使用OpenAI客户端通过OpenRouter发送请求
                    response = await client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        logprobs=True,  # 启用logprobs
                        top_logprobs=top_logprobs,
                        extra_headers=extra_headers,
                        extra_body={}
                    )
                else:  # vLLM本地部署
                    # vLLM API调用（兼容OpenAI格式）
                    response = await client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        logprobs=True,  # vLLM也支持logprobs
                        top_logprobs=top_logprobs,
                        # vLLM不需要extra_headers和extra_body
                    )
                
                # 兼容不同模型的reasoning字段名称：reasoning_content 或 reasoning
                reasoning_content = None
                if hasattr(response.choices[0].message, 'model_extra') and response.choices[0].message.model_extra:
                    if "reasoning_content" in response.choices[0].message.model_extra:
                        reasoning_content = response.choices[0].message.model_extra["reasoning_content"]
                    elif "reasoning" in response.choices[0].message.model_extra:
                        reasoning_content = response.choices[0].message.model_extra["reasoning"]
                
                if not response.choices[0].message.content.endswith("</box>"):
                    response.choices[0].message.content = response.choices[0].message.content + "</box>"
                
                if reasoning_content:
                    result = {
                        "choices": [{
                            "message": {
                                "content": "<think>\n" + reasoning_content + "\n</think>\n" + response.choices[0].message.content
                            },
                            "text": "<think>\n" + reasoning_content + "\n</think>\n" + response.choices[0].message.content,
                            "logprobs": {},
                            "reasoning_content": reasoning_content,
                            "answer": response.choices[0].message.content
                            
                        }]
                    }
                else:
                    result = {
                        "choices": [{
                            "message": {
                                "content": response.choices[0].message.content
                            },
                            "text": response.choices[0].message.content,
                            "logprobs": {},
                        }]
                    }
                
                # 如果响应中包含logprobs信息，则提取它
                if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
                    logprobs_data = response.choices[0].logprobs
                    if hasattr(logprobs_data, 'content') and logprobs_data.content:
                        # 提取tokens和top_logprobs
                        tokens = []
                        top_logprobs_list = []
                        
                        for token_data in logprobs_data.content:
                            tokens.append(token_data.token)
                            
                            # 构建top_logprobs字典
                            token_probs = {}
                            if hasattr(token_data, 'top_logprobs') and token_data.top_logprobs:
                                for top_token in token_data.top_logprobs:
                                    token_probs[top_token.token] = top_token.logprob
                            else:
                                # 如果没有top_logprobs，至少包含当前token
                                token_probs[token_data.token] = token_data.logprob
                            
                            top_logprobs_list.append(token_probs)
                        
                        result["choices"][0]["logprobs"] = {
                            "tokens": tokens,
                            "top_logprobs": top_logprobs_list
                        }
                
                return result
                
            except Exception as e:
                # 获取详细的错误信息
                error_type = type(e).__name__
                error_msg = str(e)
                error_traceback = traceback.format_exc()
                
                api_info = "vLLM本地部署" if api_type == "vllm" else "OpenRouter"
                logger.error(f"请求出错 (尝试 {attempt + 1}/{max_retries}) [{api_info}]:")
                logger.error(f"  错误类型: {error_type}")
                logger.error(f"  错误消息: {error_msg}")
                logger.error(f"  详细堆栈跟踪:")
                for line in error_traceback.split('\n'):
                    if line.strip():
                        logger.error(f"    {line}")
                
                # 如果是连接相关的错误，输出更多诊断信息
                if 'connection' in error_msg.lower() or 'connect' in error_msg.lower():
                    logger.error(f"  连接诊断信息:")
                    logger.error(f"    - API base URL: {client.base_url}")
                    logger.error(f"    - Model name: {model_name}")
                    logger.error(f"    - 建议检查网络连接和API服务状态")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避策略
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"已达到最大重试次数 ({max_retries})，放弃请求")
                    return None
        
        return None

# 从模型响应中提取概率（支持二分类、四分类和症状检测）
def extract_classification_probs(response, classification_mode="binary", debug=False, ground_truth=None):
    if classification_mode == "binary":
        return extract_binary_probs(response, debug, ground_truth)
    elif classification_mode == "multiclass":
        return extract_multiclass_probs(response, debug, ground_truth)
    elif classification_mode == "depression_symptom":
        return extract_depression_symptom_probs(response, debug, ground_truth)
    elif classification_mode == "anxiety_symptom":
        return extract_anxiety_symptom_probs(response, debug, ground_truth)
    elif classification_mode == "multilabel":
        return extract_multilabel_probs(response, debug, ground_truth)
    elif classification_mode == "icd10":
        return extract_icd10_probs(response, debug, ground_truth)
    elif classification_mode == "recommendation":
        return extract_recommendation_codes(response, debug, ground_truth)
    else:
        # 默认返回适当的None值
        return None, None, {}, {}

# 备用概率计算函数（二分类）- 无logprobs时使用
def extract_binary_probs_fallback(response_text, debug=False, ground_truth=None):
    """
    在没有logprobs的情况下，根据预测结果和正确答案比较来设置概率
    如果预测正确，对应类别概率设为1，其他为0
    如果预测错误，对应类别概率设为0，其他为1（或平均分配）
    """
    # 提取<box>标签内的内容
    box_pattern = r'<box>(.*?)</box>'
    box_matches = list(re.finditer(box_pattern, response_text, re.DOTALL))
    
    if not box_matches:
        if debug:
            logger.debug("未找到<box>标签，返回None")
        return None, None, {}, {}
    
    # 使用最后一个匹配的<box>标签
    predicted_text = box_matches[-1].group(1).strip()
    
    if debug:
        logger.debug(f"提取的预测结果: '{predicted_text}'")
        logger.debug(f"Ground truth: {ground_truth}")
    
    # 标准化预测结果
    predicted_label = None
    if any(keyword in predicted_text for keyword in ["抑郁", "depression", "Depression"]):
        predicted_label = "Depression"
    elif any(keyword in predicted_text for keyword in ["焦虑", "anxiety", "Anxiety"]):
        predicted_label = "Anxiety"
    
    if debug:
        logger.debug(f"标准化后的预测标签: {predicted_label}")
    
    # 如果没有ground_truth，无法判断正确性，返回默认概率
    if ground_truth is None:
        if predicted_label == "Depression":
            return 1.0, 0.0, {"Depression": 1.0}, {"Anxiety": 0.0}
        elif predicted_label == "Anxiety":
            return 0.0, 1.0, {"Depression": 0.0}, {"Anxiety": 1.0}
        else:
            return 0.5, 0.5, {"Unknown": 0.5}, {"Unknown": 0.5}
    
    # 标准化ground truth
    ground_truth_normalized = ground_truth.strip() if ground_truth else None
    
    # 比较预测结果和正确答案
    is_correct = (predicted_label == ground_truth_normalized)
    
    if debug:
        logger.debug(f"预测是否正确: {is_correct}")
    
    # 根据正确性设置概率
    if is_correct:
        if predicted_label == "Depression":
            depression_prob, anxiety_prob = 1.0, 0.0
            depression_tokens = {"Depression": 1.0}
            anxiety_tokens = {"Anxiety": 0.0}
        elif predicted_label == "Anxiety":
            depression_prob, anxiety_prob = 0.0, 1.0
            depression_tokens = {"Depression": 0.0}
            anxiety_tokens = {"Anxiety": 1.0}
        else:
            # 预测结果不明确但ground_truth存在的情况
            depression_prob, anxiety_prob = 0.0, 0.0
            depression_tokens = {"Unknown": 0.0}
            anxiety_tokens = {"Unknown": 0.0}
    else:
        # 预测错误，给正确答案设置概率1
        if ground_truth_normalized == "Depression":
            depression_prob, anxiety_prob = 1.0, 0.0
            depression_tokens = {"Depression": 1.0}
            anxiety_tokens = {"Anxiety": 0.0}
        elif ground_truth_normalized == "Anxiety":
            depression_prob, anxiety_prob = 0.0, 1.0
            depression_tokens = {"Depression": 0.0}
            anxiety_tokens = {"Anxiety": 1.0}
        else:
            # ground_truth也不明确的情况
            depression_prob, anxiety_prob = 0.5, 0.5
            depression_tokens = {"Unknown": 0.5}
            anxiety_tokens = {"Unknown": 0.5}
    
    if debug:
        logger.debug(f"最终概率 - 抑郁: {depression_prob}, 焦虑: {anxiety_prob}")
    
    return depression_prob, anxiety_prob, depression_tokens, anxiety_tokens

# 从模型响应中提取抑郁症和焦虑症的概率（二分类）
def extract_binary_probs(response, debug=False, ground_truth=None):
    if not response or 'choices' not in response or not response['choices']:
        return None, None, {}, {}
    
    # 获取完整的响应文本
    response_text = response['choices'][0].get('text', '')
    logprobs_info = response['choices'][0].get('logprobs', {})
    
    # 如果没有logprobs信息，使用备用逻辑
    if not logprobs_info or 'top_logprobs' not in logprobs_info or not logprobs_info['top_logprobs']:
        if debug:
            logger.debug("没有logprobs信息，使用备用概率计算逻辑")
        return extract_binary_probs_fallback(response_text, debug, ground_truth)
    
    # 获取所有token的概率分布
    all_token_probs = logprobs_info['top_logprobs']
    tokens = logprobs_info.get('tokens', [])
    
    if debug:
        logger.debug(f"响应文本: {response_text}")
        logger.debug(f"Token数量: {len(tokens)}")
    
    # 寻找最后一个<box>标签的位置
    box_start_idx = None
    box_end_idx = None
    
    # 使用正则表达式提取<box>标签内的内容
    accumulated_text = "".join(tokens)
    box_pattern = r'<box>(.*?)</box>'
    box_matches = list(re.finditer(box_pattern, accumulated_text, re.DOTALL))
    
    if box_matches:
        # 使用最后一个匹配的<box>标签
        last_match = box_matches[-1]
        box_content = last_match.group(1)
        box_start_pos = last_match.start() + 5  # <box>的长度是5
        box_end_pos = last_match.end() - 6      # </box>的长度是6
        
        if debug:
            logger.debug(f"找到<box>标签内容: '{box_content}'")
            logger.debug(f"<box>标签在文本中的位置: {box_start_pos}-{box_end_pos}")
        
        # 在token序列中找到对应的位置
        current_pos = 0
        for i, token in enumerate(tokens):
            token_end_pos = current_pos + len(token)
            
            # 如果当前token的结束位置超过了box开始位置，记录开始索引
            if box_start_idx is None and token_end_pos > box_start_pos:
                box_start_idx = i
                if debug:
                    logger.debug(f"<box>内容开始token位置: {box_start_idx}")
            
            # 如果当前token的结束位置达到或超过了box结束位置，记录结束索引
            if box_end_idx is None and token_end_pos >= box_end_pos:
                box_end_idx = i + 1  # +1因为range是左闭右开
                if debug:
                    logger.debug(f"<box>内容结束token位置: {box_end_idx}")
                break
                
            current_pos = token_end_pos
    
    # 如果没有找到<box>标签，回退到分析前几个token
    if box_start_idx is None or box_end_idx is None:
        if debug:
            logger.debug("未找到<box>标签，分析前几个token")
        # 分析前5个token
        box_start_idx = 0
        box_end_idx = min(5, len(all_token_probs))
    
    if debug:
        logger.debug(f"分析Token范围: {box_start_idx} 到 {box_end_idx}")
    
    # 分析<box>标签内（或前几个token）的概率
    depression_prob = 0
    anxiety_prob = 0
    depression_matches = {}
    anxiety_matches = {}
    
    # 搜索关键词
    depression_keywords = ["抑郁", "抑郁症", "depression", "Depression"]
    anxiety_keywords = ["焦虑", "焦虑症", "anxiety", "Anxiety"]
    
    # 首先尝试从原始响应文本中直接匹配（最优先和可靠的方案）
    box_pattern = r'<box>(.*?)</box>'
    original_box_matches = list(re.finditer(box_pattern, response_text, re.DOTALL))
    if original_box_matches:
        box_content = original_box_matches[-1].group(1).strip()
        if debug:
            logger.debug(f"从原始文本提取的<box>内容: '{box_content}'")
        
        # 直接检查box内容是否包含关键词
        for dep_keyword in depression_keywords:
            if dep_keyword in box_content:
                depression_prob += 0.9  # 提高概率权重，因为这是最可靠的匹配
                depression_matches[f"direct_match_{dep_keyword}"] = 0.9
                if debug:
                    logger.debug(f"在<box>内容中直接找到抑郁关键词: {dep_keyword}")
                break
        
        for anx_keyword in anxiety_keywords:
            if anx_keyword in box_content:
                anxiety_prob += 0.9  # 提高概率权重，因为这是最可靠的匹配
                anxiety_matches[f"direct_match_{anx_keyword}"] = 0.9
                if debug:
                    logger.debug(f"在<box>内容中直接找到焦虑关键词: {anx_keyword}")
                break
    
    # 如果直接匹配已经成功，优先返回结果（避免token级别分析的复杂性）
    if depression_prob > 0 or anxiety_prob > 0:
        if debug:
            logger.debug(f"直接匹配成功，抑郁概率: {depression_prob}, 焦虑概率: {anxiety_prob}")
            logger.debug(f"跳过token级别分析，直接返回结果")
        return depression_prob, anxiety_prob, depression_matches, anxiety_matches
    
    # 分析指定范围内的所有token
    for token_idx in range(box_start_idx, min(box_end_idx, len(all_token_probs))):
        if token_idx >= len(all_token_probs):
            break
            
        token_probs = all_token_probs[token_idx]
        current_token = tokens[token_idx] if token_idx < len(tokens) else ""
        
        if debug:
            logger.debug(f"分析Token {token_idx}: '{current_token}'")
        
        for token, logprob in token_probs.items():
            prob = math.exp(logprob)
            
            # 使用tokenizer解码token
            decoded_text = ""
            if tokenizer is not None:
                try:
                    # 方法1: 直接decode token string
                    try:
                        token_id = tokenizer.convert_tokens_to_ids(token)
                        if token_id != tokenizer.unk_token_id:
                            decoded_text = tokenizer.decode([token_id], skip_special_tokens=True)
                    except:
                        pass
                    
                    # 方法2: 如果方法1失败，尝试将token作为subword处理
                    if not decoded_text:
                        try:
                            clean_token = token
                            if token.startswith('Ġ'):
                                clean_token = token[1:]
                            
                            vocab = tokenizer.get_vocab()
                            if token in vocab:
                                token_id = vocab[token]
                                decoded_text = tokenizer.decode([token_id], skip_special_tokens=True)
                        except:
                            pass
                        
                except Exception as e:
                    if debug:
                        logger.debug(f"Token解码失败: {token}, 错误: {e}")
                    decoded_text = token
            else:
                decoded_text = token
            
            # 检查解码后的文本以及原始token
            texts_to_check = [decoded_text, token]
            
            # 移除空格前缀的变体
            if token.startswith('Ġ'):
                texts_to_check.append(token[1:])
            
            found_match = False
            for text in texts_to_check:
                if not text:
                    continue
                    
                # 检查是否是抑郁相关token
                if any(dep_keyword in text for dep_keyword in depression_keywords):
                    depression_prob += prob
                    depression_matches[f"{token_idx}_{token}"] = prob
                    if debug:
                        logger.debug(f"找到抑郁token: {token} -> '{decoded_text}' (位置: {token_idx})")
                    found_match = True
                    break
                    
                # 检查是否是焦虑相关token  
                if any(anx_keyword in text for anx_keyword in anxiety_keywords):
                    anxiety_prob += prob
                    anxiety_matches[f"{token_idx}_{token}"] = prob
                    if debug:
                        logger.debug(f"找到焦虑token: {token} -> '{decoded_text}' (位置: {token_idx})")
                    found_match = True
                    break
            
            if found_match:
                continue
    
    if debug:
        if depression_matches:
            logger.debug(f"找到抑郁相关token: {depression_matches}")
        if anxiety_matches:
            logger.debug(f"找到焦虑相关token: {anxiety_matches}")
        logger.debug(f"抑郁概率总和: {depression_prob}")
        logger.debug(f"焦虑概率总和: {anxiety_prob}")
    
    return depression_prob, anxiety_prob, depression_matches, anxiety_matches

# 备用概率计算函数（四分类）- 无logprobs时使用
def extract_multiclass_probs_fallback(response_text, debug=False, ground_truth=None):
    """四分类备用概率计算"""
    # 提取<box>标签内的内容
    box_pattern = r'<box>(.*?)</box>'
    box_matches = list(re.finditer(box_pattern, response_text, re.DOTALL))
    
    if not box_matches:
        if debug:
            logger.debug("未找到<box>标签，返回None")
        return None, None, None, None, {}, {}, {}, {}
    
    predicted_text = box_matches[-1].group(1).strip()
    
    # 标准化预测结果
    predicted_label = None
    if any(keyword in predicted_text for keyword in ["抑郁", "depression", "Depression"]):
        predicted_label = "Depression"
    elif any(keyword in predicted_text for keyword in ["焦虑", "anxiety", "Anxiety"]):
        predicted_label = "Anxiety"
    elif any(keyword in predicted_text for keyword in ["mix", "Mix", "混合"]):
        predicted_label = "Mix"
    elif any(keyword in predicted_text for keyword in ["others", "Others", "其他", "正常"]):
        predicted_label = "Others"
    
    # 如果没有ground_truth或预测结果不明确，返回默认概率
    if ground_truth is None or predicted_label is None:
        if predicted_label == "Depression":
            return 1.0, 0.0, 0.0, 0.0, {"Depression": 1.0}, {}, {}, {}
        elif predicted_label == "Anxiety":
            return 0.0, 1.0, 0.0, 0.0, {}, {"Anxiety": 1.0}, {}, {}
        elif predicted_label == "Mix":
            return 0.0, 0.0, 1.0, 0.0, {}, {}, {"Mix": 1.0}, {}
        elif predicted_label == "Others":
            return 0.0, 0.0, 0.0, 1.0, {}, {}, {}, {"Others": 1.0}
        else:
            return 0.25, 0.25, 0.25, 0.25, {}, {}, {}, {}
    
    # 比较预测结果和正确答案
    is_correct = (predicted_label == ground_truth.strip() if ground_truth else False)
    
    if is_correct:
        if predicted_label == "Depression":
            return 1.0, 0.0, 0.0, 0.0, {"Depression": 1.0}, {"Anxiety": 0.0}, {"Mix": 0.0}, {"Others": 0.0}
        elif predicted_label == "Anxiety":
            return 0.0, 1.0, 0.0, 0.0, {"Depression": 0.0}, {"Anxiety": 1.0}, {"Mix": 0.0}, {"Others": 0.0}
        elif predicted_label == "Mix":
            return 0.0, 0.0, 1.0, 0.0, {"Depression": 0.0}, {"Anxiety": 0.0}, {"Mix": 1.0}, {"Others": 0.0}
        elif predicted_label == "Others":
            return 0.0, 0.0, 0.0, 1.0, {"Depression": 0.0}, {"Anxiety": 0.0}, {"Mix": 0.0}, {"Others": 1.0}
    else:
        # 预测错误，给正确答案设置概率1
        if ground_truth == "Depression":
            return 1.0, 0.0, 0.0, 0.0, {"Depression": 1.0}, {"Anxiety": 0.0}, {"Mix": 0.0}, {"Others": 0.0}
        elif ground_truth == "Anxiety":
            return 0.0, 1.0, 0.0, 0.0, {"Depression": 0.0}, {"Anxiety": 1.0}, {"Mix": 0.0}, {"Others": 0.0}
        elif ground_truth == "Mix":
            return 0.0, 0.0, 1.0, 0.0, {"Depression": 0.0}, {"Anxiety": 0.0}, {"Mix": 1.0}, {"Others": 0.0}
        elif ground_truth == "Others":
            return 0.0, 0.0, 0.0, 1.0, {"Depression": 0.0}, {"Anxiety": 0.0}, {"Mix": 0.0}, {"Others": 1.0}
        else:
            return 0.25, 0.25, 0.25, 0.25, {}, {}, {}, {}

# 从模型响应中提取四分类概率（抑郁/焦虑/mix/others）
def extract_multiclass_probs(response, debug=False, ground_truth=None):
    if not response or 'choices' not in response or not response['choices']:
        return None, None, None, None, {}, {}, {}, {}
    
    # 获取完整的响应文本
    response_text = response['choices'][0].get('text', '')
    logprobs_info = response['choices'][0].get('logprobs', {})
    
    # 如果没有logprobs信息，使用备用逻辑
    if not logprobs_info or 'top_logprobs' not in logprobs_info or not logprobs_info['top_logprobs']:
        if debug:
            logger.debug("没有logprobs信息，使用备用概率计算逻辑")
        return extract_multiclass_probs_fallback(response_text, debug, ground_truth)
    
    # 获取所有token的概率分布
    all_token_probs = logprobs_info['top_logprobs']
    tokens = logprobs_info.get('tokens', [])
    
    if debug:
        logger.debug(f"响应文本: {response_text}")
        logger.debug(f"Token数量: {len(tokens)}")
    
    # 寻找最后一个<box>标签的位置
    box_start_idx = None
    box_end_idx = None
    
    # 使用正则表达式提取<box>标签内的内容
    accumulated_text = "".join(tokens)
    box_pattern = r'<box>(.*?)</box>'
    box_matches = list(re.finditer(box_pattern, accumulated_text, re.DOTALL))
    
    if box_matches:
        # 使用最后一个匹配的<box>标签
        last_match = box_matches[-1]
        box_content = last_match.group(1)
        box_start_pos = last_match.start() + 5  # <box>的长度是5
        box_end_pos = last_match.end() - 6      # </box>的长度是6
        
        if debug:
            logger.debug(f"找到<box>标签内容: '{box_content}'")
            logger.debug(f"<box>标签在文本中的位置: {box_start_pos}-{box_end_pos}")
        
        # 在token序列中找到对应的位置
        current_pos = 0
        for i, token in enumerate(tokens):
            token_end_pos = current_pos + len(token)
            
            # 如果当前token的结束位置超过了box开始位置，记录开始索引
            if box_start_idx is None and token_end_pos > box_start_pos:
                box_start_idx = i
                if debug:
                    logger.debug(f"<box>内容开始token位置: {box_start_idx}")
            
            # 如果当前token的结束位置达到或超过了box结束位置，记录结束索引
            if box_end_idx is None and token_end_pos >= box_end_pos:
                box_end_idx = i + 1  # +1因为range是左闭右开
                if debug:
                    logger.debug(f"<box>内容结束token位置: {box_end_idx}")
                break
                
            current_pos = token_end_pos
    
    # 如果没有找到<box>标签，回退到分析前几个token
    if box_start_idx is None or box_end_idx is None:
        if debug:
            logger.debug("未找到<box>标签，分析前几个token")
        # 分析前5个token
        box_start_idx = 0
        box_end_idx = min(5, len(all_token_probs))
    
    if debug:
        logger.debug(f"分析Token范围: {box_start_idx} 到 {box_end_idx}")
    
    # 分析<box>标签内（或前几个token）的概率
    depression_prob = 0
    anxiety_prob = 0
    mix_prob = 0
    others_prob = 0
    depression_matches = {}
    anxiety_matches = {}
    mix_matches = {}
    others_matches = {}
    
    # 搜索关键词
    depression_keywords = ["抑郁", "抑郁症", "depression", "Depression"]
    anxiety_keywords = ["焦虑", "焦虑症", "anxiety", "Anxiety"] 
    mix_keywords = ["mix", "Mix", "混合", "共病", "同时"]
    others_keywords = ["others", "Others", "其他", "其它", "正常", "无", "other"]
    
    # 首先尝试从原始响应文本中直接匹配（作为备用方案）
    box_pattern = r'<box>(.*?)</box>'
    original_box_matches = list(re.finditer(box_pattern, response_text, re.DOTALL))
    if original_box_matches:
        box_content = original_box_matches[-1].group(1).strip()
        if debug:
            logger.debug(f"从原始文本提取的<box>内容: '{box_content}'")
        
        # 直接检查box内容是否包含关键词
        for dep_keyword in depression_keywords:
            if dep_keyword in box_content:
                depression_prob += 0.8
                depression_matches[f"direct_match_{dep_keyword}"] = 0.8
                if debug:
                    logger.debug(f"在<box>内容中直接找到抑郁关键词: {dep_keyword}")
                break
        
        for anx_keyword in anxiety_keywords:
            if anx_keyword in box_content:
                anxiety_prob += 0.8
                anxiety_matches[f"direct_match_{anx_keyword}"] = 0.8
                if debug:
                    logger.debug(f"在<box>内容中直接找到焦虑关键词: {anx_keyword}")
                break
        
        for mix_keyword in mix_keywords:
            if mix_keyword in box_content:
                mix_prob += 0.8
                mix_matches[f"direct_match_{mix_keyword}"] = 0.8
                if debug:
                    logger.debug(f"在<box>内容中直接找到mix关键词: {mix_keyword}")
                break
        
        for others_keyword in others_keywords:
            if others_keyword in box_content:
                others_prob += 0.8
                others_matches[f"direct_match_{others_keyword}"] = 0.8
                if debug:
                    logger.debug(f"在<box>内容中直接找到others关键词: {others_keyword}")
                break
    
    # 如果直接匹配已经成功，优先返回结果（避免token级别分析的复杂性）
    if depression_prob > 0 or anxiety_prob > 0 or mix_prob > 0 or others_prob > 0:
        if debug:
            logger.debug(f"直接匹配成功，抑郁: {depression_prob}, 焦虑: {anxiety_prob}, mix: {mix_prob}, others: {others_prob}")
            logger.debug(f"跳过token级别分析，直接返回结果")
        return depression_prob, anxiety_prob, mix_prob, others_prob, depression_matches, anxiety_matches, mix_matches, others_matches
    
    # 分析指定范围内的所有token
    for token_idx in range(box_start_idx, min(box_end_idx, len(all_token_probs))):
        if token_idx >= len(all_token_probs):
            break
            
        token_probs = all_token_probs[token_idx]
        current_token = tokens[token_idx] if token_idx < len(tokens) else ""
        
        if debug:
            logger.debug(f"分析Token {token_idx}: '{current_token}'")
        
        for token, logprob in token_probs.items():
            prob = math.exp(logprob)
            
            # 使用tokenizer解码token
            decoded_text = ""
            if tokenizer is not None:
                try:
                    # 方法1: 直接decode token string
                    try:
                        token_id = tokenizer.convert_tokens_to_ids(token)
                        if token_id != tokenizer.unk_token_id:
                            decoded_text = tokenizer.decode([token_id], skip_special_tokens=True)
                    except:
                        pass
                    
                    # 方法2: 如果方法1失败，尝试将token作为subword处理
                    if not decoded_text:
                        try:
                            clean_token = token
                            if token.startswith('Ġ'):
                                clean_token = token[1:]
                            
                            vocab = tokenizer.get_vocab()
                            if token in vocab:
                                token_id = vocab[token]
                                decoded_text = tokenizer.decode([token_id], skip_special_tokens=True)
                        except:
                            pass
                        
                except Exception as e:
                    if debug:
                        logger.debug(f"Token解码失败: {token}, 错误: {e}")
                    decoded_text = token
            else:
                decoded_text = token
            
            # 检查解码后的文本以及原始token
            texts_to_check = [decoded_text, token]
            
            # 移除空格前缀的变体
            if token.startswith('Ġ'):
                texts_to_check.append(token[1:])
            
            found_match = False
            for text in texts_to_check:
                if not text:
                    continue
                    
                # 检查是否是抑郁相关token
                if any(dep_keyword in text for dep_keyword in depression_keywords):
                    depression_prob += prob
                    depression_matches[f"{token_idx}_{token}"] = prob
                    if debug:
                        logger.debug(f"找到抑郁token: {token} -> '{decoded_text}' (位置: {token_idx})")
                    found_match = True
                    break
                    
                # 检查是否是焦虑相关token  
                if any(anx_keyword in text for anx_keyword in anxiety_keywords):
                    anxiety_prob += prob
                    anxiety_matches[f"{token_idx}_{token}"] = prob
                    if debug:
                        logger.debug(f"找到焦虑token: {token} -> '{decoded_text}' (位置: {token_idx})")
                    found_match = True
                    break
                
                # 检查是否是mix相关token
                if any(mix_keyword in text for mix_keyword in mix_keywords):
                    mix_prob += prob
                    mix_matches[f"{token_idx}_{token}"] = prob
                    if debug:
                        logger.debug(f"找到mix token: {token} -> '{decoded_text}' (位置: {token_idx})")
                    found_match = True
                    break
                
                # 检查是否是others相关token
                if any(others_keyword in text for others_keyword in others_keywords):
                    others_prob += prob
                    others_matches[f"{token_idx}_{token}"] = prob
                    if debug:
                        logger.debug(f"找到others token: {token} -> '{decoded_text}' (位置: {token_idx})")
                    found_match = True
                    break
            
            if found_match:
                continue
    
    if debug:
        logger.debug(f"抑郁概率总和: {depression_prob}")
        logger.debug(f"焦虑概率总和: {anxiety_prob}")
        logger.debug(f"mix概率总和: {mix_prob}")
        logger.debug(f"others概率总和: {others_prob}")
    
    return depression_prob, anxiety_prob, mix_prob, others_prob, depression_matches, anxiety_matches, mix_matches, others_matches

# 从模型响应中提取抑郁症状存在概率
def extract_depression_symptom_probs(response, debug=False, ground_truth=None):
    if not response or 'choices' not in response or not response['choices']:
        return None, None, {}, {}
    
    # 获取完整的响应文本
    response_text = response['choices'][0].get('text', '')
    logprobs_info = response['choices'][0].get('logprobs', {})
    
    if not logprobs_info or 'top_logprobs' not in logprobs_info or not logprobs_info['top_logprobs']:
        if debug:
            logger.debug("没有logprobs信息，使用直接文本匹配备用逻辑")
        
        # 直接从响应文本中提取<box>内容作为备用方案
        box_pattern = r'<box>(.*?)</box>'
        original_box_matches = list(re.finditer(box_pattern, response_text, re.DOTALL))
        if original_box_matches:
            box_content = original_box_matches[-1].group(1).strip()
            if debug:
                logger.debug(f"从原始文本提取的抑郁症状<box>内容: '{box_content}'")
            
            # 检查是否包含"是"或"否"相关关键词
            has_symptom_keywords = ["有", "存在", "是", "yes", "Yes", "有症状", "有抑郁症状"]
            no_symptom_keywords = ["没有", "无", "否", "不存在", "no", "No", "无症状", "没有症状", "无抑郁症状"]
            
            has_symptom_prob = 0
            no_symptom_prob = 0
            has_symptom_matches = {}
            no_symptom_matches = {}
            
            # 检查关键词匹配
            for keyword in has_symptom_keywords:
                if keyword in box_content:
                    has_symptom_prob = 0.9
                    has_symptom_matches[f"direct_match_{keyword}"] = 0.9
                    if debug:
                        logger.debug(f"直接匹配到有症状关键词: {keyword}")
                    break
            
            if has_symptom_prob == 0:  # 如果没有找到"有症状"关键词，检查"无症状"
                for keyword in no_symptom_keywords:
                    if keyword in box_content:
                        no_symptom_prob = 0.9
                        no_symptom_matches[f"direct_match_{keyword}"] = 0.9
                        if debug:
                            logger.debug(f"直接匹配到无症状关键词: {keyword}")
                        break
            
            # 如果都没有找到，给一个默认的无症状概率
            if has_symptom_prob == 0 and no_symptom_prob == 0:
                no_symptom_prob = 0.1
                no_symptom_matches["default_no_symptom"] = 0.1
                if debug:
                    logger.debug("未找到明确关键词，默认设为无症状")
            
            return has_symptom_prob, no_symptom_prob, has_symptom_matches, no_symptom_matches
        else:
            if debug:
                logger.debug("未找到<box>标签，返回默认无症状")
            return 0, 0.1, {}, {"default_no_symptom": 0.1}
    
    # 获取所有token的概率分布
    all_token_probs = logprobs_info['top_logprobs']
    tokens = logprobs_info.get('tokens', [])
    
    if debug:
        logger.debug(f"响应文本: {response_text}")
        logger.debug(f"Token数量: {len(tokens)}")
    
    # 寻找最后一个<box>标签的位置
    box_start_idx = None
    box_end_idx = None
    
    # 使用正则表达式提取<box>标签内的内容
    accumulated_text = "".join(tokens)
    box_pattern = r'<box>(.*?)</box>'
    box_matches = list(re.finditer(box_pattern, accumulated_text, re.DOTALL))
    
    if box_matches:
        # 使用最后一个匹配的<box>标签
        last_match = box_matches[-1]
        box_content = last_match.group(1)
        box_start_pos = last_match.start() + 5  # <box>的长度是5
        box_end_pos = last_match.end() - 6      # </box>的长度是6
        
        if debug:
            logger.debug(f"找到<box>标签内容: '{box_content}'")
            logger.debug(f"<box>标签在文本中的位置: {box_start_pos}-{box_end_pos}")
        
        # 在token序列中找到对应的位置
        current_pos = 0
        for i, token in enumerate(tokens):
            token_end_pos = current_pos + len(token)
            
            # 如果当前token的结束位置超过了box开始位置，记录开始索引
            if box_start_idx is None and token_end_pos > box_start_pos:
                box_start_idx = i
                if debug:
                    logger.debug(f"<box>内容开始token位置: {box_start_idx}")
            
            # 如果当前token的结束位置达到或超过了box结束位置，记录结束索引
            if box_end_idx is None and token_end_pos >= box_end_pos:
                box_end_idx = i + 1  # +1因为range是左闭右开
                if debug:
                    logger.debug(f"<box>内容结束token位置: {box_end_idx}")
                break
                
            current_pos = token_end_pos
    
    # 如果没有找到<box>标签，回退到分析前几个token
    if box_start_idx is None or box_end_idx is None:
        if debug:
            logger.debug("未找到<box>标签，分析前几个token")
        # 分析前5个token
        box_start_idx = 0
        box_end_idx = min(5, len(all_token_probs))
    
    if debug:
        logger.debug(f"分析Token范围: {box_start_idx} 到 {box_end_idx}")
    
    # 分析<box>标签内（或前几个token）的概率
    has_symptom_prob = 0
    no_symptom_prob = 0
    has_symptom_matches = {}
    no_symptom_matches = {}
    
    # 搜索关键词
    has_symptom_keywords = ["有", "存在", "是", "yes", "Yes", "有症状", "有抑郁症状"]
    no_symptom_keywords = ["没有", "无", "否", "不存在", "no", "No", "无症状", "没有症状", "无抑郁症状"]
    
    # 首先尝试从原始响应文本中直接匹配（最优先和可靠的方案）
    box_pattern = r'<box>(.*?)</box>'
    original_box_matches = list(re.finditer(box_pattern, response_text, re.DOTALL))
    if original_box_matches:
        box_content = original_box_matches[-1].group(1).strip()
        if debug:
            logger.debug(f"从原始文本提取的抑郁症状<box>内容: '{box_content}'")
        
        # 直接检查box内容是否包含关键词
        for keyword in has_symptom_keywords:
            if keyword in box_content:
                has_symptom_prob += 0.9
                has_symptom_matches[f"direct_match_{keyword}"] = 0.9
                if debug:
                    logger.debug(f"在<box>内容中直接找到有症状关键词: {keyword}")
                break
        
        if has_symptom_prob == 0:  # 如果没有找到"有症状"关键词，检查"无症状"
            for keyword in no_symptom_keywords:
                if keyword in box_content:
                    no_symptom_prob += 0.9
                    no_symptom_matches[f"direct_match_{keyword}"] = 0.9
                    if debug:
                        logger.debug(f"在<box>内容中直接找到无症状关键词: {keyword}")
                    break
    
    # 如果直接匹配已经成功，优先返回结果（避免token级别分析的复杂性）
    if has_symptom_prob > 0 or no_symptom_prob > 0:
        if debug:
            logger.debug(f"直接匹配成功，有症状: {has_symptom_prob}, 无症状: {no_symptom_prob}")
            logger.debug(f"跳过token级别分析，直接返回结果")
        return has_symptom_prob, no_symptom_prob, has_symptom_matches, no_symptom_matches
    
    # 分析指定范围内的所有token
    for token_idx in range(box_start_idx, min(box_end_idx, len(all_token_probs))):
        if token_idx >= len(all_token_probs):
            break
            
        token_probs = all_token_probs[token_idx]
        current_token = tokens[token_idx] if token_idx < len(tokens) else ""
        
        if debug:
            logger.debug(f"分析Token {token_idx}: '{current_token}'")
        
        for token, logprob in token_probs.items():
            prob = math.exp(logprob)
            
            # 使用tokenizer解码token
            decoded_text = ""
            if tokenizer is not None:
                try:
                    # 方法1: 直接decode token string
                    try:
                        token_id = tokenizer.convert_tokens_to_ids(token)
                        if token_id != tokenizer.unk_token_id:
                            decoded_text = tokenizer.decode([token_id], skip_special_tokens=True)
                    except:
                        pass
                    
                    # 方法2: 如果方法1失败，尝试将token作为subword处理
                    if not decoded_text:
                        try:
                            clean_token = token
                            if token.startswith('Ġ'):
                                clean_token = token[1:]
                            
                            vocab = tokenizer.get_vocab()
                            if token in vocab:
                                token_id = vocab[token]
                                decoded_text = tokenizer.decode([token_id], skip_special_tokens=True)
                        except:
                            pass
                        
                except Exception as e:
                    if debug:
                        logger.debug(f"Token解码失败: {token}, 错误: {e}")
                    decoded_text = token
            else:
                decoded_text = token
            
            # 检查解码后的文本以及原始token
            texts_to_check = [decoded_text, token]
            
            # 移除空格前缀的变体
            if token.startswith('Ġ'):
                texts_to_check.append(token[1:])
            
            found_match = False
            for text in texts_to_check:
                if not text:
                    continue
                    
                # 检查是否是"有症状"相关token
                if any(keyword in text for keyword in has_symptom_keywords):
                    has_symptom_prob += prob
                    has_symptom_matches[f"{token_idx}_{token}"] = prob
                    if debug:
                        logger.debug(f"找到有症状token: {token} -> '{decoded_text}' (位置: {token_idx})")
                    found_match = True
                    break
                    
                # 检查是否是"无症状"相关token  
                if any(keyword in text for keyword in no_symptom_keywords):
                    no_symptom_prob += prob
                    no_symptom_matches[f"{token_idx}_{token}"] = prob
                    if debug:
                        logger.debug(f"找到无症状token: {token} -> '{decoded_text}' (位置: {token_idx})")
                    found_match = True
                    break
            
            if found_match:
                continue
    
    if debug:
        if has_symptom_matches:
            logger.debug(f"找到有症状相关token: {has_symptom_matches}")
        if no_symptom_matches:
            logger.debug(f"找到无症状相关token: {no_symptom_matches}")
        logger.debug(f"有症状概率总和: {has_symptom_prob}")
        logger.debug(f"无症状概率总和: {no_symptom_prob}")
    
    return has_symptom_prob, no_symptom_prob, has_symptom_matches, no_symptom_matches

# 从模型响应中提取焦虑症状存在概率
def extract_anxiety_symptom_probs(response, debug=False, ground_truth=None):
    if not response or 'choices' not in response or not response['choices']:
        return None, None, {}, {}
    
    # 获取完整的响应文本
    response_text = response['choices'][0].get('text', '')
    logprobs_info = response['choices'][0].get('logprobs', {})
    
    if not logprobs_info or 'top_logprobs' not in logprobs_info or not logprobs_info['top_logprobs']:
        if debug:
            logger.debug("没有logprobs信息，使用直接文本匹配备用逻辑")
        
        # 直接从响应文本中提取<box>内容作为备用方案
        box_pattern = r'<box>(.*?)</box>'
        original_box_matches = list(re.finditer(box_pattern, response_text, re.DOTALL))
        if original_box_matches:
            box_content = original_box_matches[-1].group(1).strip()
            if debug:
                logger.debug(f"从原始文本提取的焦虑症状<box>内容: '{box_content}'")
            
            # 检查是否包含"是"或"否"相关关键词
            has_symptom_keywords = ["是","有", "存在", "yes", "Yes", "有症状", "有焦虑症状", "有抑郁症状"]
            no_symptom_keywords = ["否", "没有", "无", "不存在", "no", "No", "无症状", "没有症状", "无焦虑症状", "无抑郁症状"]
            
            has_symptom_prob = 0
            no_symptom_prob = 0
            has_symptom_matches = {}
            no_symptom_matches = {}
            
            # 检查关键词匹配
            for keyword in has_symptom_keywords:
                if keyword in box_content:
                    has_symptom_prob = 0.9
                    has_symptom_matches[f"direct_match_{keyword}"] = 0.9
                    if debug:
                        logger.debug(f"直接匹配到有症状关键词: {keyword}")
                    break
            
            if has_symptom_prob == 0:  # 如果没有找到"有症状"关键词，检查"无症状"
                for keyword in no_symptom_keywords:
                    if keyword in box_content:
                        no_symptom_prob = 0.9
                        no_symptom_matches[f"direct_match_{keyword}"] = 0.9
                        if debug:
                            logger.debug(f"直接匹配到无症状关键词: {keyword}")
                        break
            
            # 如果都没有找到，给一个默认的无症状概率
            if has_symptom_prob == 0 and no_symptom_prob == 0:
                no_symptom_prob = 0.1
                no_symptom_matches["default_no_symptom"] = 0.1
                if debug:
                    logger.debug("未找到明确关键词，默认设为无症状")
            
            return has_symptom_prob, no_symptom_prob, has_symptom_matches, no_symptom_matches
        else:
            if debug:
                logger.debug("未找到<box>标签，返回默认无症状")
            return 0, 0.1, {}, {"default_no_symptom": 0.1}
    
    # 获取所有token的概率分布
    all_token_probs = logprobs_info['top_logprobs']
    tokens = logprobs_info.get('tokens', [])
    
    if debug:
        logger.debug(f"响应文本: {response_text}")
        logger.debug(f"Token数量: {len(tokens)}")
    
    # 寻找最后一个<box>标签的位置
    box_start_idx = None
    box_end_idx = None
    
    # 使用正则表达式提取<box>标签内的内容
    accumulated_text = "".join(tokens)
    box_pattern = r'<box>(.*?)</box>'
    box_matches = list(re.finditer(box_pattern, accumulated_text, re.DOTALL))
    
    if box_matches:
        # 使用最后一个匹配的<box>标签
        last_match = box_matches[-1]
        box_content = last_match.group(1)
        box_start_pos = last_match.start() + 5  # <box>的长度是5
        box_end_pos = last_match.end() - 6      # </box>的长度是6
        
        if debug:
            logger.debug(f"找到<box>标签内容: '{box_content}'")
            logger.debug(f"<box>标签在文本中的位置: {box_start_pos}-{box_end_pos}")
        
        # 在token序列中找到对应的位置
        current_pos = 0
        for i, token in enumerate(tokens):
            token_end_pos = current_pos + len(token)
            
            # 如果当前token的结束位置超过了box开始位置，记录开始索引
            if box_start_idx is None and token_end_pos > box_start_pos:
                box_start_idx = i
                if debug:
                    logger.debug(f"<box>内容开始token位置: {box_start_idx}")
            
            # 如果当前token的结束位置达到或超过了box结束位置，记录结束索引
            if box_end_idx is None and token_end_pos >= box_end_pos:
                box_end_idx = i + 1  # +1因为range是左闭右开
                if debug:
                    logger.debug(f"<box>内容结束token位置: {box_end_idx}")
                break
                
            current_pos = token_end_pos
    
    # 如果没有找到<box>标签，回退到分析前几个token
    if box_start_idx is None or box_end_idx is None:
        if debug:
            logger.debug("未找到<box>标签，分析前几个token")
        # 分析前5个token
        box_start_idx = 0
        box_end_idx = min(5, len(all_token_probs))
    
    if debug:
        logger.debug(f"分析Token范围: {box_start_idx} 到 {box_end_idx}")
    
    # 分析<box>标签内（或前几个token）的概率
    has_symptom_prob = 0
    no_symptom_prob = 0
    has_symptom_matches = {}
    no_symptom_matches = {}
    
    # 搜索关键词
    has_symptom_keywords = ["是","有", "存在", "yes", "Yes", "有症状", "有焦虑症状", "有抑郁症状"]
    no_symptom_keywords = ["否", "没有", "无", "不存在", "no", "No", "无症状", "没有症状", "无焦虑症状", "无抑郁症状"]
    
    # 首先尝试从原始响应文本中直接匹配（最优先和可靠的方案）
    box_pattern = r'<box>(.*?)</box>'
    original_box_matches = list(re.finditer(box_pattern, response_text, re.DOTALL))
    if original_box_matches:
        box_content = original_box_matches[-1].group(1).strip()
        if debug:
            logger.debug(f"从原始文本提取的焦虑症状<box>内容: '{box_content}'")
        
        # 直接检查box内容是否包含关键词
        for keyword in has_symptom_keywords:
            if keyword in box_content:
                has_symptom_prob += 0.9
                has_symptom_matches[f"direct_match_{keyword}"] = 0.9
                if debug:
                    logger.debug(f"在<box>内容中直接找到有症状关键词: {keyword}")
                break
        
        if has_symptom_prob == 0:  # 如果没有找到"有症状"关键词，检查"无症状"
            for keyword in no_symptom_keywords:
                if keyword in box_content:
                    no_symptom_prob += 0.9
                    no_symptom_matches[f"direct_match_{keyword}"] = 0.9
                    if debug:
                        logger.debug(f"在<box>内容中直接找到无症状关键词: {keyword}")
                    break
    
    # 如果直接匹配已经成功，优先返回结果（避免token级别分析的复杂性）
    if has_symptom_prob > 0 or no_symptom_prob > 0:
        if debug:
            logger.debug(f"直接匹配成功，有症状: {has_symptom_prob}, 无症状: {no_symptom_prob}")
            logger.debug(f"跳过token级别分析，直接返回结果")
        return has_symptom_prob, no_symptom_prob, has_symptom_matches, no_symptom_matches
    
    # 分析指定范围内的所有token
    for token_idx in range(box_start_idx, min(box_end_idx, len(all_token_probs))):
        if token_idx >= len(all_token_probs):
            break
            
        token_probs = all_token_probs[token_idx]
        current_token = tokens[token_idx] if token_idx < len(tokens) else ""
        
        if debug:
            logger.debug(f"分析Token {token_idx}: '{current_token}'")
        
        for token, logprob in token_probs.items():
            prob = math.exp(logprob)
            
            # 使用tokenizer解码token
            decoded_text = ""
            if tokenizer is not None:
                try:
                    # 方法1: 直接decode token string
                    try:
                        token_id = tokenizer.convert_tokens_to_ids(token)
                        if token_id != tokenizer.unk_token_id:
                            decoded_text = tokenizer.decode([token_id], skip_special_tokens=True)
                    except:
                        pass
                    
                    # 方法2: 如果方法1失败，尝试将token作为subword处理
                    if not decoded_text:
                        try:
                            clean_token = token
                            if token.startswith('Ġ'):
                                clean_token = token[1:]
                            
                            vocab = tokenizer.get_vocab()
                            if token in vocab:
                                token_id = vocab[token]
                                decoded_text = tokenizer.decode([token_id], skip_special_tokens=True)
                        except:
                            pass
                        
                except Exception as e:
                    if debug:
                        logger.debug(f"Token解码失败: {token}, 错误: {e}")
                    decoded_text = token
            else:
                decoded_text = token
            
            # 检查解码后的文本以及原始token
            texts_to_check = [decoded_text, token]
            
            # 移除空格前缀的变体
            if token.startswith('Ġ'):
                texts_to_check.append(token[1:])
            
            found_match = False
            for text in texts_to_check:
                if not text:
                    continue
                    
                # 检查是否是"有症状"相关token
                if any(keyword in text for keyword in has_symptom_keywords):
                    has_symptom_prob += prob
                    has_symptom_matches[f"{token_idx}_{token}"] = prob
                    if debug:
                        logger.debug(f"找到有症状token: {token} -> '{decoded_text}' (位置: {token_idx})")
                    found_match = True
                    break
                    
                # 检查是否是"无症状"相关token  
                if any(keyword in text for keyword in no_symptom_keywords):
                    no_symptom_prob += prob
                    no_symptom_matches[f"{token_idx}_{token}"] = prob
                    if debug:
                        logger.debug(f"找到无症状token: {token} -> '{decoded_text}' (位置: {token_idx})")
                    found_match = True
                    break
            
            if found_match:
                continue
    
    if debug:
        if has_symptom_matches:
            logger.debug(f"找到有症状相关token: {has_symptom_matches}")
        if no_symptom_matches:
            logger.debug(f"找到无症状相关token: {no_symptom_matches}")
        logger.debug(f"有症状概率总和: {has_symptom_prob}")
        logger.debug(f"无症状概率总和: {no_symptom_prob}")
    
    return has_symptom_prob, no_symptom_prob, has_symptom_matches, no_symptom_matches

# 从模型响应中提取多标签概率（抑郁症+焦虑症）
def extract_multilabel_probs(response, debug=False, ground_truth=None):
    if not response or 'choices' not in response or not response['choices']:
        return None, None, {}, {}
    
    # 获取完整的响应文本
    response_text = response['choices'][0].get('text', '')
    logprobs_info = response['choices'][0].get('logprobs', {})
    
    if not logprobs_info or 'top_logprobs' not in logprobs_info or not logprobs_info['top_logprobs']:
        if debug:
            logger.debug("没有logprobs信息，多标签检测返回None（暂未实现备用逻辑）")
        return None, None, {}, {}
    
    # 获取所有token的概率分布
    all_token_probs = logprobs_info['top_logprobs']
    tokens = logprobs_info.get('tokens', [])
    
    if debug:
        logger.debug(f"响应文本: {response_text}")
        logger.debug(f"Token数量: {len(tokens)}")
    
    # 寻找最后一个<box>标签的位置
    box_start_idx = None
    box_end_idx = None
    
    # 使用正则表达式提取<box>标签内的内容
    accumulated_text = "".join(tokens)
    box_pattern = r'<box>(.*?)</box>'
    box_matches = list(re.finditer(box_pattern, accumulated_text, re.DOTALL))
    
    if box_matches:
        # 使用最后一个匹配的<box>标签
        last_match = box_matches[-1]
        box_content = last_match.group(1)
        box_start_pos = last_match.start() + 5  # <box>的长度是5
        box_end_pos = last_match.end() - 6      # </box>的长度是6
        
        if debug:
            logger.debug(f"找到<box>标签内容: '{box_content}'")
            logger.debug(f"<box>标签在文本中的位置: {box_start_pos}-{box_end_pos}")
        
        # 在token序列中找到对应的位置
        current_pos = 0
        for i, token in enumerate(tokens):
            token_end_pos = current_pos + len(token)
            
            # 如果当前token的结束位置超过了box开始位置，记录开始索引
            if box_start_idx is None and token_end_pos > box_start_pos:
                box_start_idx = i
                if debug:
                    logger.debug(f"<box>内容开始token位置: {box_start_idx}")
            
            # 如果当前token的结束位置达到或超过了box结束位置，记录结束索引
            if box_end_idx is None and token_end_pos >= box_end_pos:
                box_end_idx = i + 1  # +1因为range是左闭右开
                if debug:
                    logger.debug(f"<box>内容结束token位置: {box_end_idx}")
                break
                
            current_pos = token_end_pos
    
    # 如果没有找到<box>标签，回退到分析前几个token
    if box_start_idx is None or box_end_idx is None:
        if debug:
            logger.debug("未找到<box>标签，分析前几个token")
        # 分析前10个token（多标签需要更多token）
        box_start_idx = 0
        box_end_idx = min(10, len(all_token_probs))
    
    if debug:
        logger.debug(f"分析Token范围: {box_start_idx} 到 {box_end_idx}")
    
    # 分析<box>标签内（或前几个token）的概率
    depression_yes_prob = 0
    depression_no_prob = 0
    anxiety_yes_prob = 0
    anxiety_no_prob = 0
    
    depression_yes_matches = {}
    depression_no_matches = {}
    anxiety_yes_matches = {}
    anxiety_no_matches = {}
    
    # 搜索关键词 - 更简单直接的方式
    yes_keywords = ["是", "有", "Yes", "yes", "存在"]
    no_keywords = ["否", "无", "No", "no", "不存在", "没有"]
    
    # 先找到完整的box内容来分析结构
    full_box_content = ""
    if box_matches:
        # 从原始响应文本中提取，而不是从token拼接的文本中
        original_box_matches = list(re.finditer(box_pattern, response_text, re.DOTALL))
        if original_box_matches:
            full_box_content = original_box_matches[-1].group(1).strip()
        else:
            full_box_content = box_matches[-1].group(1).strip()
    
    if debug:
        logger.debug(f"完整的box内容: '{full_box_content}'")
    
    # 分析box内容的结构，找到抑郁症和焦虑症的行
    depression_line = ""
    anxiety_line = ""
    
    for line in full_box_content.split('\n'):
        line = line.strip()
        if '抑郁' in line:
            depression_line = line
        elif '焦虑' in line:
            anxiety_line = line
    
    if debug:
        logger.debug(f"抑郁症行: '{depression_line}'")
        logger.debug(f"焦虑症行: '{anxiety_line}'")
    
    # 分析指定范围内的所有token
    for token_idx in range(box_start_idx, min(box_end_idx, len(all_token_probs))):
        if token_idx >= len(all_token_probs):
            break
            
        token_probs = all_token_probs[token_idx]
        current_token = tokens[token_idx] if token_idx < len(tokens) else ""
        
        if debug:
            logger.debug(f"分析Token {token_idx}: '{current_token}'")
        
        for token, logprob in token_probs.items():
            prob = math.exp(logprob)
            
            # 使用tokenizer解码token
            decoded_text = ""
            if tokenizer is not None:
                try:
                    # 方法1: 直接decode token string
                    try:
                        token_id = tokenizer.convert_tokens_to_ids(token)
                        if token_id != tokenizer.unk_token_id:
                            decoded_text = tokenizer.decode([token_id], skip_special_tokens=True)
                    except:
                        pass
                    
                    # 方法2: 如果方法1失败，尝试将token作为subword处理
                    if not decoded_text:
                        try:
                            clean_token = token
                            if token.startswith('Ġ'):
                                clean_token = token[1:]
                            
                            vocab = tokenizer.get_vocab()
                            if token in vocab:
                                token_id = vocab[token]
                                decoded_text = tokenizer.decode([token_id], skip_special_tokens=True)
                        except:
                            pass
                        
                except Exception as e:
                    if debug:
                        logger.debug(f"Token解码失败: {token}, 错误: {e}")
                    decoded_text = token
            else:
                decoded_text = token
            
            # 检查解码后的文本以及原始token
            texts_to_check = [decoded_text, token]
            
            # 移除空格前缀的变体
            if token.startswith('Ġ'):
                texts_to_check.append(token[1:])
            
            # 检查前面的上下文来确定是抑郁症还是焦虑症相关
            context_before = "".join(tokens[max(0, token_idx-5):token_idx])
            context_after = "".join(tokens[token_idx:min(len(tokens), token_idx+3)])
            full_context = context_before + context_after
            
            found_match = False
            for text in texts_to_check:
                if not text:
                    continue
                    
                # 如果找到"是"相关的token
                if any(keyword == text.strip() for keyword in yes_keywords):
                    # 判断是抑郁症还是焦虑症
                    if "抑郁" in context_before or "抑郁" in full_context:
                        depression_yes_prob += prob
                        depression_yes_matches[f"{token_idx}_{token}"] = prob
                        if debug:
                            logger.debug(f"找到抑郁症-是token: {token} -> '{decoded_text}' (位置: {token_idx}, 上下文: {context_before})")
                        found_match = True
                        break
                    elif "焦虑" in context_before or "焦虑" in full_context:
                        anxiety_yes_prob += prob
                        anxiety_yes_matches[f"{token_idx}_{token}"] = prob
                        if debug:
                            logger.debug(f"找到焦虑症-是token: {token} -> '{decoded_text}' (位置: {token_idx}, 上下文: {context_before})")
                        found_match = True
                        break
                
                # 如果找到"否"相关的token
                elif any(keyword == text.strip() for keyword in no_keywords):
                    # 判断是抑郁症还是焦虑症
                    if "抑郁" in context_before or "抑郁" in full_context:
                        depression_no_prob += prob
                        depression_no_matches[f"{token_idx}_{token}"] = prob
                        if debug:
                            logger.debug(f"找到抑郁症-否token: {token} -> '{decoded_text}' (位置: {token_idx}, 上下文: {context_before})")
                        found_match = True
                        break
                    elif "焦虑" in context_before or "焦虑" in full_context:
                        anxiety_no_prob += prob
                        anxiety_no_matches[f"{token_idx}_{token}"] = prob
                        if debug:
                            logger.debug(f"找到焦虑症-否token: {token} -> '{decoded_text}' (位置: {token_idx}, 上下文: {context_before})")
                        found_match = True
                        break
            
            if found_match:
                continue
    
    # 计算每个标签的概率
    depression_total = depression_yes_prob + depression_no_prob
    anxiety_total = anxiety_yes_prob + anxiety_no_prob
    
    depression_prob = depression_yes_prob / depression_total if depression_total > 0 else 0
    anxiety_prob = anxiety_yes_prob / anxiety_total if anxiety_total > 0 else 0
    
    # 如果token概率分析没有找到任何匹配，回退到文本分析
    if depression_total == 0 and anxiety_total == 0:
        if debug:
            logger.debug("Token概率分析失败，回退到文本分析")
        
        # 直接分析box内容
        if full_box_content:
            # 解析抑郁症
            if depression_line:
                if any(keyword in depression_line for keyword in yes_keywords):
                    depression_prob = 0.8  # 给一个较高的概率
                elif any(keyword in depression_line for keyword in no_keywords):
                    depression_prob = 0.2  # 给一个较低的概率
            
            # 解析焦虑症
            if anxiety_line:
                if any(keyword in anxiety_line for keyword in yes_keywords):
                    anxiety_prob = 0.8  # 给一个较高的概率
                elif any(keyword in anxiety_line for keyword in no_keywords):
                    anxiety_prob = 0.2  # 给一个较低的概率
            
            if debug:
                logger.debug(f"文本分析结果 - 抑郁症概率: {depression_prob}, 焦虑症概率: {anxiety_prob}")
    
    # 返回两个标签的概率和匹配的tokens
    all_matches = {
        "depression_yes": depression_yes_matches,
        "depression_no": depression_no_matches,
        "anxiety_yes": anxiety_yes_matches,
        "anxiety_no": anxiety_no_matches
    }
    
    if debug:
        logger.debug(f"最终抑郁症概率: {depression_prob}")
        logger.debug(f"最终焦虑症概率: {anxiety_prob}")
        logger.debug(f"匹配的tokens: {all_matches}")
    
    return depression_prob, anxiety_prob, all_matches, {}

# 从模型响应中提取ICD-10诊断代码概率
def extract_icd10_code(response, debug=False):
    """从响应中提取ICD-10诊断代码"""
    if not response or 'choices' not in response or not response['choices']:
        return None
    
    # 获取完整的响应文本
    response_text = response['choices'][0].get('text', '')
    
    if debug:
        logger.debug(f"ICD10响应文本: {response_text}")
    
    # 提取<box>标签内的内容
    import re
    box_pattern = r'<box>(.*?)</box>'
    box_matches = list(re.finditer(box_pattern, response_text, re.DOTALL))
    
    if not box_matches:
        if debug:
            logger.debug("未找到<box>标签")
        return None
    
    # 使用最后一个匹配的<box>标签
    predicted_text = box_matches[-1].group(1).strip()
    
    if debug:
        logger.debug(f"提取的ICD-10代码: '{predicted_text}'")
    
    # 标准化ICD-10代码格式
    # 匹配F开头的代码和Z71代码，如F32, F32.1, F41.0, Z71.9等
    icd10_pattern = r'(?:F\d+(?:\.\d+)?|Z71(?:\.\d+)?)'
    matches = re.findall(icd10_pattern, predicted_text)
    
    if matches:
        predicted_code = matches[0]  # 取第一个匹配的代码
        if debug:
            logger.debug(f"标准化后的ICD-10代码: {predicted_code}")
        return predicted_code
    else:
        if debug:
            logger.debug("未找到有效的ICD-10代码格式")
        return predicted_text  # 返回原始文本以便调试

def preprocess_diagnosis_codes(diagnosis_code):
    """
    预处理DiagnosisCode列的数据，支持多个ICD代码
    
    Args:
        diagnosis_code: 原始的诊断代码，可能包含多个代码
        
    Returns:
        list: 清理后的ICD代码列表
    """
    if pd.isna(diagnosis_code) or diagnosis_code is None:
        return []
    
    # 转换为字符串并清理
    code_str = str(diagnosis_code).strip()
    
    if not code_str:
        return []
    
    # 分割多个代码（可能用逗号、分号、空格等分隔）
    import re
    # 使用正则表达式匹配F开头的ICD代码和Z71代码
    codes = re.findall(r'(?:F\d+(?:\.\d+)?|Z71(?:\.\d+)?)', code_str, re.IGNORECASE)
    
    # 标准化代码格式（统一为大写）
    standardized_codes = [code.upper() for code in codes if code]
    
    return standardized_codes

def check_icd_code_match(predicted_code, ground_truth_codes):
    """
    检查预测的ICD代码是否与ground truth中的任一代码匹配
    
    Args:
        predicted_code: 预测的ICD代码
        ground_truth_codes: ground truth的ICD代码列表
        
    Returns:
        bool: 是否匹配
    """
    if not predicted_code or not ground_truth_codes:
        return False
    
    # 标准化预测代码
    pred_code = str(predicted_code).strip().upper()
    
    # 检查是否与任一ground truth代码匹配
    for gt_code in ground_truth_codes:
        gt_code = str(gt_code).strip().upper()
        if pred_code == gt_code:
            return True
    
    return False

def extract_icd10_probs(response, debug=False, ground_truth=None):
    """保持兼容性的包装函数"""
    if not response or 'choices' not in response or not response['choices']:
        return None, None, None, None, {}, {}, {}, {}
    
    # 获取完整的响应文本
    response_text = response['choices'][0].get('text', '')
    logprobs_info = response['choices'][0].get('logprobs', {})
    
    # 对于ICD-10模式，我们不再使用概率，而是直接提取诊断代码
    if debug:
        logger.debug("ICD10模式：直接提取诊断代码，不计算概率")
    
    # 直接提取诊断代码
    predicted_code = extract_icd10_code(response, debug)
    
    # 返回兼容格式，但实际不使用概率
    return predicted_code, None, None, None, {}, {}, {}, {}

def extract_recommendation_codes(response, debug=False, ground_truth=None):
    """从响应中提取推荐的多个ICD-10大类代码（支持分号分隔格式）"""
    if not response or 'choices' not in response or not response['choices']:
        return []
    
    # 获取完整的响应文本
    response_text = response['choices'][0].get('text', '')
    
    if debug:
        logger.debug(f"Recommendation响应文本: {response_text}")
    
    # 提取<box>标签内的内容
    import re
    box_pattern = r'<box>(.*?)</box>'
    box_matches = list(re.finditer(box_pattern, response_text, re.DOTALL))
    
    if not box_matches:
        if debug:
            logger.debug("未找到<box>标签")
        return []
    
    # 使用最后一个匹配的<box>标签
    predicted_text = box_matches[-1].group(1).strip()
    
    if debug:
        logger.debug(f"提取的推荐代码文本: '{predicted_text}'")
    
    # 处理分号分隔的格式
    recommended_codes = []
    
    # 首先尝试分号分隔的格式
    if ';' in predicted_text:
        # 按分号分割
        parts = predicted_text.split(';')
        for part in parts:
            part = part.strip()
            # 从每个部分提取ICD-10代码
            icd10_matches = re.findall(r'F\d+', part)
            recommended_codes.extend(icd10_matches)
    else:
        # 如果没有分号，尝试其他分隔符或按行分割
        # 尝试逗号分隔
        if ',' in predicted_text:
            parts = predicted_text.split(',')
            for part in parts:
                part = part.strip()
                icd10_matches = re.findall(r'F\d+', part)
                recommended_codes.extend(icd10_matches)
        else:
            # 按行分割（兼容旧格式）
            lines = predicted_text.split('\n')
            for line in lines:
                line = line.strip()
                if line:
                    icd10_matches = re.findall(r'F\d+', line)
                    recommended_codes.extend(icd10_matches)
    
    # 如果上述方法都没有找到代码，直接在整个文本中搜索
    if not recommended_codes:
        icd10_matches = re.findall(r'F\d+', predicted_text)
        recommended_codes.extend(icd10_matches)
    
    # 去重并保持顺序
    final_codes = []
    seen = set()
    for code in recommended_codes:
        if code not in seen:
            final_codes.append(code)
            seen.add(code)
    
    if debug:
        logger.debug(f"提取的推荐代码列表: {final_codes}")
        if ';' in predicted_text:
            logger.debug("使用分号分隔格式解析")
        elif ',' in predicted_text:
            logger.debug("使用逗号分隔格式解析")
        else:
            logger.debug("使用行分隔格式解析")
    
    return final_codes

# 生成模型完整响应文本
def get_complete_response(response):
    if not response or 'choices' not in response or not response['choices']:
        return "无法获取响应"
    
    return response['choices'][0].get('text', "无响应文本")

# 处理单个记录的异步函数
async def process_record_async(client, semaphore, visit_number, text, model_name, classification_mode="binary", primary_diagnosis=None, debug=False, save_full_response=True, top_logprobs=20, site_url="", site_name="", save_json_logs=True, api_type="openrouter"):
    """异步处理单个记录"""
    # 跳过空文本
    if pd.isna(text) or text.strip() == "":
        logger.warning(f"警告: VisitNumber {visit_number} 的文本为空，已跳过")
        
        if classification_mode == "binary":
            result = {
                "VisitNumber": visit_number,
                "OverallDiagnosis": primary_diagnosis,
                "Depression_Probability": None,
                "Anxiety_Probability": None,
                "Total_Probability": None,
                "Normalized_Depression_Prob": None,
                "Normalized_Anxiety_Prob": None,
            }
            
            if save_full_response:
                result.update({
                    "Model_Response": None,
                    "Depression_Tokens": None,
                    "Anxiety_Tokens": None
                })
        elif classification_mode == "multiclass":
            result = {
                "VisitNumber": visit_number,
                "OverallDiagnosis": primary_diagnosis,
                "Depression_Probability": None,
                "Anxiety_Probability": None,
                "Mix_Probability": None,
                "Others_Probability": None,
                "Total_Probability": None,
                "Normalized_Depression_Prob": None,
                "Normalized_Anxiety_Prob": None,
                "Normalized_Mix_Prob": None,
                "Normalized_Others_Prob": None,
            }
            
            if save_full_response:
                result.update({
                    "Model_Response": None,
                    "Depression_Tokens": None,
                    "Anxiety_Tokens": None,
                    "Mix_Tokens": None,
                    "Others_Tokens": None
                })
        elif classification_mode == "depression_symptom":
            result = {
                "VisitNumber": visit_number,
                "DepressionDiagnosis": primary_diagnosis,
                "Has_Depression_Symptom_Probability": None,
                "No_Depression_Symptom_Probability": None,
                "Total_Probability": None,
                "Normalized_Has_Depression_Symptom_Prob": None,
                "Normalized_No_Depression_Symptom_Prob": None,
            }
            
            if save_full_response:
                result.update({
                    "Model_Response": None,
                    "Has_Depression_Symptom_Tokens": None,
                    "No_Depression_Symptom_Tokens": None
                })
        elif classification_mode == "anxiety_symptom":
            result = {
                "VisitNumber": visit_number,
                "AnxietyDiagnosis": primary_diagnosis,
                "Has_Anxiety_Symptom_Probability": None,
                "No_Anxiety_Symptom_Probability": None,
                "Total_Probability": None,
                "Normalized_Has_Anxiety_Symptom_Prob": None,
                "Normalized_No_Anxiety_Symptom_Prob": None,
            }
            
            if save_full_response:
                result.update({
                    "Model_Response": None,
                    "Has_Anxiety_Symptom_Tokens": None,
                    "No_Anxiety_Symptom_Tokens": None
                })
        elif classification_mode == "multilabel":
            result = {
                "VisitNumber": visit_number,
                "MultilabelDiagnosis": primary_diagnosis,
                "Depression_Probability": None,
                "Anxiety_Probability": None,
                "Depression_Label": None,
                "Anxiety_Label": None,
                "Combined_Label": None,
            }
            
            if save_full_response:
                result.update({
                    "Model_Response": None,
                    "Multilabel_Tokens": None
                })
        elif classification_mode == "icd10":
            result = {
                "VisitNumber": visit_number,
                "OverallDiagnosis": primary_diagnosis,
                "F32_Probability": None,
                "F41_Probability": None,
                "F43_Probability": None,
                "F60_69_Probability": None,
                "Total_Probability": None,
                "Normalized_F32_Prob": None,
                "Normalized_F41_Prob": None,
                "Normalized_F43_Prob": None,
                "Normalized_F60_69_Prob": None,
                "Predicted_ICD10": None,
            }
            
            if save_full_response:
                result.update({
                    "Model_Response": None,
                    "F32_Tokens": None,
                    "F41_Tokens": None,
                    "F43_Tokens": None,
                    "F60_69_Tokens": None
                })
        elif classification_mode == "recommendation":
            result = {
                "VisitNumber": visit_number,
                "Ground_Truth_ICD10": primary_diagnosis,
                "Recommended_ICD10_Codes": None,
                "Top1_Code": None,
                "Top3_Codes": None,
                "Num_Recommended": None,
            }
            
            if save_full_response:
                result.update({
                    "Model_Response": None,
                })
        
        return result
    
    # 创建prompt
    messages = create_prompt(text, classification_mode)
    
    # 获取token概率
    response = await get_token_probabilities_async(client, semaphore, messages, model_name, top_logprobs=top_logprobs, site_url=site_url, site_name=site_name, api_type=api_type)
    
    # 记录请求和响应到JSON日志
    await log_request_response(visit_number, messages, response, model_name, classification_mode, primary_diagnosis, save_json_logs)
    
    # 提取概率
    if classification_mode == "binary":
        depression_prob, anxiety_prob, depression_tokens, anxiety_tokens = extract_classification_probs(response, classification_mode, debug, primary_diagnosis)
        
        # 计算总概率和归一化概率
        total_prob = 0
        norm_depression_prob = None
        norm_anxiety_prob = None
        
        if depression_prob is not None and anxiety_prob is not None:
            total_prob = depression_prob + anxiety_prob
            if total_prob > 0:
                norm_depression_prob = depression_prob / total_prob
                norm_anxiety_prob = anxiety_prob / total_prob
        
        result = {
            "VisitNumber": visit_number,
            "OverallDiagnosis": primary_diagnosis,
            "Depression_Probability": depression_prob,
            "Anxiety_Probability": anxiety_prob,
            "Total_Probability": total_prob,
            "Normalized_Depression_Prob": norm_depression_prob,
            "Normalized_Anxiety_Prob": norm_anxiety_prob,
        }
        
        if save_full_response:
            result.update({
                "Model_Response": get_complete_response(response),
                "Depression_Tokens": str(depression_tokens),
                "Anxiety_Tokens": str(anxiety_tokens)
            })
            
        if debug:
            logger.debug(f"VisitNumber {visit_number} - 抑郁概率: {depression_prob}, 焦虑概率: {anxiety_prob}")
    
    elif classification_mode == "multiclass":
        result = extract_classification_probs(response, classification_mode, debug, primary_diagnosis)
        if result is None or len(result) < 8:
            depression_prob, anxiety_prob, mix_prob, others_prob = None, None, None, None
            depression_tokens, anxiety_tokens, mix_tokens, others_tokens = {}, {}, {}, {}
        else:
            depression_prob, anxiety_prob, mix_prob, others_prob, depression_tokens, anxiety_tokens, mix_tokens, others_tokens = result
        
        # 计算总概率和归一化概率
        total_prob = 0
        norm_depression_prob = None
        norm_anxiety_prob = None
        norm_mix_prob = None
        norm_others_prob = None
        
        if all(prob is not None for prob in [depression_prob, anxiety_prob, mix_prob, others_prob]):
            total_prob = (depression_prob or 0) + (anxiety_prob or 0) + (mix_prob or 0) + (others_prob or 0)
            if total_prob > 0:
                norm_depression_prob = depression_prob / total_prob
                norm_anxiety_prob = anxiety_prob / total_prob
                norm_mix_prob = mix_prob / total_prob
                norm_others_prob = others_prob / total_prob
        else:
            # 如果有任何概率为None，则设置total_prob为0
            total_prob = 0
        
        result = {
            "VisitNumber": visit_number,
            "OverallDiagnosis": primary_diagnosis,
            "Depression_Probability": depression_prob,
            "Anxiety_Probability": anxiety_prob,
            "Mix_Probability": mix_prob,
            "Others_Probability": others_prob,
            "Total_Probability": total_prob,
            "Normalized_Depression_Prob": norm_depression_prob,
            "Normalized_Anxiety_Prob": norm_anxiety_prob,
            "Normalized_Mix_Prob": norm_mix_prob,
            "Normalized_Others_Prob": norm_others_prob,
        }
        
        if save_full_response:
            result.update({
                "Model_Response": get_complete_response(response),
                "Depression_Tokens": str(depression_tokens),
                "Anxiety_Tokens": str(anxiety_tokens),
                "Mix_Tokens": str(mix_tokens),
                "Others_Tokens": str(others_tokens)
            })
            
        if debug:
            logger.debug(f"VisitNumber {visit_number} - 抑郁: {depression_prob}, 焦虑: {anxiety_prob}, mix: {mix_prob}, others: {others_prob}")
    
    elif classification_mode == "depression_symptom":
        has_symptom_prob, no_symptom_prob, has_symptom_tokens, no_symptom_tokens = extract_classification_probs(response, classification_mode, debug, primary_diagnosis)
        
        # 计算总概率和归一化概率
        total_prob = 0
        norm_has_symptom_prob = None
        norm_no_symptom_prob = None
        
        if has_symptom_prob is not None and no_symptom_prob is not None:
            total_prob = has_symptom_prob + no_symptom_prob
            if total_prob > 0:
                norm_has_symptom_prob = has_symptom_prob / total_prob
                norm_no_symptom_prob = no_symptom_prob / total_prob
        
        result = {
            "VisitNumber": visit_number,
            "DepressionDiagnosis": primary_diagnosis,  # 这里应该是DepressionDiagnosis的ground truth
            "Has_Depression_Symptom_Probability": has_symptom_prob,
            "No_Depression_Symptom_Probability": no_symptom_prob,
            "Total_Probability": total_prob,
            "Normalized_Has_Depression_Symptom_Prob": norm_has_symptom_prob,
            "Normalized_No_Depression_Symptom_Prob": norm_no_symptom_prob,
        }
        
        if save_full_response:
            result.update({
                "Model_Response": get_complete_response(response),
                "Has_Depression_Symptom_Tokens": str(has_symptom_tokens),
                "No_Depression_Symptom_Tokens": str(no_symptom_tokens)
            })
            
        if debug:
            logger.debug(f"VisitNumber {visit_number} - 有抑郁症状: {has_symptom_prob}, 无抑郁症状: {no_symptom_prob}")
    
    elif classification_mode == "anxiety_symptom":
        has_symptom_prob, no_symptom_prob, has_symptom_tokens, no_symptom_tokens = extract_classification_probs(response, classification_mode, debug, primary_diagnosis)
        
        # 计算总概率和归一化概率
        total_prob = 0
        norm_has_symptom_prob = None
        norm_no_symptom_prob = None
        
        if has_symptom_prob is not None and no_symptom_prob is not None:
            total_prob = has_symptom_prob + no_symptom_prob
            if total_prob > 0:
                norm_has_symptom_prob = has_symptom_prob / total_prob
                norm_no_symptom_prob = no_symptom_prob / total_prob
        
        result = {
            "VisitNumber": visit_number,
            "AnxietyDiagnosis": primary_diagnosis,  # 这里应该是AnxietyDiagnosis的ground truth
            "Has_Anxiety_Symptom_Probability": has_symptom_prob,
            "No_Anxiety_Symptom_Probability": no_symptom_prob,
            "Total_Probability": total_prob,
            "Normalized_Has_Anxiety_Symptom_Prob": norm_has_symptom_prob,
            "Normalized_No_Anxiety_Symptom_Prob": norm_no_symptom_prob,
        }
        
        if save_full_response:
            result.update({
                "Model_Response": get_complete_response(response),
                "Has_Anxiety_Symptom_Tokens": str(has_symptom_tokens),
                "No_Anxiety_Symptom_Tokens": str(no_symptom_tokens)
            })
            
        if debug:
            logger.debug(f"VisitNumber {visit_number} - 有焦虑症状: {has_symptom_prob}, 无焦虑症状: {no_symptom_prob}")
    
    elif classification_mode == "multilabel":
        result = extract_classification_probs(response, classification_mode, debug, primary_diagnosis)
        if result is None or result[0] is None:
            depression_prob, anxiety_prob, multilabel_tokens = None, None, {}
        else:
            depression_prob, anxiety_prob, multilabel_tokens, _ = result
        
        # 计算标签（二进制）
        depression_label = 1 if depression_prob and depression_prob > 0.5 else 0
        anxiety_label = 1 if anxiety_prob and anxiety_prob > 0.5 else 0
        
        # 创建组合标签：[depression_label, anxiety_label]
        combined_label = [depression_label, anxiety_label]
        
        result = {
            "VisitNumber": visit_number,
            "MultilabelDiagnosis": primary_diagnosis,
            "Depression_Probability": depression_prob,
            "Anxiety_Probability": anxiety_prob,
            "Depression_Label": depression_label,
            "Anxiety_Label": anxiety_label,
            "Combined_Label": str(combined_label),
        }
        
        if save_full_response:
            result.update({
                "Model_Response": get_complete_response(response),
                "Multilabel_Tokens": str(multilabel_tokens)
            })
            
        if debug:
            logger.debug(f"VisitNumber {visit_number} - 抑郁概率: {depression_prob}, 焦虑概率: {anxiety_prob}, 标签: {combined_label}")
    
    elif classification_mode == "icd10":
        # 直接提取ICD-10诊断代码，不计算概率
        predicted_icd10 = extract_icd10_code(response, debug)
        
        # 提取大类和小类
        predicted_major_class = None
        predicted_minor_class = None
        
        if predicted_icd10:
            # 提取大类（如F32）和小类（如F32.1）
            import re
            major_match = re.match(r'(F\d+)', predicted_icd10)
            if major_match:
                predicted_major_class = major_match.group(1)
                predicted_minor_class = predicted_icd10  # 完整代码作为小类
        
        result = {
            "VisitNumber": visit_number,
            "Ground_Truth_ICD10": primary_diagnosis if not isinstance(primary_diagnosis, list) else primary_diagnosis,
            "Predicted_ICD10": predicted_icd10,
            "Predicted_Major_Class": predicted_major_class,
            "Predicted_Minor_Class": predicted_minor_class,
        }
        
        if save_full_response:
            result.update({
                "Model_Response": get_complete_response(response),
            })
            
        if debug:
            logger.debug(f"VisitNumber {visit_number} - 预测ICD10: {predicted_icd10}, 大类: {predicted_major_class}, 小类: {predicted_minor_class}, Ground Truth: {primary_diagnosis}")
    
    elif classification_mode == "recommendation":
        # 提取推荐的ICD-10代码列表
        recommended_codes = extract_classification_probs(response, classification_mode, debug, primary_diagnosis)
        
        # 处理推荐结果
        if recommended_codes and isinstance(recommended_codes, list):
            top1_code = recommended_codes[0] if len(recommended_codes) > 0 else None
            top3_codes = recommended_codes[:3] if len(recommended_codes) >= 3 else recommended_codes
            num_recommended = len(recommended_codes)
        else:
            top1_code = None
            top3_codes = []
            num_recommended = 0
            recommended_codes = []
        
        result = {
            "VisitNumber": visit_number,
            "Ground_Truth_ICD10": primary_diagnosis if not isinstance(primary_diagnosis, list) else primary_diagnosis,
            "Recommended_ICD10_Codes": str(recommended_codes),
            "Top1_Code": top1_code,
            "Top3_Codes": str(top3_codes),
            "Num_Recommended": num_recommended,
        }
        
        if save_full_response:
            result.update({
                "Model_Response": get_complete_response(response),
            })
            
        if debug:
            logger.debug(f"VisitNumber {visit_number} - 推荐代码: {recommended_codes}, Top1: {top1_code}, Top3: {top3_codes}, Ground Truth: {primary_diagnosis}")
            
            # 计算样本级别的metrics（限制为11种疾病）
            if primary_diagnosis is not None and recommended_codes:
                # 使用统一的ICD-10工具类
                
                # 提取推荐代码的大类
                recommended_majors = []
                if top1_code:
                    major = ICD10Utils.extract_major_class(top1_code)
                    if major:
                        recommended_majors.append(major)
                
                if top3_codes:
                    for code in top3_codes:
                        major = ICD10Utils.extract_major_class(code)
                        if major and major not in recommended_majors:
                            recommended_majors.append(major)
                
                # 提取ground truth的大类并去重
                gt_majors = set()
                if isinstance(primary_diagnosis, list):
                    for code in primary_diagnosis:
                        major = ICD10Utils.extract_major_class(code)
                        if major:
                            gt_majors.add(major)
                else:
                    major = ICD10Utils.extract_major_class(primary_diagnosis)
                    if major:
                        gt_majors.add(major)
                
                # 检查Top1大类匹配
                top1_major_match = False
                if top1_code:
                    top1_major = ICD10Utils.extract_major_class(top1_code)
                    top1_major_match = top1_major in gt_majors if top1_major else False
                
                # 检查Top3大类匹配
                top3_major_match = False
                if top3_codes:
                    for code in top3_codes:
                        major = ICD10Utils.extract_major_class(code)
                        if major and major in gt_majors:
                            top3_major_match = True
                            break
                
                logger.debug(f"VisitNumber {visit_number} - 样本Metrics（限制11种疾病）: Top1大类匹配={top1_major_match}, Top3大类匹配={top3_major_match}, 推荐大类={recommended_majors}, GT大类={list(gt_majors)}")
            
    return result

# 可视化结果
def visualize_results(results_df, output_prefix, classification_mode="binary", results_dir="./results"):
    """生成结果可视化图表"""
    try:
        # 确保输出目录存在
        visualization_dir = os.path.join(results_dir, "visualization")
        os.makedirs(visualization_dir, exist_ok=True)
        
        # 设置Seaborn风格
        sns.set(style="whitegrid")
        
        # 检查各分类模式下的额外列是否存在
        if classification_mode == "multiclass":
            required_cols = ["Mix_Probability", "Others_Probability", "Normalized_Mix_Prob", "Normalized_Others_Prob"]
            missing_cols = [col for col in required_cols if col not in results_df.columns]
            if missing_cols:
                logger.warning(f"四分类模式下缺少必要的列: {missing_cols}，将回退到二分类可视化")
                classification_mode = "binary"
        elif classification_mode == "depression_symptom":
            required_cols = ["Has_Depression_Symptom_Probability", "No_Depression_Symptom_Probability", 
                           "Normalized_Has_Depression_Symptom_Prob", "Normalized_No_Depression_Symptom_Prob"]
            missing_cols = [col for col in required_cols if col not in results_df.columns]
            if missing_cols:
                logger.warning(f"抑郁症状检测模式下缺少必要的列: {missing_cols}，跳过可视化")
                return
        elif classification_mode == "anxiety_symptom":
            required_cols = ["Has_Anxiety_Symptom_Probability", "No_Anxiety_Symptom_Probability", 
                           "Normalized_Has_Anxiety_Symptom_Prob", "Normalized_No_Anxiety_Symptom_Prob"]
            missing_cols = [col for col in required_cols if col not in results_df.columns]
            if missing_cols:
                logger.warning(f"焦虑症状检测模式下缺少必要的列: {missing_cols}，跳过可视化")
                return
        
        if classification_mode == "binary":
            # 二分类可视化
            logger.info("开始生成二分类可视化图表...")
            try:
                # 1. 抑郁症和焦虑症概率分布直方图
                plt.figure(figsize=(12, 10))
                
                plt.subplot(2, 2, 1)
                sns.histplot(results_df["Depression_Probability"].dropna(), kde=True, color="blue")
                plt.title("Depression Probability Distribution")
                plt.xlabel("Probability")
                plt.ylabel("Frequency")
                
                plt.subplot(2, 2, 2)
                sns.histplot(results_df["Anxiety_Probability"].dropna(), kde=True, color="red")
                plt.title("Anxiety Probability Distribution")
                plt.xlabel("Probability")
                plt.ylabel("Frequency")
                
                plt.subplot(2, 2, 3)
                sns.histplot(results_df["Normalized_Depression_Prob"].dropna(), kde=True, color="darkblue")
                plt.title("Normalized Depression Probability Distribution")
                plt.xlabel("Probability")
                plt.ylabel("Frequency")
                
                plt.subplot(2, 2, 4)
                sns.histplot(results_df["Normalized_Anxiety_Prob"].dropna(), kde=True, color="darkred")
                plt.title("Normalized Anxiety Probability Distribution")
                plt.xlabel("Probability")
                plt.ylabel("Frequency")
                
                plt.tight_layout()
                plt.savefig(f"{visualization_dir}/{output_prefix}_probability_distributions.png")
                plt.close()  # 释放图表资源
                logger.info("概率分布图已保存")
            except Exception as e:
                logger.error(f"生成概率分布图时出错: {e}")
                plt.close()  # 确保释放资源
            
            try:
                # 2. 抑郁症vs焦虑症概率散点图
                plt.figure(figsize=(10, 8))
                sns.scatterplot(
                    x="Depression_Probability", 
                    y="Anxiety_Probability", 
                    data=results_df.dropna(),
                    alpha=0.6
                )
                plt.title("Depression vs Anxiety Probability Distribution")
                plt.xlabel("Depression Probability")
                plt.ylabel("Anxiety Probability")
                # 添加45度对角线
                max_val = max(
                    results_df["Depression_Probability"].max() or 0, 
                    results_df["Anxiety_Probability"].max() or 0
                )
                plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(f"{visualization_dir}/{output_prefix}_depression_vs_anxiety.png")
                plt.close()  # 释放图表资源
                logger.info("散点图已保存")
            except Exception as e:
                logger.error(f"生成散点图时出错: {e}")
                plt.close()  # 确保释放资源
            
            try:
                # 3. 归一化概率的饼图 - 总体分布
                plt.figure(figsize=(10, 8))
                avg_depression = results_df["Normalized_Depression_Prob"].mean()
                avg_anxiety = results_df["Normalized_Anxiety_Prob"].mean()
                
                if not (math.isnan(avg_depression) or math.isnan(avg_anxiety)):
                    plt.pie(
                        [avg_depression, avg_anxiety], 
                        labels=["Depression", "Anxiety"], 
                        autopct='%1.1f%%', 
                        colors=["blue", "red"],
                        startangle=90
                    )
                    plt.title("Overall Depression/Anxiety Tendency Distribution")
                    plt.savefig(f"{visualization_dir}/{output_prefix}_overall_distribution.png")
                    logger.info("饼图已保存")
                else:
                    logger.warning("无法生成饼图：数据包含NaN值")
                plt.close()  # 释放图表资源
            except Exception as e:
                logger.error(f"生成饼图时出错: {e}")
                plt.close()  # 确保释放资源
                
        elif classification_mode == "multiclass":  # multiclass
            # 四分类可视化
            logger.info("开始生成四分类可视化图表...")
            try:
                # 1. 所有四个类别的概率分布直方图
                plt.figure(figsize=(16, 12))
                
                plt.subplot(2, 4, 1)
                sns.histplot(results_df["Depression_Probability"].dropna(), kde=True, color="blue")
                plt.title("Depression Probability Distribution")
                plt.xlabel("Probability")
                plt.ylabel("Frequency")
                
                plt.subplot(2, 4, 2)
                sns.histplot(results_df["Anxiety_Probability"].dropna(), kde=True, color="red")
                plt.title("Anxiety Probability Distribution")
                plt.xlabel("Probability")
                plt.ylabel("Frequency")
                
                plt.subplot(2, 4, 3)
                sns.histplot(results_df["Mix_Probability"].dropna(), kde=True, color="green")
                plt.title("Mix Probability Distribution")
                plt.xlabel("Probability")
                plt.ylabel("Frequency")
                
                plt.subplot(2, 4, 4)
                sns.histplot(results_df["Others_Probability"].dropna(), kde=True, color="orange")
                plt.title("Others Probability Distribution")
                plt.xlabel("Probability")
                plt.ylabel("Frequency")
                
                plt.subplot(2, 4, 5)
                sns.histplot(results_df["Normalized_Depression_Prob"].dropna(), kde=True, color="darkblue")
                plt.title("Normalized Depression Prob")
                plt.xlabel("Probability")
                plt.ylabel("Frequency")
                
                plt.subplot(2, 4, 6)
                sns.histplot(results_df["Normalized_Anxiety_Prob"].dropna(), kde=True, color="darkred")
                plt.title("Normalized Anxiety Prob")
                plt.xlabel("Probability")
                plt.ylabel("Frequency")
                
                plt.subplot(2, 4, 7)
                sns.histplot(results_df["Normalized_Mix_Prob"].dropna(), kde=True, color="darkgreen")
                plt.title("Normalized Mix Prob")
                plt.xlabel("Probability")
                plt.ylabel("Frequency")
                
                plt.subplot(2, 4, 8)
                sns.histplot(results_df["Normalized_Others_Prob"].dropna(), kde=True, color="darkorange")
                plt.title("Normalized Others Prob")
                plt.xlabel("Probability")
                plt.ylabel("Frequency")
                
                plt.tight_layout()
                plt.savefig(f"{visualization_dir}/{output_prefix}_multiclass_distributions.png")
                plt.close()  # 释放图表资源
                logger.info("四分类概率分布图已保存")
            except Exception as e:
                logger.error(f"生成四分类概率分布图时出错: {e}")
                plt.close()  # 确保释放资源
            
            try:
                # 2. 四分类归一化概率的饼图
                plt.figure(figsize=(10, 8))
                avg_depression = results_df["Normalized_Depression_Prob"].mean()
                avg_anxiety = results_df["Normalized_Anxiety_Prob"].mean()
                avg_mix = results_df["Normalized_Mix_Prob"].mean()
                avg_others = results_df["Normalized_Others_Prob"].mean()
                
                # 过滤掉NaN值
                values = []
                labels = []
                colors = []
                
                if not math.isnan(avg_depression):
                    values.append(avg_depression)
                    labels.append("Depression")
                    colors.append("blue")
                if not math.isnan(avg_anxiety):
                    values.append(avg_anxiety)
                    labels.append("Anxiety")
                    colors.append("red")
                if not math.isnan(avg_mix):
                    values.append(avg_mix)
                    labels.append("Mix")
                    colors.append("green")
                if not math.isnan(avg_others):
                    values.append(avg_others)
                    labels.append("Others")
                    colors.append("orange")
                
                if values:
                    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
                    plt.title("Overall Four-Class Distribution")
                    plt.savefig(f"{visualization_dir}/{output_prefix}_multiclass_distribution.png")
                    logger.info("四分类饼图已保存")
                else:
                    logger.warning("无法生成四分类饼图：所有数据都包含NaN值")
                plt.close()  # 释放图表资源
            except Exception as e:
                logger.error(f"生成四分类饼图时出错: {e}")
                plt.close()  # 确保释放资源
        
        elif classification_mode == "depression_symptom":
            # 抑郁症状检测可视化
            logger.info("开始生成抑郁症状检测可视化图表...")
            try:
                # 1. 有/无抑郁症状的概率分布直方图
                plt.figure(figsize=(12, 8))
                
                plt.subplot(2, 2, 1)
                sns.histplot(results_df["Has_Depression_Symptom_Probability"].dropna(), kde=True, color="blue")
                plt.title("Has Depression Symptom Probability Distribution")
                plt.xlabel("Probability")
                plt.ylabel("Frequency")
                
                plt.subplot(2, 2, 2)
                sns.histplot(results_df["No_Depression_Symptom_Probability"].dropna(), kde=True, color="red")
                plt.title("No Depression Symptom Probability Distribution")
                plt.xlabel("Probability")
                plt.ylabel("Frequency")
                
                plt.subplot(2, 2, 3)
                sns.histplot(results_df["Normalized_Has_Depression_Symptom_Prob"].dropna(), kde=True, color="darkblue")
                plt.title("Normalized Has Depression Symptom Prob")
                plt.xlabel("Probability")
                plt.ylabel("Frequency")
                
                plt.subplot(2, 2, 4)
                sns.histplot(results_df["Normalized_No_Depression_Symptom_Prob"].dropna(), kde=True, color="darkred")
                plt.title("Normalized No Depression Symptom Prob")
                plt.xlabel("Probability")
                plt.ylabel("Frequency")
                
                plt.tight_layout()
                plt.savefig(f"{visualization_dir}/{output_prefix}_depression_symptom_distributions.png")
                plt.close()
                logger.info("抑郁症状检测概率分布图已保存")
            except Exception as e:
                logger.error(f"生成抑郁症状检测概率分布图时出错: {e}")
                plt.close()
            
            try:
                # 2. 有/无抑郁症状的饼图
                plt.figure(figsize=(10, 8))
                avg_has_symptom = results_df["Normalized_Has_Depression_Symptom_Prob"].mean()
                avg_no_symptom = results_df["Normalized_No_Depression_Symptom_Prob"].mean()
                
                if not (math.isnan(avg_has_symptom) or math.isnan(avg_no_symptom)):
                    plt.pie([avg_has_symptom, avg_no_symptom], 
                           labels=["有抑郁症状", "无抑郁症状"], 
                           autopct='%1.1f%%', 
                           colors=["blue", "red"],
                           startangle=90)
                    plt.title("Overall Depression Symptom Distribution")
                    plt.savefig(f"{visualization_dir}/{output_prefix}_depression_symptom_pie.png")
                    logger.info("抑郁症状检测饼图已保存")
                else:
                    logger.warning("无法生成抑郁症状检测饼图：数据包含NaN值")
                plt.close()
            except Exception as e:
                logger.error(f"生成抑郁症状检测饼图时出错: {e}")
                plt.close()
        
        elif classification_mode == "anxiety_symptom":
            # 焦虑症状检测可视化
            logger.info("开始生成焦虑症状检测可视化图表...")
            try:
                # 1. 有/无焦虑症状的概率分布直方图
                plt.figure(figsize=(12, 8))
                
                plt.subplot(2, 2, 1)
                sns.histplot(results_df["Has_Anxiety_Symptom_Probability"].dropna(), kde=True, color="orange")
                plt.title("Has Anxiety Symptom Probability Distribution")
                plt.xlabel("Probability")
                plt.ylabel("Frequency")
                
                plt.subplot(2, 2, 2)
                sns.histplot(results_df["No_Anxiety_Symptom_Probability"].dropna(), kde=True, color="green")
                plt.title("No Anxiety Symptom Probability Distribution")
                plt.xlabel("Probability")
                plt.ylabel("Frequency")
                
                plt.subplot(2, 2, 3)
                sns.histplot(results_df["Normalized_Has_Anxiety_Symptom_Prob"].dropna(), kde=True, color="darkorange")
                plt.title("Normalized Has Anxiety Symptom Prob")
                plt.xlabel("Probability")
                plt.ylabel("Frequency")
                
                plt.subplot(2, 2, 4)
                sns.histplot(results_df["Normalized_No_Anxiety_Symptom_Prob"].dropna(), kde=True, color="darkgreen")
                plt.title("Normalized No Anxiety Symptom Prob")
                plt.xlabel("Probability")
                plt.ylabel("Frequency")
                
                plt.tight_layout()
                plt.savefig(f"{visualization_dir}/{output_prefix}_anxiety_symptom_distributions.png")
                plt.close()
                logger.info("焦虑症状检测概率分布图已保存")
            except Exception as e:
                logger.error(f"生成焦虑症状检测概率分布图时出错: {e}")
                plt.close()
            
            try:
                # 2. 有/无焦虑症状的饼图
                plt.figure(figsize=(10, 8))
                avg_has_symptom = results_df["Normalized_Has_Anxiety_Symptom_Prob"].mean()
                avg_no_symptom = results_df["Normalized_No_Anxiety_Symptom_Prob"].mean()
                
                if not (math.isnan(avg_has_symptom) or math.isnan(avg_no_symptom)):
                    plt.pie([avg_has_symptom, avg_no_symptom], 
                           labels=["有焦虑症状", "无焦虑症状"], 
                           autopct='%1.1f%%', 
                           colors=["orange", "green"],
                           startangle=90)
                    plt.title("Overall Anxiety Symptom Distribution")
                    plt.savefig(f"{visualization_dir}/{output_prefix}_anxiety_symptom_pie.png")
                    logger.info("焦虑症状检测饼图已保存")
                else:
                    logger.warning("无法生成焦虑症状检测饼图：数据包含NaN值")
                plt.close()
            except Exception as e:
                logger.error(f"生成焦虑症状检测饼图时出错: {e}")
                plt.close()
        
        logger.info(f"结果可视化图表已保存在 {visualization_dir} 目录下")
        
    except Exception as e:
        logger.error(f"可视化过程中出现严重错误: {e}")
        # 确保即使在异常情况下也释放matplotlib资源
        try:
            plt.close('all')
        except:
            pass

# ICD-10工具函数模块
class ICD10Utils:
    """ICD-10代码处理工具类"""
    
    # 定义11种允许的疾病大类
    ALLOWED_DISEASES = {'F32', 'F41', 'F39', 'F51', 'F98', 'F42', 'F31', 'F43', 'F45', 'F20', 'Z71'}
    
    @staticmethod
    def extract_major_class(code):
        """
        从ICD-10代码中提取大类
        
        Args:
            code: ICD-10代码 (如 'F32.900', 'F41.100')
            
        Returns:
            str: 大类代码 (如 'F32', 'F41') 或 None
        """
        import re
        if pd.isna(code) or code is None:
            return None
        
        code_str = str(code).strip()
        # 匹配F开头的代码或Z71
        major_match = re.match(r'(F\d+|Z71)', code_str)
        if major_match:
            major_code = major_match.group(1)
            # 只返回允许的疾病大类
            return major_code if major_code in ICD10Utils.ALLOWED_DISEASES else None
        return None
    
    @staticmethod
    def extract_major_minor(code):
        """
        提取ICD-10代码的大类和小类
        
        Args:
            code: ICD-10代码
            
        Returns:
            tuple: (major_class, minor_class) 或 (None, None)
        """
        if pd.isna(code) or code is None:
            return None, None
        
        code_str = str(code).strip()
        
        # 尝试匹配完整的ICD-10格式 (如 F32.900)
        if '.' in code_str:
            parts = code_str.split('.')
            major = parts[0]
            minor = code_str if len(parts) > 1 else None
        else:
            # 处理没有小数点的格式 (如 F32900)
            major = code_str[:3] if len(code_str) >= 3 else code_str
            minor = code_str if len(code_str) > 3 else None
        
        # 验证大类是否在允许列表中
        if major in ICD10Utils.ALLOWED_DISEASES:
            return major, minor
        else:
            return None, None
    
    @staticmethod
    def extract_major_classes_from_list(diagnosis_codes):
        """
        从诊断代码列表中提取所有大类
        
        Args:
            diagnosis_codes: 诊断代码列表或单个代码
            
        Returns:
            list: 大类代码列表
        """
        # 首先检查是否为None或空值
        if diagnosis_codes is None:
            return []
        
        # 如果是列表，直接处理
        if isinstance(diagnosis_codes, list):
            if len(diagnosis_codes) == 0:
                return []
            diagnosis_list = diagnosis_codes
        else:
            # 对于非列表类型，检查是否为NaN
            try:
                if pd.isna(diagnosis_codes):
                    return []
            except (TypeError, ValueError):
                # 如果pd.isna()失败，说明不是标准的NaN值
                pass
            
            # 处理字符串格式的诊断代码
            if isinstance(diagnosis_codes, str):
                try:
                    # 尝试解析为列表
                    import ast
                    diagnosis_list = ast.literal_eval(diagnosis_codes)
                    if isinstance(diagnosis_list, list):
                        diagnosis_list = diagnosis_list
                    else:
                        diagnosis_list = [diagnosis_codes]
                except:
                    diagnosis_list = [diagnosis_codes]
            else:
                diagnosis_list = [diagnosis_codes]
        
        # 提取所有大类
        major_classes = []
        for code in diagnosis_list:
            major_class = ICD10Utils.extract_major_class(code)
            if major_class and major_class not in major_classes:
                major_classes.append(major_class)
        
        return major_classes
    
    @staticmethod
    def filter_allowed_codes(codes):
        """
        过滤出允许的疾病代码
        
        Args:
            codes: 代码列表
            
        Returns:
            list: 过滤后的代码列表
        """
        if not isinstance(codes, list):
            codes = [codes] if codes is not None else []
        
        filtered_codes = []
        for code in codes:
            major_class = ICD10Utils.extract_major_class(code)
            if major_class:
                filtered_codes.append(code)
        
        return filtered_codes


# 计算模型评估指标
def balanced_sampling_with_resume(df, classification_mode, target_sample_count, processed_visits=None, min_samples_per_class=5, random_state=42):
    """
    实现按分布的平衡采样，确保每个类别至少有min_samples_per_class个样本
    支持resume功能，在已有样本基础上补充不足的类别
    
    对于ICD10和recommendation模式，只考虑ALLOWED_DISEASES中的11个大类
    """
    
    # 确定ground truth列和采样策略
    if classification_mode in ["binary", "multiclass"]:
        label_col = "OverallDiagnosis"
        use_icd_major_classes = False
    elif classification_mode == "depression_symptom":
        label_col = "DepressionDiagnosis"
        use_icd_major_classes = False
    elif classification_mode == "anxiety_symptom":
        label_col = "AnxietyDiagnosis"
        use_icd_major_classes = False
    elif classification_mode == "multilabel":
        label_col = "MultilabelDiagnosis"
        use_icd_major_classes = False
    elif classification_mode in ["icd10", "recommendation"]:
        label_col = "DiagnosisCode"
        use_icd_major_classes = True  # 对于ICD10和recommendation模式，使用大类进行平衡采样
    else:
        logger.warning(f"不支持的分类模式: {classification_mode}，使用随机采样")
        if processed_visits:
            remaining_df = df[~df["VisitNumber"].isin(processed_visits)].copy()
            remaining_needed = max(0, target_sample_count - len(processed_visits))
            if remaining_needed > 0 and len(remaining_df) > 0:
                sampled_df = remaining_df.sample(n=min(remaining_needed, len(remaining_df)), random_state=random_state)
                return sampled_df, {"method": "random_resume", "total_samples": len(sampled_df)}
            else:
                return pd.DataFrame(), {"method": "random_resume", "total_samples": 0}
        else:
            sampled_df = df.sample(n=min(target_sample_count, len(df)), random_state=random_state)
            return sampled_df, {"method": "random", "total_samples": len(sampled_df)}
    
    if label_col not in df.columns:
        logger.warning(f"未找到标签列 {label_col}，使用随机采样")
        if processed_visits:
            remaining_df = df[~df["VisitNumber"].isin(processed_visits)].copy()
            remaining_needed = max(0, target_sample_count - len(processed_visits))
            if remaining_needed > 0 and len(remaining_df) > 0:
                sampled_df = remaining_df.sample(n=min(remaining_needed, len(remaining_df)), random_state=random_state)
                return sampled_df, {"method": "random_resume", "total_samples": len(sampled_df)}
            else:
                return pd.DataFrame(), {"method": "random_resume", "total_samples": 0}
        else:
            sampled_df = df.sample(n=min(target_sample_count, len(df)), random_state=random_state)
            return sampled_df, {"method": "random", "total_samples": len(sampled_df)}
    
    # 如果有已处理的样本，先分析已有样本的分布
    already_sampled_df = pd.DataFrame()
    if processed_visits:
        already_sampled_df = df[df["VisitNumber"].isin(processed_visits)].copy()
        logger.info(f"已有样本数: {len(already_sampled_df)}")
        
        if len(already_sampled_df) >= target_sample_count:
            logger.info("已有样本数已达到目标，无需额外采样")
            return pd.DataFrame(), {"method": "resume_complete", "total_samples": 0}
    
    # 分析当前标签分布
    available_df = df[~df["VisitNumber"].isin(processed_visits)] if processed_visits else df.copy()
    
    # 处理缺失值
    available_df = available_df[available_df[label_col].notna()].copy()
    
    if len(available_df) == 0:
        logger.warning("没有可用的样本进行采样")
        return pd.DataFrame(), {"method": "no_available_samples", "total_samples": 0}
    
    # 根据模式处理标签分布
    if use_icd_major_classes:
        # 对于ICD10和recommendation模式，基于大类进行采样
        logger.info("使用ICD-10大类进行平衡采样（仅考虑ALLOWED_DISEASES中的11个大类）")
        
        # 为每个样本提取其包含的大类
        available_df['major_classes'] = available_df[label_col].apply(ICD10Utils.extract_major_classes_from_list)
        
        # 过滤出至少包含一个允许大类的样本
        available_df = available_df[available_df['major_classes'].apply(lambda x: len(x) > 0)].copy()
        
        if len(available_df) == 0:
            logger.warning("没有包含ALLOWED_DISEASES大类的样本")
            return pd.DataFrame(), {"method": "no_allowed_disease_samples", "total_samples": 0}
        
        # 统计每个大类的样本数（一个样本可能包含多个大类）
        major_class_counts = {}
        for _, row in available_df.iterrows():
            for major_class in row['major_classes']:
                major_class_counts[major_class] = major_class_counts.get(major_class, 0) + 1
        
        unique_labels = list(ICD10Utils.ALLOWED_DISEASES)  # 使用所有允许的大类作为目标
        logger.info(f"可用数据中的ICD-10大类分布: {major_class_counts}")
        
        # 分析已有样本的大类分布
        existing_label_counts = {}
        if len(already_sampled_df) > 0:
            already_sampled_df['major_classes'] = already_sampled_df[label_col].apply(ICD10Utils.extract_major_classes_from_list)
            for _, row in already_sampled_df.iterrows():
                for major_class in row['major_classes']:
                    existing_label_counts[major_class] = existing_label_counts.get(major_class, 0) + 1
            logger.info(f"已有样本的ICD-10大类分布: {existing_label_counts}")
    else:
        # 对于其他模式，使用原始标签
        label_counts = available_df[label_col].value_counts()
        unique_labels = label_counts.index.tolist()
        
        logger.info(f"可用数据中的标签分布: {label_counts.to_dict()}")
        
        # 分析已有样本的标签分布
        existing_label_counts = {}
        if len(already_sampled_df) > 0:
            existing_label_counts = already_sampled_df[label_col].value_counts().to_dict()
            logger.info(f"已有样本的标签分布: {existing_label_counts}")
    
    # 计算每个类别还需要多少样本
    samples_needed_per_class = {}
    total_needed = 0
    
    for label in unique_labels:
        existing_count = existing_label_counts.get(label, 0)
        needed = max(0, min_samples_per_class - existing_count)
        samples_needed_per_class[label] = needed
        total_needed += needed
    
    logger.info(f"每个类别还需要的样本数: {samples_needed_per_class}")
    logger.info(f"满足最小要求总共还需要: {total_needed} 个样本")
    
    # 计算剩余可分配的样本数
    remaining_budget = target_sample_count - len(already_sampled_df) - total_needed
    logger.info(f"满足最小要求后剩余可分配样本数: {remaining_budget}")
    
    # 第一步：确保每个类别至少有min_samples_per_class个样本
    selected_samples = []
    
    if use_icd_major_classes:
        # 对于ICD大类模式，需要特殊处理
        for major_class in unique_labels:
            needed = samples_needed_per_class[major_class]
            if needed > 0:
                # 找到包含该大类的样本
                class_samples = available_df[available_df['major_classes'].apply(lambda x: major_class in x)]
                
                if len(class_samples) >= needed:
                    # 随机选择需要的样本数
                    selected = class_samples.sample(n=needed, random_state=random_state)
                    selected_samples.append(selected)
                    # 从可用数据中移除已选择的样本
                    available_df = available_df.drop(selected.index)
                else:
                    # 如果该类别样本不足，全部选择
                    logger.warning(f"大类 {major_class} 只有 {len(class_samples)} 个样本，少于最小要求 {needed}")
                    if len(class_samples) > 0:
                        selected_samples.append(class_samples)
                        available_df = available_df.drop(class_samples.index)
    else:
        # 对于其他模式，使用原始逻辑
        for label in unique_labels:
            needed = samples_needed_per_class[label]
            if needed > 0:
                label_samples = available_df[available_df[label_col] == label]
                if len(label_samples) >= needed:
                    # 随机选择需要的样本数
                    selected = label_samples.sample(n=needed, random_state=random_state)
                    selected_samples.append(selected)
                    # 从可用数据中移除已选择的样本
                    available_df = available_df.drop(selected.index)
                else:
                    # 如果该类别样本不足，全部选择
                    logger.warning(f"类别 {label} 只有 {len(label_samples)} 个样本，少于最小要求 {needed}")
                    selected_samples.append(label_samples)
                    available_df = available_df.drop(label_samples.index)
    
    # 第二步：如果还有剩余预算，按比例分配给各类别
    if remaining_budget > 0 and len(available_df) > 0:
        if use_icd_major_classes:
            # 对于ICD大类模式，按大类比例分配剩余样本
            remaining_major_counts = {}
            for _, row in available_df.iterrows():
                for major_class in row['major_classes']:
                    remaining_major_counts[major_class] = remaining_major_counts.get(major_class, 0) + 1
            
            total_remaining_major = sum(remaining_major_counts.values())
            
            for major_class, count in remaining_major_counts.items():
                if remaining_budget <= 0:
                    break
                    
                # 按比例计算该大类应该分配的样本数
                proportion = count / total_remaining_major
                allocated = min(int(remaining_budget * proportion), count)
                
                if allocated > 0:
                    class_samples = available_df[available_df['major_classes'].apply(lambda x: major_class in x)]
                    if len(class_samples) > 0:
                        selected = class_samples.sample(n=min(allocated, len(class_samples)), random_state=random_state)
                        selected_samples.append(selected)
                        available_df = available_df.drop(selected.index)
                        remaining_budget -= len(selected)
        else:
            # 对于其他模式，使用原始逻辑
            remaining_label_counts = available_df[label_col].value_counts()
            total_remaining = len(available_df)
            
            for label in remaining_label_counts.index:
                if remaining_budget <= 0:
                    break
                    
                # 按比例计算该类别应该分配的样本数
                proportion = remaining_label_counts[label] / total_remaining
                allocated = min(int(remaining_budget * proportion), remaining_label_counts[label])
                
                if allocated > 0:
                    label_samples = available_df[available_df[label_col] == label]
                    selected = label_samples.sample(n=allocated, random_state=random_state)
                    selected_samples.append(selected)
                    available_df = available_df.drop(selected.index)
                    remaining_budget -= allocated
        
        # 如果还有剩余预算，随机分配
        if remaining_budget > 0 and len(available_df) > 0:
            final_selected = available_df.sample(n=min(remaining_budget, len(available_df)), random_state=random_state)
            selected_samples.append(final_selected)
    
    # 合并所有选择的样本
    if selected_samples:
        sampled_df = pd.concat(selected_samples, ignore_index=True)
    else:
        sampled_df = pd.DataFrame()
    
    # 生成采样信息
    if use_icd_major_classes and len(sampled_df) > 0:
        # 对于ICD大类模式，计算大类分布
        sampled_df['major_classes'] = sampled_df[label_col].apply(ICD10Utils.extract_major_classes_from_list)
        final_major_distribution = {}
        for _, row in sampled_df.iterrows():
            for major_class in row['major_classes']:
                final_major_distribution[major_class] = final_major_distribution.get(major_class, 0) + 1
        
        sampling_info = {
            "method": "balanced_sampling_with_resume_icd_major",
            "total_samples": len(sampled_df),
            "min_samples_per_class": min_samples_per_class,
            "target_sample_count": target_sample_count,
            "existing_samples": len(already_sampled_df),
            "samples_needed_per_class": samples_needed_per_class,
            "final_major_class_distribution": final_major_distribution,
            "allowed_diseases": list(ICD10Utils.ALLOWED_DISEASES)
        }
    else:
        sampling_info = {
            "method": "balanced_sampling_with_resume",
            "total_samples": len(sampled_df),
            "min_samples_per_class": min_samples_per_class,
            "target_sample_count": target_sample_count,
            "existing_samples": len(already_sampled_df),
            "samples_needed_per_class": samples_needed_per_class,
            "final_distribution": sampled_df[label_col].value_counts().to_dict() if len(sampled_df) > 0 else {}
        }
    
    logger.info(f"平衡采样完成，共选择 {len(sampled_df)} 个样本")
    if len(sampled_df) > 0:
        if use_icd_major_classes:
            # 显示ICD大类分布
            logger.info(f"新采样的ICD-10大类分布: {final_major_distribution}")
            
            # 计算总的大类分布（包括已有样本）
            if len(already_sampled_df) > 0:
                total_major_distribution = {}
                # 已有样本的大类分布
                for major_class, count in existing_label_counts.items():
                    total_major_distribution[major_class] = count
                # 新采样的大类分布
                for major_class, count in final_major_distribution.items():
                    total_major_distribution[major_class] = total_major_distribution.get(major_class, 0) + count
                logger.info(f"总ICD-10大类分布（已有+新采样）: {total_major_distribution}")
        else:
            # 显示原始标签分布
            final_distribution = sampled_df[label_col].value_counts()
            logger.info(f"新采样的标签分布: {final_distribution.to_dict()}")
            
            # 计算总分布（包括已有样本）
            if len(already_sampled_df) > 0:
                total_distribution = pd.concat([already_sampled_df, sampled_df])[label_col].value_counts()
                logger.info(f"总标签分布（已有+新采样）: {total_distribution.to_dict()}")
    
    return sampled_df, sampling_info


def calculate_evaluation_metrics(results_df, classification_mode="binary"):
    """计算模型评估指标：weighted-F1, balanced ACC, 和AUPRC"""
    
    # 根据分类模式确定ground truth列和需要的列
    print(f"classification_mode: {classification_mode}")
    if classification_mode in ["binary", "multiclass"]:
        ground_truth_col = "OverallDiagnosis"
        if classification_mode == "binary":
            required_cols = ["Normalized_Depression_Prob", "Normalized_Anxiety_Prob"]
        else:  # multiclass
            required_cols = ["Normalized_Depression_Prob", "Normalized_Anxiety_Prob", 
                            "Normalized_Mix_Prob", "Normalized_Others_Prob"]
    elif classification_mode == "depression_symptom":
        ground_truth_col = "DepressionDiagnosis"
        required_cols = ["Normalized_Has_Depression_Symptom_Prob", "Normalized_No_Depression_Symptom_Prob"]
    elif classification_mode == "anxiety_symptom":
        ground_truth_col = "AnxietyDiagnosis"
        required_cols = ["Normalized_Has_Anxiety_Symptom_Prob", "Normalized_No_Anxiety_Symptom_Prob"]
    elif classification_mode == "multilabel":
        ground_truth_col = "MultilabelDiagnosis"
        required_cols = ["Depression_Probability", "Anxiety_Probability", "Depression_Label", "Anxiety_Label"]
    elif classification_mode == "icd10":
        ground_truth_col = "DiagnosisCode"
        required_cols = ["Predicted_ICD10", "Ground_Truth_ICD10", "Predicted_Major_Class", "Predicted_Minor_Class"]
    elif classification_mode == "recommendation":
        ground_truth_col = "DiagnosisCode"
        required_cols = ["Recommended_ICD10_Codes", "Ground_Truth_ICD10", "Top1_Code", "Top3_Codes"]
    else:
        logger.warning(f"不支持的分类模式: {classification_mode}")
        return {}
    
    # 检查是否有ground truth列
    if classification_mode in ["icd10", "recommendation"]:
        # ICD10和recommendation模式检查Ground_Truth_ICD10列
        if "Ground_Truth_ICD10" not in results_df.columns:
            logger.warning(f"数据中缺少Ground_Truth_ICD10列，无法计算评估指标")
            return {}
    else:
        if ground_truth_col not in results_df.columns:
            logger.warning(f"数据中缺少{ground_truth_col}列，无法计算评估指标")
            return {}
    
    # 检查必要的列是否存在
    missing_cols = [col for col in required_cols if col not in results_df.columns]
    if missing_cols:
        logger.warning(f"数据中缺少必要的列: {missing_cols}，无法计算评估指标")
        return {}
    
    # 过滤有效数据（排除空值和无效预测）
    if classification_mode == "icd10":
        # ICD10模式：检查预测代码和ground truth代码
        valid_mask = (
            results_df["Ground_Truth_ICD10"].notna() & 
            results_df["Predicted_ICD10"].notna()
        )
        valid_data = results_df[valid_mask].copy()
        # ICD10模式的y_true使用Ground_Truth_ICD10列
        y_true = valid_data["Ground_Truth_ICD10"].values
    elif classification_mode == "recommendation":
        # Recommendation模式：检查推荐代码和ground truth代码
        valid_mask = (
            results_df["Ground_Truth_ICD10"].notna() & 
            results_df["Recommended_ICD10_Codes"].notna()
        )
        valid_data = results_df[valid_mask].copy()
        # Recommendation模式的y_true使用Ground_Truth_ICD10列
        y_true = valid_data["Ground_Truth_ICD10"].values
    else:
        # 其他模式：检查概率相关列
        valid_mask = (
            results_df[ground_truth_col].notna() & 
            results_df[required_cols].notna().all(axis=1)
        )
        
        # 对于非multilabel模式，还需要检查Total_Probability
        if classification_mode != "multilabel" and "Total_Probability" in results_df.columns:
            valid_mask = valid_mask & (results_df["Total_Probability"] > 0)
        
        valid_data = results_df[valid_mask].copy()
        # 获取真实标签
        y_true = valid_data[ground_truth_col].values
    
    if len(valid_data) == 0:
        logger.warning("没有有效的数据用于评估")
        return {}
    
    logger.info(f"用于评估的有效样本数: {len(valid_data)}")
    
    if classification_mode == "binary":
        return calculate_binary_evaluation_metrics(valid_data, y_true)
    elif classification_mode == "multiclass":
        return calculate_multiclass_evaluation_metrics(valid_data, y_true)
    elif classification_mode == "depression_symptom":
        return calculate_symptom_evaluation_metrics(valid_data, y_true, "depression")
    elif classification_mode == "anxiety_symptom":
        return calculate_symptom_evaluation_metrics(valid_data, y_true, "anxiety")
    elif classification_mode == "multilabel":
        return calculate_multilabel_evaluation_metrics(valid_data, y_true)
    elif classification_mode == "icd10":
        return calculate_icd10_evaluation_metrics(valid_data, y_true)
    elif classification_mode == "recommendation":
        return calculate_recommendation_evaluation_metrics(valid_data, y_true)

# 计算二分类评估指标
def calculate_binary_evaluation_metrics(valid_data, y_true):
    """计算二分类评估指标"""
    # 获取预测概率
    depression_probs = valid_data["Normalized_Depression_Prob"].values
    anxiety_probs = valid_data["Normalized_Anxiety_Prob"].values
    
    # 基于最高概率进行预测
    predictions = []
    prediction_probs = []  # 用于AUPRC计算
    
    for i in range(len(valid_data)):
        dep_prob = depression_probs[i]
        anx_prob = anxiety_probs[i]
        
        # 存储预测概率
        prediction_probs.append([dep_prob, anx_prob])
        
        # 基于最高概率进行预测
        if dep_prob >= anx_prob:
            predictions.append("抑郁")
        else:
            predictions.append("焦虑")
    
    y_pred = np.array(predictions)
    prediction_probs = np.array(prediction_probs)
    
    # 统一标签格式
    y_true_processed = []
    for label in y_true:
        label_lower = str(label).lower()
        if 'depression' in label_lower or 'depres' in label_lower:
            y_true_processed.append("抑郁")
        elif 'anxiety' in label_lower or 'anxie' in label_lower:
            y_true_processed.append("焦虑")
        else:
            # 对于二分类，其他标签暂时归类为others但不参与评估
            y_true_processed.append("others")
    
    y_true_processed = np.array(y_true_processed)
    
    # 过滤掉others类别的数据（二分类只关注抑郁 vs 焦虑）
    binary_mask = np.isin(y_true_processed, ["抑郁", "焦虑"])
    y_true_binary = y_true_processed[binary_mask]
    y_pred_binary = y_pred[binary_mask]
    prediction_probs_binary = prediction_probs[binary_mask]
    
    if len(y_true_binary) == 0:
        logger.warning("过滤后没有抑郁或焦虑标签的数据用于二分类评估")
        return {}
    
    # 获取唯一的标签
    unique_labels = ["抑郁", "焦虑"]
    
    # 检查实际存在的类别
    actual_labels = list(set(y_true_binary))
    logger.info(f"实际存在的类别: {actual_labels}")
    
    # 如果只有一个类别，无法进行有意义的二分类评估
    if len(actual_labels) < 2:
        logger.warning(f"数据中只包含一个类别: {actual_labels}，无法进行二分类评估")
        return {
            "classification_mode": "binary",
            "error": "insufficient_classes",
            "message": f"只包含一个类别: {actual_labels}，无法进行二分类评估",
            "total_samples": len(y_true_binary),
            "unique_labels": actual_labels
        }
    
    try:
        # 计算weighted F1-score
        weighted_f1 = f1_score(y_true_binary, y_pred_binary, labels=unique_labels, average='weighted', zero_division='warn')
        
        # 计算balanced accuracy
        balanced_acc = balanced_accuracy_score(y_true_binary, y_pred_binary)
        
        # 计算AUPRC - 使用sklearn的precision_recall_curve和auc
        auprc_scores = []
        for i, label in enumerate(unique_labels):
            # 将当前类别标记为1，其他类别标记为0
            y_true_binary_label = (y_true_binary == label).astype(int)
            y_scores = prediction_probs_binary[:, i]
            
            # 计算precision-recall曲线和AUPRC
            precision, recall, _ = precision_recall_curve(y_true_binary_label, y_scores)
            auprc = auc(recall, precision)
            auprc_scores.append(auprc)
            
            logger.debug(f"{label} AUPRC计算:")
            logger.debug(f"  正类样本数: {sum(y_true_binary_label)}")
            logger.debug(f"  负类样本数: {len(y_true_binary_label) - sum(y_true_binary_label)}")
            logger.debug(f"  AUPRC: {auprc}")
        
        # 计算macro平均AUPRC
        macro_auprc = np.mean(auprc_scores)
        
        # 计算加权平均AUPRC
        class_counts = [np.sum(y_true_binary == label) for label in unique_labels]
        total_samples = len(y_true_binary)
        weights = [count / total_samples for count in class_counts]
        weighted_auprc = np.average(auprc_scores, weights=weights)
        
        # 生成分类报告
        class_report = classification_report(y_true_binary, y_pred_binary, labels=unique_labels, zero_division='warn')
        
        # 生成混淆矩阵
        cm = confusion_matrix(y_true_binary, y_pred_binary, labels=unique_labels)
        
        # 计算每个类别的指标
        individual_f1 = f1_score(y_true_binary, y_pred_binary, labels=unique_labels, average=None, zero_division='warn')
        
        # 确保individual_f1是数组格式
        if isinstance(individual_f1, (int, float)):
            individual_f1 = [individual_f1]
        elif hasattr(individual_f1, 'tolist'):
            individual_f1 = individual_f1.tolist()
        
        metrics = {
            "classification_mode": "binary",
            "weighted_f1": weighted_f1,
            "balanced_accuracy": balanced_acc,
            "macro_auprc": macro_auprc,
            "weighted_auprc": weighted_auprc,
            "individual_auprc": dict(zip(unique_labels, auprc_scores)),
            "individual_f1": dict(zip(unique_labels, individual_f1)),
            "classification_report": class_report,
            "confusion_matrix": cm,
            "sample_counts": dict(zip(unique_labels, class_counts)),
            "total_samples": total_samples,
            "unique_labels": unique_labels
        }
        
        # 记录结果
        logger.info("=== 二分类模型评估指标 ===")
        logger.info(f"总样本数: {total_samples}")
        logger.info(f"各类别样本数: {dict(zip(unique_labels, class_counts))}")
        logger.info(f"Weighted F1-Score: {weighted_f1:.4f}")
        logger.info(f"Balanced Accuracy: {balanced_acc:.4f}")
        logger.info(f"Macro AUPRC: {macro_auprc:.4f}")
        logger.info(f"Weighted AUPRC: {weighted_auprc:.4f}")
        
        logger.info("各类别F1-Score:")
        for label, f1 in zip(unique_labels, individual_f1):
            logger.info(f"  {label}: {f1:.4f}")
        
        logger.info("各类别AUPRC:")
        for label, auprc in zip(unique_labels, auprc_scores):
            logger.info(f"  {label}: {auprc:.4f}")
        
        logger.info("分类报告:")
        logger.info(f"\n{class_report}")
        
        logger.info("混淆矩阵:")
        logger.info(f"\n{cm}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"计算二分类评估指标时出错: {e}")
        return {}

# 计算四分类评估指标
def calculate_multiclass_evaluation_metrics(valid_data, y_true):
    """计算四分类评估指标"""
    # 获取预测概率
    depression_probs = valid_data["Normalized_Depression_Prob"].values
    anxiety_probs = valid_data["Normalized_Anxiety_Prob"].values
    mix_probs = valid_data["Normalized_Mix_Prob"].values
    others_probs = valid_data["Normalized_Others_Prob"].values
    
    # 基于最高概率进行预测
    predictions = []
    prediction_probs = []  # 用于AUPRC计算
    
    for i in range(len(valid_data)):
        probs = [depression_probs[i], anxiety_probs[i], mix_probs[i], others_probs[i]]
        prediction_probs.append(probs)
        
        # 基于最高概率进行预测
        max_idx = np.argmax(probs)
        if max_idx == 0:
            predictions.append("抑郁")
        elif max_idx == 1:
            predictions.append("焦虑")
        elif max_idx == 2:
            predictions.append("mix")
        else:
            predictions.append("others")
    
    y_pred = np.array(predictions)
    prediction_probs = np.array(prediction_probs)
    
    # 统一标签格式
    y_true_processed = []
    for label in y_true:
        label_str = str(label)
        label_lower = label_str.lower()
        # 先检查mix类别，因为它可能与其他关键词冲突
        if label_str == 'Mix' or 'mix' in label_lower or '混合' in label_str or '共病' in label_str:
            y_true_processed.append("mix")
        elif 'depression' in label_lower or 'depres' in label_lower:
            y_true_processed.append("抑郁")
        elif 'anxiety' in label_lower or 'anxie' in label_lower:
            y_true_processed.append("焦虑")
        else:
            y_true_processed.append("others")
    
    y_true_processed = np.array(y_true_processed)
    
    # 获取唯一的标签
    unique_labels = ["抑郁", "焦虑", "mix", "others"]
    
    try:
        # 计算weighted F1-score
        weighted_f1 = f1_score(y_true_processed, y_pred, labels=unique_labels, average='weighted', zero_division='warn')
        
        # 计算balanced accuracy
        balanced_acc = balanced_accuracy_score(y_true_processed, y_pred)
        
        # 计算AUPRC - 使用sklearn的precision_recall_curve和auc
        auprc_scores = []
        for i, label in enumerate(unique_labels):
            # 将当前类别标记为1，其他类别标记为0
            y_true_binary_label = (y_true_processed == label).astype(int)
            y_scores = prediction_probs[:, i]
            
            # 计算precision-recall曲线和AUPRC
            precision, recall, _ = precision_recall_curve(y_true_binary_label, y_scores)
            auprc = auc(recall, precision)
            auprc_scores.append(auprc)
            
            logger.debug(f"{label} AUPRC计算:")
            logger.debug(f"  正类样本数: {sum(y_true_binary_label)}")
            logger.debug(f"  负类样本数: {len(y_true_binary_label) - sum(y_true_binary_label)}")
            logger.debug(f"  AUPRC: {auprc}")
        
        # 计算macro平均AUPRC
        macro_auprc = np.mean(auprc_scores)
        
        # 计算加权平均AUPRC
        class_counts = [np.sum(y_true_processed == label) for label in unique_labels]
        total_samples = len(y_true_processed)
        weights = [count / total_samples for count in class_counts]
        weighted_auprc = np.average(auprc_scores, weights=weights)
        
        # 生成分类报告
        class_report = classification_report(y_true_processed, y_pred, labels=unique_labels, zero_division='warn')
        
        # 生成混淆矩阵
        cm = confusion_matrix(y_true_processed, y_pred, labels=unique_labels)
        
        # 计算每个类别的指标
        individual_f1 = f1_score(y_true_processed, y_pred, labels=unique_labels, average=None, zero_division='warn')
        
        # 确保individual_f1是数组格式
        if isinstance(individual_f1, (int, float)):
            individual_f1 = [individual_f1]
        elif hasattr(individual_f1, 'tolist'):
            individual_f1 = individual_f1.tolist()
        
        metrics = {
            "classification_mode": "multiclass",
            "weighted_f1": weighted_f1,
            "balanced_accuracy": balanced_acc,
            "macro_auprc": macro_auprc,
            "weighted_auprc": weighted_auprc,
            "individual_auprc": dict(zip(unique_labels, auprc_scores)),
            "individual_f1": dict(zip(unique_labels, individual_f1)),
            "classification_report": class_report,
            "confusion_matrix": cm,
            "sample_counts": dict(zip(unique_labels, class_counts)),
            "total_samples": total_samples,
            "unique_labels": unique_labels
        }
        
        # 记录结果
        logger.info("=== 四分类模型评估指标 ===")
        logger.info(f"总样本数: {total_samples}")
        logger.info(f"各类别样本数: {dict(zip(unique_labels, class_counts))}")
        logger.info(f"Weighted F1-Score: {weighted_f1:.4f}")
        logger.info(f"Balanced Accuracy: {balanced_acc:.4f}")
        logger.info(f"Macro AUPRC: {macro_auprc:.4f}")
        logger.info(f"Weighted AUPRC: {weighted_auprc:.4f}")
        
        logger.info("各类别F1-Score:")
        for label, f1 in zip(unique_labels, individual_f1):
            logger.info(f"  {label}: {f1:.4f}")
        
        logger.info("各类别AUPRC:")
        for label, auprc in zip(unique_labels, auprc_scores):
            logger.info(f"  {label}: {auprc:.4f}")
        
        logger.info("分类报告:")
        logger.info(f"\n{class_report}")
        
        logger.info("混淆矩阵:")
        logger.info(f"\n{cm}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"计算四分类评估指标时出错: {e}")
        return {}

# 计算症状检测评估指标
def calculate_symptom_evaluation_metrics(valid_data, y_true, symptom_type="depression"):
    """计算症状检测评估指标"""
    
    # 获取预测概率
    if symptom_type == "depression":
        has_symptom_probs = valid_data["Normalized_Has_Depression_Symptom_Prob"].values
        no_symptom_probs = valid_data["Normalized_No_Depression_Symptom_Prob"].values
    else:  # anxiety
        has_symptom_probs = valid_data["Normalized_Has_Anxiety_Symptom_Prob"].values
        no_symptom_probs = valid_data["Normalized_No_Anxiety_Symptom_Prob"].values
    
    # 基于最高概率进行预测
    predictions = []
    prediction_probs = []  # 用于AUPRC计算
    
    for i in range(len(valid_data)):
        has_prob = has_symptom_probs[i]
        no_prob = no_symptom_probs[i]
        
        # 存储预测概率
        prediction_probs.append([has_prob, no_prob])
        
        # 基于最高概率进行预测
        if has_prob >= no_prob:
            predictions.append("有")
        else:
            predictions.append("没有")
    
    y_pred = np.array(predictions)
    prediction_probs = np.array(prediction_probs)
    
    # 统一标签格式
    y_true_processed = []
    for label in y_true:
        label_str = str(label)
        if label_str in ["有", "1", "True", "true", "Yes", "yes", "是"]:
            y_true_processed.append("有")
        elif label_str in ["没有", "无", "0", "False", "false", "No", "no", "否"]:
            y_true_processed.append("没有")
        else:
            # 默认处理
            y_true_processed.append("没有")
    
    y_true_processed = np.array(y_true_processed)
    
    # 获取唯一的标签
    unique_labels = ["有", "没有"]
    
    # 检查实际存在的类别
    actual_labels = list(set(y_true_processed))
    logger.info(f"实际存在的类别: {actual_labels}")
    
    # 如果只有一个类别，无法进行有意义的二分类评估
    if len(actual_labels) < 2:
        logger.warning(f"数据中只包含一个类别: {actual_labels}，无法进行症状检测评估")
        return {
            "classification_mode": f"{symptom_type}_symptom",
            "error": "insufficient_classes",
            "message": f"只包含一个类别: {actual_labels}，无法进行症状检测评估",
            "total_samples": len(y_true_processed),
            "unique_labels": actual_labels
        }
    
    try:
        # 计算weighted F1-score
        weighted_f1 = f1_score(y_true_processed, y_pred, labels=unique_labels, average='weighted', zero_division='warn')
        
        # 计算balanced accuracy
        balanced_acc = balanced_accuracy_score(y_true_processed, y_pred)
        
        # 计算AUPRC - 使用sklearn的precision_recall_curve和auc
        auprc_scores = []
        for i, label in enumerate(unique_labels):
            # 将当前类别标记为1，其他类别标记为0
            y_true_binary_label = (y_true_processed == label).astype(int)
            y_scores = prediction_probs[:, i]
            
            # 计算precision-recall曲线和AUPRC
            precision, recall, _ = precision_recall_curve(y_true_binary_label, y_scores)
            auprc = auc(recall, precision)
            auprc_scores.append(auprc)
            
            logger.debug(f"{label} AUPRC计算:")
            logger.debug(f"  正类样本数: {sum(y_true_binary_label)}")
            logger.debug(f"  负类样本数: {len(y_true_binary_label) - sum(y_true_binary_label)}")
            logger.debug(f"  AUPRC: {auprc}")
        
        # 计算macro平均AUPRC
        macro_auprc = np.mean(auprc_scores)
        
        # 计算加权平均AUPRC
        class_counts = [np.sum(y_true_processed == label) for label in unique_labels]
        total_samples = len(y_true_processed)
        weights = [count / total_samples for count in class_counts]
        weighted_auprc = np.average(auprc_scores, weights=weights)
        
        # 生成分类报告
        class_report = classification_report(y_true_processed, y_pred, labels=unique_labels, zero_division='warn')
        
        # 生成混淆矩阵
        cm = confusion_matrix(y_true_processed, y_pred, labels=unique_labels)
        
        # 计算每个类别的指标
        individual_f1 = f1_score(y_true_processed, y_pred, labels=unique_labels, average=None, zero_division='warn')
        
        # 确保individual_f1是数组格式
        if isinstance(individual_f1, (int, float)):
            individual_f1 = [individual_f1]
        elif hasattr(individual_f1, 'tolist'):
            individual_f1 = individual_f1.tolist()
        
        metrics = {
            "classification_mode": f"{symptom_type}_symptom",
            "weighted_f1": weighted_f1,
            "balanced_accuracy": balanced_acc,
            "macro_auprc": macro_auprc,
            "weighted_auprc": weighted_auprc,
            "individual_auprc": dict(zip(unique_labels, auprc_scores)),
            "individual_f1": dict(zip(unique_labels, individual_f1)),
            "classification_report": class_report,
            "confusion_matrix": cm,
            "sample_counts": dict(zip(unique_labels, class_counts)),
            "total_samples": total_samples,
            "unique_labels": unique_labels
        }
        
        # 记录结果
        logger.info(f"=== {symptom_type.title()}症状检测模型评估指标 ===")
        logger.info(f"总样本数: {total_samples}")
        logger.info(f"各类别样本数: {dict(zip(unique_labels, class_counts))}")
        logger.info(f"Weighted F1-Score: {weighted_f1:.4f}")
        logger.info(f"Balanced Accuracy: {balanced_acc:.4f}")
        logger.info(f"Macro AUPRC: {macro_auprc:.4f}")
        logger.info(f"Weighted AUPRC: {weighted_auprc:.4f}")
        
        logger.info("各类别F1-Score:")
        for label, f1 in zip(unique_labels, individual_f1):
            logger.info(f"  {label}: {f1:.4f}")
        
        logger.info("各类别AUPRC:")
        for label, auprc in zip(unique_labels, auprc_scores):
            logger.info(f"  {label}: {auprc:.4f}")
        
        logger.info("分类报告:")
        logger.info(f"\n{class_report}")
        
        logger.info("混淆矩阵:")
        logger.info(f"\n{cm}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"计算{symptom_type}症状检测评估指标时出错: {e}")
        return {}

# 计算多标签评估指标
def calculate_multilabel_evaluation_metrics(valid_data, y_true):
    """计算多标签评估指标"""
    
    # 优先使用4分类评估，无论是否有OverallDiagnosis列
    # 这样可以输出4分类的confusion matrix而不是两个独立的2分类confusion matrix
    if "OverallDiagnosis" in valid_data.columns:
        return calculate_multilabel_as_multiclass_evaluation_metrics(valid_data, y_true)
    else:
        # 即使没有OverallDiagnosis列，也将多标签结果转换为4分类进行评估
        return calculate_multilabel_as_multiclass_evaluation_metrics_without_ground_truth(valid_data, y_true)
    

# 将多标签结果转换为4分类进行评估（没有OverallDiagnosis列的情况）
def calculate_multilabel_as_multiclass_evaluation_metrics_without_ground_truth(valid_data, y_true):
    """将多标签结果转换为4分类问题，基于MultilabelDiagnosis进行评估"""
    
    # 将多标签预测结果转换为4分类标签
    y_pred_multiclass = []
    for _, row in valid_data.iterrows():
        depression_label = row.get("Depression_Label", 0)
        anxiety_label = row.get("Anxiety_Label", 0)
        
        # 转换为4分类标签
        if depression_label == 0 and anxiety_label == 0:
            y_pred_multiclass.append("others")
        elif depression_label == 0 and anxiety_label == 1:
            y_pred_multiclass.append("焦虑")
        elif depression_label == 1 and anxiety_label == 0:
            y_pred_multiclass.append("抑郁")
        else:  # depression_label == 1 and anxiety_label == 1
            y_pred_multiclass.append("mix")
    
    y_pred_multiclass = np.array(y_pred_multiclass)
    
    # 将MultilabelDiagnosis转换为4分类标签
    y_true_multiclass = []
    for diagnosis in y_true:
        if pd.isna(diagnosis):
            y_true_multiclass.append("others")
        else:
            diagnosis_str = str(diagnosis).lower()
            # 解析不同格式的多标签ground truth
            if '[' in diagnosis_str and ']' in diagnosis_str:
                # 格式: [0, 1] 或 "[0, 1]"
                try:
                    import ast
                    parsed = ast.literal_eval(diagnosis_str)
                    if isinstance(parsed, list) and len(parsed) == 2:
                        dep_label, anx_label = parsed
                        if dep_label == 0 and anx_label == 0:
                            y_true_multiclass.append("others")
                        elif dep_label == 0 and anx_label == 1:
                            y_true_multiclass.append("焦虑")
                        elif dep_label == 1 and anx_label == 0:
                            y_true_multiclass.append("抑郁")
                        else:  # dep_label == 1 and anx_label == 1
                            y_true_multiclass.append("mix")
                    else:
                        y_true_multiclass.append("others")
                except:
                    y_true_multiclass.append("others")
            else:
                # 基于文本描述推断4分类标签
                if 'mix' in diagnosis_str:
                    y_true_multiclass.append("mix")
                elif 'depression' in diagnosis_str or 'depres' in diagnosis_str:
                    y_true_multiclass.append("抑郁")
                elif 'anxiety' in diagnosis_str or 'anxie' in diagnosis_str:
                    y_true_multiclass.append("焦虑")
                else:
                    y_true_multiclass.append("others")
    
    y_true_multiclass = np.array(y_true_multiclass)
    
    # 检查数据有效性
    if len(y_true_multiclass) != len(y_pred_multiclass):
        logger.error(f"真实标签和预测标签的长度不匹配: {len(y_true_multiclass)} vs {len(y_pred_multiclass)}")
        return {}
    
    # 获取唯一的标签
    unique_labels = ["抑郁", "焦虑", "mix", "others"]
    
    try:
        # 计算weighted F1-score
        weighted_f1 = f1_score(y_true_multiclass, y_pred_multiclass, labels=unique_labels, average='weighted', zero_division='warn')
        
        # 计算balanced accuracy
        balanced_acc = balanced_accuracy_score(y_true_multiclass, y_pred_multiclass)
        
        # 计算准确率
        accuracy = np.mean(y_true_multiclass == y_pred_multiclass)
        
        # 生成分类报告
        class_report = classification_report(y_true_multiclass, y_pred_multiclass, labels=unique_labels, zero_division='warn')
        
        # 生成混淆矩阵
        cm = confusion_matrix(y_true_multiclass, y_pred_multiclass, labels=unique_labels)
        
        # 计算每个类别的指标
        individual_f1 = f1_score(y_true_multiclass, y_pred_multiclass, labels=unique_labels, average=None, zero_division='warn')
        
        # 确保individual_f1是数组格式
        if isinstance(individual_f1, (int, float)):
            individual_f1 = [individual_f1]
        elif hasattr(individual_f1, 'tolist'):
            individual_f1 = individual_f1.tolist()
        
        # 计算每个类别的样本数
        class_counts = [np.sum(y_true_multiclass == label) for label in unique_labels]
        total_samples = len(y_true_multiclass)
        
        # 计算预测分布
        pred_counts = [np.sum(y_pred_multiclass == label) for label in unique_labels]
        
        metrics = {
            "classification_mode": "multilabel_as_multiclass",
            "accuracy": accuracy,
            "weighted_f1": weighted_f1,
            "balanced_accuracy": balanced_acc,
            "individual_f1": dict(zip(unique_labels, individual_f1)),
            "classification_report": class_report,
            "confusion_matrix": cm,
            "sample_counts": dict(zip(unique_labels, class_counts)),
            "prediction_counts": dict(zip(unique_labels, pred_counts)),
            "total_samples": total_samples,
            "unique_labels": unique_labels
        }
        
        # 记录结果
        logger.info("=== 多标签转4分类模型评估指标 ===")
        logger.info(f"总样本数: {total_samples}")
        logger.info(f"各类别真实样本数: {dict(zip(unique_labels, class_counts))}")
        logger.info(f"各类别预测样本数: {dict(zip(unique_labels, pred_counts))}")
        logger.info(f"准确率 (Accuracy): {accuracy:.4f}")
        logger.info(f"Weighted F1-Score: {weighted_f1:.4f}")
        logger.info(f"Balanced Accuracy: {balanced_acc:.4f}")
        
        logger.info("各类别F1-Score:")
        for label, f1 in zip(unique_labels, individual_f1):
            logger.info(f"  {label}: {f1:.4f}")
        
        logger.info("分类报告:")
        logger.info(f"\n{class_report}")
        
        logger.info("混淆矩阵:")
        logger.info(f"\n{cm}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"计算多标签转4分类评估指标时出错: {e}")
        return {}

# 将多标签结果转换为4分类进行评估
def calculate_multilabel_as_multiclass_evaluation_metrics(valid_data, y_true):
    """将多标签结果转换为4分类问题与OverallDiagnosis进行评估"""
    
    # 将多标签预测结果转换为4分类标签
    y_pred_multiclass = []
    for _, row in valid_data.iterrows():
        depression_label = row.get("Depression_Label", 0)
        anxiety_label = row.get("Anxiety_Label", 0)
        
        # 转换为4分类标签
        if depression_label == 0 and anxiety_label == 0:
            y_pred_multiclass.append("others")
        elif depression_label == 0 and anxiety_label == 1:
            y_pred_multiclass.append("焦虑")
        elif depression_label == 1 and anxiety_label == 0:
            y_pred_multiclass.append("抑郁")
        else:  # depression_label == 1 and anxiety_label == 1
            y_pred_multiclass.append("mix")
    
    y_pred_multiclass = np.array(y_pred_multiclass)
    
    # 获取真实标签（支持OverallDiagnosis和MultilabelDiagnosis）
    y_true_multiclass = []
    for diagnosis in y_true:
        if pd.isna(diagnosis):
            y_true_multiclass.append("Others")
        else:
            diagnosis_str = str(diagnosis)
            diagnosis_lower = diagnosis_str.lower()
            
            # 先检查Mix类别的各种表示方式
            if ('mix' in diagnosis_lower or 
                '混合' in diagnosis_str or 
                '共病' in diagnosis_str or
                'mixed' in diagnosis_lower or
                'comorbid' in diagnosis_lower or
                diagnosis_str == 'Mix' or
                ('depression' in diagnosis_lower and 'anxiety' in diagnosis_lower)):
                y_true_multiclass.append("Mix")
            # 然后检查单一的Depression
            elif ('depression' in diagnosis_lower and 'anxiety' not in diagnosis_lower) or 'depres' in diagnosis_lower:
                y_true_multiclass.append("Depression")
            # 然后检查单一的Anxiety
            elif ('anxiety' in diagnosis_lower and 'depression' not in diagnosis_lower) or 'anxie' in diagnosis_lower:
                y_true_multiclass.append("Anxiety")
            else:
                y_true_multiclass.append("Others")
    
    y_true_multiclass = np.array(y_true_multiclass)
    
    # 检查数据有效性
    if len(y_true_multiclass) != len(y_pred_multiclass):
        logger.error(f"真实标签和预测标签的长度不匹配: {len(y_true_multiclass)} vs {len(y_pred_multiclass)}")
        return {}
    
    # 获取唯一的标签
    unique_labels = ["Depression", "Anxiety", "Mix", "Others"]
    
    try:
        # 计算weighted F1-score
        weighted_f1 = f1_score(y_true_multiclass, y_pred_multiclass, labels=unique_labels, average='weighted', zero_division='warn')
        
        # 计算balanced accuracy
        balanced_acc = balanced_accuracy_score(y_true_multiclass, y_pred_multiclass)
        
        # 计算准确率
        accuracy = np.mean(y_true_multiclass == y_pred_multiclass)
        
        # 生成分类报告
        class_report = classification_report(y_true_multiclass, y_pred_multiclass, labels=unique_labels, zero_division='warn')
        
        # 生成混淆矩阵
        cm = confusion_matrix(y_true_multiclass, y_pred_multiclass, labels=unique_labels)
        
        # 计算每个类别的指标
        individual_f1 = f1_score(y_true_multiclass, y_pred_multiclass, labels=unique_labels, average=None, zero_division=0)
        
        # 确保individual_f1是数组格式
        if isinstance(individual_f1, (int, float)):
            individual_f1 = [individual_f1]
        elif hasattr(individual_f1, 'tolist'):
            individual_f1 = individual_f1.tolist()
        
        # 计算每个类别的样本数
        class_counts = [np.sum(y_true_multiclass == label) for label in unique_labels]
        total_samples = len(y_true_multiclass)
        
        # 计算预测分布
        pred_counts = [np.sum(y_pred_multiclass == label) for label in unique_labels]
        
        metrics = {
            "classification_mode": "multilabel_as_multiclass",
            "accuracy": accuracy,
            "weighted_f1": weighted_f1,
            "balanced_accuracy": balanced_acc,
            "individual_f1": dict(zip(unique_labels, individual_f1)),
            "classification_report": class_report,
            "confusion_matrix": cm,
            "sample_counts": dict(zip(unique_labels, class_counts)),
            "prediction_counts": dict(zip(unique_labels, pred_counts)),
            "total_samples": total_samples,
            "unique_labels": unique_labels
        }
        
        # 记录结果
        logger.info("=== 多标签转4分类模型评估指标 ===")
        logger.info(f"总样本数: {total_samples}")
        logger.info(f"各类别真实样本数: {dict(zip(unique_labels, class_counts))}")
        logger.info(f"各类别预测样本数: {dict(zip(unique_labels, pred_counts))}")
        logger.info(f"准确率 (Accuracy): {accuracy:.4f}")
        logger.info(f"Weighted F1-Score: {weighted_f1:.4f}")
        logger.info(f"Balanced Accuracy: {balanced_acc:.4f}")
        
        logger.info("各类别F1-Score:")
        for label, f1 in zip(unique_labels, individual_f1):
            logger.info(f"  {label}: {f1:.4f}")
        
        logger.info("分类报告:")
        logger.info(f"\n{class_report}")
        
        logger.info("混淆矩阵:")
        logger.info(f"\n{cm}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"计算多标签转4分类评估指标时出错: {e}")
        return {}

def calculate_icd10_evaluation_metrics(valid_data, y_true):
    """计算ICD-10分类评估指标（大类和小类），支持多个ground truth代码"""
    try:
        import re
        
        # 检查必要列是否存在
        required_columns = ["Predicted_ICD10", "Ground_Truth_ICD10"]
        missing_cols = [col for col in required_columns if col not in valid_data.columns]
        if missing_cols:
            logger.error(f"缺少必要的列: {missing_cols}")
            return {
                "error": True,
                "message": f"Missing required columns: {missing_cols}"
            }
        
        # 获取预测和真实值
        predicted_codes = valid_data["Predicted_ICD10"].values
        ground_truth_codes = valid_data["Ground_Truth_ICD10"].values
        
        # 提取大类和小类的预测和真实值
        predicted_major = []
        predicted_minor = []
        ground_truth_major = []
        ground_truth_minor = []
        
        # 处理所有预测和真实值，支持多个ground truth代码
        valid_indices = []
        for i in range(len(predicted_codes)):
            pred_major, pred_minor = ICD10Utils.extract_major_minor(predicted_codes[i])
            
            # ground_truth_codes[i] 现在是一个代码列表
            gt_codes_list = ground_truth_codes[i]
            if not isinstance(gt_codes_list, list):
                # 如果不是列表，尝试转换为列表
                if pd.isna(gt_codes_list) or gt_codes_list is None:
                    gt_codes_list = []
                else:
                    gt_codes_list = [gt_codes_list]
            
            # 检查预测是否与任一ground truth代码匹配
            best_match_major = None
            best_match_minor = None
            
            # 如果有预测代码，找到最佳匹配的ground truth代码
            if pred_major is not None and gt_codes_list:
                for gt_code in gt_codes_list:
                    gt_major, gt_minor = ICD10Utils.extract_major_minor(gt_code)
                    if gt_major is not None:
                        # 如果预测的大类匹配，使用这个ground truth
                        if pred_major == gt_major:
                            best_match_major = gt_major
                            best_match_minor = gt_minor
                            break
                        # 如果还没找到匹配，至少记录第一个有效的ground truth
                        elif best_match_major is None:
                            best_match_major = gt_major
                            best_match_minor = gt_minor
            
            # 只包含有效的预测和至少一个有效的ground truth
            if pred_major is not None and best_match_major is not None:
                predicted_major.append(pred_major)
                predicted_minor.append(pred_minor)
                ground_truth_major.append(best_match_major)
                ground_truth_minor.append(best_match_minor)
                valid_indices.append(i)
        
        if len(predicted_major) == 0:
            return {
                "error": True,
                "message": "No valid ICD-10 predictions found"
            }
        
        logger.info(f"ICD-10评估：有效样本数 {len(predicted_major)}/{len(predicted_codes)}")
        
        # 计算大类准确性指标
        logger.info("计算大类准确性指标...")
        major_weighted_f1 = f1_score(ground_truth_major, predicted_major, average='weighted', zero_division=0)
        major_balanced_acc = balanced_accuracy_score(ground_truth_major, predicted_major)
        
        # 计算大类分类报告和混淆矩阵
        major_unique_labels = sorted(list(set(ground_truth_major + predicted_major)))
        major_class_report = classification_report(
            ground_truth_major, predicted_major, 
            labels=major_unique_labels, zero_division=0
        )
        major_cm = confusion_matrix(ground_truth_major, predicted_major, labels=major_unique_labels)
        
        # 计算小类准确性指标
        logger.info("计算小类准确性指标...")
        minor_weighted_f1 = f1_score(ground_truth_minor, predicted_minor, average='weighted', zero_division=0)
        minor_balanced_acc = balanced_accuracy_score(ground_truth_minor, predicted_minor)
        
        # 计算小类分类报告和混淆矩阵
        minor_unique_labels = sorted(list(set(ground_truth_minor + predicted_minor)))
        minor_class_report = classification_report(
            ground_truth_minor, predicted_minor, 
            labels=minor_unique_labels, zero_division=0
        )
        minor_cm = confusion_matrix(ground_truth_minor, predicted_minor, labels=minor_unique_labels)
        
        # 计算各类别F1分数
        major_individual_f1 = {}
        for label in major_unique_labels:
            f1 = f1_score(ground_truth_major, predicted_major, labels=[label], average='weighted', zero_division=0)
            major_individual_f1[label] = f1
        
        minor_individual_f1 = {}
        for label in minor_unique_labels:
            f1 = f1_score(ground_truth_minor, predicted_minor, labels=[label], average='weighted', zero_division=0)
            minor_individual_f1[label] = f1
        
        logger.info(f"大类准确性 - Weighted F1: {major_weighted_f1:.4f}, Balanced Accuracy: {major_balanced_acc:.4f}")
        logger.info(f"小类准确性 - Weighted F1: {minor_weighted_f1:.4f}, Balanced Accuracy: {minor_balanced_acc:.4f}")
        
        return {
            "classification_mode": "icd10",
            "sample_count": len(predicted_major),
            
            # 大类准确性指标
            "major_class_weighted_f1": major_weighted_f1,
            "major_class_balanced_accuracy": major_balanced_acc,
            "major_class_individual_f1": major_individual_f1,
            "major_class_classification_report": major_class_report,
            "major_class_confusion_matrix": major_cm,
            "major_class_unique_labels": major_unique_labels,
            
            # 小类准确性指标
            "minor_class_weighted_f1": minor_weighted_f1,
            "minor_class_balanced_accuracy": minor_balanced_acc,
            "minor_class_individual_f1": minor_individual_f1,
            "minor_class_classification_report": minor_class_report,
            "minor_class_confusion_matrix": minor_cm,
            "minor_class_unique_labels": minor_unique_labels,
            
            # 为兼容性保留的字段
            "weighted_f1": major_weighted_f1,  # 默认使用大类
            "balanced_accuracy": major_balanced_acc,
            "individual_f1": major_individual_f1,
            "classification_report": major_class_report,
            "confusion_matrix": major_cm,
            "unique_labels": major_unique_labels,
        }
        
    except Exception as e:
        logger.error(f"计算ICD-10评估指标时出错: {e}")
        return {
            "error": True,
            "message": str(e)
        }

def calculate_recommendation_evaluation_metrics(valid_data, y_true):
    """计算推荐模式的评估指标：限制为11种特定疾病，计算precision, recall, f1 score，过滤ground truth不在11类中的样本"""
    try:
        import ast
        from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
        
        # 使用统一的ICD-10工具类
        
        # 检查必要列是否存在
        required_columns = ["Recommended_ICD10_Codes", "Ground_Truth_ICD10", "Top1_Code", "Top3_Codes"]
        missing_cols = [col for col in required_columns if col not in valid_data.columns]
        if missing_cols:
            logger.error(f"缺少必要的列: {missing_cols}")
            return {
                "error": True,
                "message": f"Missing required columns: {missing_cols}"
            }
        
        # 获取推荐和真实值
        recommended_codes_raw = valid_data["Recommended_ICD10_Codes"].values
        ground_truth_codes = valid_data["Ground_Truth_ICD10"].values
        top1_codes = valid_data["Top1_Code"].values
        top3_codes_raw = valid_data["Top3_Codes"].values
        
        # 解析推荐代码和Top3代码
        recommended_codes_list = []
        top3_codes_list = []
        
        for i in range(len(recommended_codes_raw)):
            # 解析推荐代码
            try:
                rec_codes = ast.literal_eval(recommended_codes_raw[i]) if isinstance(recommended_codes_raw[i], str) else recommended_codes_raw[i]
                if not isinstance(rec_codes, list):
                    rec_codes = [rec_codes] if rec_codes else []
                recommended_codes_list.append(rec_codes)
            except:
                recommended_codes_list.append([])
            
            # 解析Top3代码
            try:
                top3_codes = ast.literal_eval(top3_codes_raw[i]) if isinstance(top3_codes_raw[i], str) else top3_codes_raw[i]
                if not isinstance(top3_codes, list):
                    top3_codes = [top3_codes] if top3_codes else []
                top3_codes_list.append(top3_codes)
            except:
                top3_codes_list.append([])
        
        # 提取并过滤大类代码（只保留11种允许的疾病）
        def extract_and_filter_major_codes(codes_list):
            """从代码列表中提取大类代码，并过滤为11种允许的疾病"""
            return ICD10Utils.extract_major_classes_from_list(codes_list)
        
        # 计算各种指标
        top1_correct = 0
        top3_correct = 0
        exact_match = 0
        valid_samples = 0
        
        # 用于计算precision, recall, f1的数据
        all_gt_labels = []  # 所有ground truth标签
        all_pred_labels = []  # 所有预测标签（基于推荐代码）
        all_top1_labels = []  # 所有Top1预测标签
        
        # 首先过滤样本：只保留ground truth中至少有一个代码在11类中的样本
        filtered_indices = []
        for i in range(len(ground_truth_codes)):
            gt_major_codes = extract_and_filter_major_codes(ground_truth_codes[i])
            if gt_major_codes:  # 只保留有有效ground truth的样本
                filtered_indices.append(i)
        
        if not filtered_indices:
            logger.warning("没有样本的ground truth在11类疾病中，无法计算评估指标")
            return {
                "error": True,
                "message": "No samples with ground truth in the 11 allowed disease categories"
            }
        
        logger.info(f"过滤前样本数: {len(ground_truth_codes)}, 过滤后样本数: {len(filtered_indices)}")
        
        for idx in filtered_indices:
            gt_major_codes = extract_and_filter_major_codes(ground_truth_codes[idx])
                
            valid_samples += 1
            gt_major_codes_set = set(gt_major_codes)
            
            # 处理推荐代码（过滤为11种疾病）
            rec_major_codes = extract_and_filter_major_codes(recommended_codes_list[idx])
            rec_major_codes_set = set(rec_major_codes)
            
            # 处理Top1代码
            top1_code = top1_codes[idx]
            top1_filtered = None
            if top1_code:
                import re
                major_match = re.match(r'(F\d+|Z71)', str(top1_code).strip())
                if major_match and major_match.group(1) in ICD10Utils.ALLOWED_DISEASES:
                    top1_filtered = major_match.group(1)
            
            # 处理Top3代码
            top3_list = top3_codes_list[idx]
            top3_filtered = []
            for code in top3_list:
                if code:
                    import re
                    major_match = re.match(r'(F\d+|Z71)', str(code).strip())
                    if major_match and major_match.group(1) in ICD10Utils.ALLOWED_DISEASES:
                        top3_filtered.append(major_match.group(1))
            top3_filtered_set = set(top3_filtered)
            
            # Top1 Accuracy
            if top1_filtered and top1_filtered in gt_major_codes_set:
                top1_correct += 1
            
            # Top3 Accuracy
            if any(code in gt_major_codes_set for code in top3_filtered):
                top3_correct += 1
            
            # Exact Match
            if rec_major_codes_set == gt_major_codes_set:
                exact_match += 1
            
            # 为每种疾病创建二进制标签
            for disease in ICD10Utils.ALLOWED_DISEASES:
                # Ground truth标签
                gt_label = 1 if disease in gt_major_codes_set else 0
                all_gt_labels.append(gt_label)
                
                # 推荐标签
                pred_label = 1 if disease in rec_major_codes_set else 0
                all_pred_labels.append(pred_label)
                
                # Top1标签
                top1_label = 1 if disease == top1_filtered else 0
                all_top1_labels.append(top1_label)
        
        if valid_samples == 0:
            return {
                "error": True,
                "message": "No valid samples found for recommendation evaluation"
            }
        
        # 计算基本指标
        top1_accuracy = top1_correct / valid_samples
        top3_accuracy = top3_correct / valid_samples
        exact_match_ratio = exact_match / valid_samples
        
        # 计算precision, recall, f1 score
        precision = precision_score(all_gt_labels, all_pred_labels, average='weighted', zero_division=0)
        recall = recall_score(all_gt_labels, all_pred_labels, average='weighted', zero_division=0)
        f1 = f1_score(all_gt_labels, all_pred_labels, average='weighted', zero_division=0)
        
        # 计算macro平均
        precision_macro = precision_score(all_gt_labels, all_pred_labels, average='macro', zero_division=0)
        recall_macro = recall_score(all_gt_labels, all_pred_labels, average='macro', zero_division=0)
        f1_macro = f1_score(all_gt_labels, all_pred_labels, average='macro', zero_division=0)
        
        # 计算Top1的precision, recall, f1
        top1_precision = precision_score(all_gt_labels, all_top1_labels, average='weighted', zero_division=0)
        top1_recall = recall_score(all_gt_labels, all_top1_labels, average='weighted', zero_division=0)
        top1_f1 = f1_score(all_gt_labels, all_top1_labels, average='weighted', zero_division=0)
        
        logger.info("=== 推荐模式评估指标（限制11种疾病）===")
        logger.info(f"有效样本数: {valid_samples}")
        logger.info(f"Top1 Accuracy: {top1_accuracy:.4f}")
        logger.info(f"Top3 Accuracy: {top3_accuracy:.4f}")
        logger.info(f"Exact Match: {exact_match_ratio:.4f}")
        logger.info(f"Precision (weighted): {precision:.4f}")
        logger.info(f"Recall (weighted): {recall:.4f}")
        logger.info(f"F1 Score (weighted): {f1:.4f}")
        logger.info(f"Precision (macro): {precision_macro:.4f}")
        logger.info(f"Recall (macro): {recall_macro:.4f}")
        logger.info(f"F1 Score (macro): {f1_macro:.4f}")
        logger.info(f"Top1 Precision: {top1_precision:.4f}")
        logger.info(f"Top1 Recall: {top1_recall:.4f}")
        logger.info(f"Top1 F1: {top1_f1:.4f}")
        
        return {
            "classification_mode": "recommendation",
            "sample_count": valid_samples,
            "allowed_diseases": list(ICD10Utils.ALLOWED_DISEASES),
            "top1_accuracy": top1_accuracy,
            "top3_accuracy": top3_accuracy,
            "exact_match": exact_match_ratio,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "top1_precision": top1_precision,
            "top1_recall": top1_recall,
            "top1_f1": top1_f1,
            
            # 为兼容性保留的字段
            "weighted_f1": f1,
            "balanced_accuracy": top1_accuracy,
        }
        
    except Exception as e:
        logger.error(f"计算推荐模式评估指标时出错: {e}")
        return {
            "error": True,
            "message": str(e)
        }

# 可视化评估结果
def visualize_evaluation_results(evaluation_metrics, output_prefix, results_dir="./results"):
    """可视化评估结果，包括混淆矩阵和指标对比"""
    if not evaluation_metrics:
        return
    
    # 检查是否有错误
    if "error" in evaluation_metrics:
        logger.warning(f"评估出现错误，跳过可视化: {evaluation_metrics.get('message', 'Unknown error')}")
        return
    
    # 确保输出目录存在
    visualization_dir = os.path.join(results_dir, "visualization")
    os.makedirs(visualization_dir, exist_ok=True)
    
    # 设置Seaborn风格
    sns.set(style="whitegrid")
    
    # 获取分类模式和标签
    classification_mode = evaluation_metrics.get("classification_mode", "binary")
    unique_labels = evaluation_metrics.get("unique_labels", ["Depression", "Anxiety"])
    
    # 处理multilabel_as_multiclass模式
    if classification_mode == "multilabel_as_multiclass":
        classification_mode = "multiclass"  # 使用multiclass的可视化方式
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 混淆矩阵热图
    if "confusion_matrix" in evaluation_metrics:
        cm = evaluation_metrics["confusion_matrix"]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=unique_labels, yticklabels=unique_labels,
                   ax=axes[0, 0])
        axes[0, 0].set_title(f'Confusion Matrix ({classification_mode.title()})')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('True')
    else:
        axes[0, 0].text(0.5, 0.5, 'No Confusion Matrix\nAvailable', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Confusion Matrix (Unavailable)')
    
    # 2. 各类别F1-Score对比
    if "individual_f1" in evaluation_metrics:
        f1_scores = evaluation_metrics["individual_f1"]
        labels = list(f1_scores.keys())
        values = list(f1_scores.values())
        
        # 为不同类别设置不同颜色
        if classification_mode == "binary":
            colors = ['skyblue', 'lightcoral']
        else:  # multiclass
            colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightyellow']
        
        bars = axes[0, 1].bar(labels, values, color=colors[:len(labels)])
        axes[0, 1].set_title(f'F1-Score by Class ({classification_mode.title()})')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_ylim(0, 1)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
    else:
        axes[0, 1].text(0.5, 0.5, 'No F1-Score Data\nAvailable', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('F1-Score by Class (Unavailable)')
    
    # 3. 各类别AUPRC对比
    if "individual_auprc" in evaluation_metrics:
        auprc_scores = evaluation_metrics["individual_auprc"]
        labels = list(auprc_scores.keys())
        values = list(auprc_scores.values())
        
        # 为不同类别设置不同颜色
        if classification_mode == "binary":
            colors = ['gold', 'orange']
        else:  # multiclass
            colors = ['gold', 'orange', 'lightpink', 'lightblue']
        
        bars = axes[1, 0].bar(labels, values, color=colors[:len(labels)])
        axes[1, 0].set_title(f'AUPRC by Class ({classification_mode.title()})')
        axes[1, 0].set_ylabel('AUPRC')
        axes[1, 0].set_ylim(0, 1)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
    else:
        axes[1, 0].text(0.5, 0.5, 'No AUPRC Data\nAvailable', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('AUPRC by Class (Unavailable)')
    
    # 4. 整体指标对比
    overall_metrics = {
        'Weighted F1': evaluation_metrics.get("weighted_f1", 0),
        'Balanced Acc': evaluation_metrics.get("balanced_accuracy", 0),
        'Macro AUPRC': evaluation_metrics.get("macro_auprc", 0),
        'Weighted AUPRC': evaluation_metrics.get("weighted_auprc", 0)
    }
    
    metric_names = list(overall_metrics.keys())
    metric_values = list(overall_metrics.values())
    
    bars = axes[1, 1].bar(metric_names, metric_values, 
                         color=['steelblue', 'darkorange', 'green', 'purple'])
    axes[1, 1].set_title(f'Overall Performance Metrics ({classification_mode.title()})')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars, metric_values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{visualization_dir}/{output_prefix}_evaluation_results.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"评估结果可视化图表已保存至: {visualization_dir}/{output_prefix}_evaluation_results.png")

# 主函数
async def main():
    # 加载环境变量
    load_dotenv()
    
    args = parse_args()
    
    # 从模型路径中提取basename
    model_basename = os.path.basename(args.model.rstrip('/'))
    
    # 确保日志目录存在
    os.makedirs(args.logs_dir, exist_ok=True)
    
    # 设置包含模型basename和分类模式的日志文件名
    log_filename = f"{args.logs_dir}/{current_date}_{model_basename}_{args.classification_mode}_{current_time}.log"
    
    # 重新配置日志
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True  # 强制重新配置，覆盖之前的配置
    )
    
    logger.info(f"日志文件: {log_filename}")
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # 如果指定了evaluate_only，直接进行评估
    if args.evaluate_only:
        if not os.path.exists(args.evaluate_only):
            logger.error(f"错误：评估文件 {args.evaluate_only} 不存在")
            return
            
        try:
            logger.info(f"正在读取评估文件: {args.evaluate_only}")
            results_df = pd.read_excel(args.evaluate_only)
            logger.info(f"成功读取文件，共 {len(results_df)} 条记录")
            
            # 检查必要的列是否存在
            if args.classification_mode in ["binary", "multiclass"]:
                required_cols = ["OverallDiagnosis"]
                if args.classification_mode == "binary":
                    required_cols.extend(["Normalized_Depression_Prob", "Normalized_Anxiety_Prob"])
                else:  # multiclass
                    required_cols.extend([
                        "Normalized_Depression_Prob", "Normalized_Anxiety_Prob",
                        "Normalized_Mix_Prob", "Normalized_Others_Prob"
                    ])
            elif args.classification_mode == "depression_symptom":
                required_cols = ["DepressionDiagnosis", "Normalized_Has_Depression_Symptom_Prob", "Normalized_No_Depression_Symptom_Prob"]
            elif args.classification_mode == "anxiety_symptom":
                required_cols = ["AnxietyDiagnosis", "Normalized_Has_Anxiety_Symptom_Prob", "Normalized_No_Anxiety_Symptom_Prob"]
            elif args.classification_mode == "multilabel":
                required_cols = ["Depression_Probability", "Anxiety_Probability", "Depression_Label", "Anxiety_Label"]
            elif args.classification_mode == "icd10":
                required_cols = ["Predicted_ICD10", "Ground_Truth_ICD10", "Predicted_Major_Class", "Predicted_Minor_Class"]
            elif args.classification_mode == "recommendation":
                required_cols = ["Recommended_ICD10_Codes", "Ground_Truth_ICD10", "Top1_Code", "Top3_Codes"]
            else:
                logger.error(f"不支持的分类模式: {args.classification_mode}")
                return
            
            missing_cols = [col for col in required_cols if col not in results_df.columns]
            if missing_cols:
                logger.error(f"错误：评估文件缺少必要的列: {missing_cols}")
                return
            
            # 计算评估指标
            evaluation_metrics = calculate_evaluation_metrics(results_df, args.classification_mode)
            
            if evaluation_metrics:
                # 生成评估结果可视化图表
                output_prefix = os.path.splitext(os.path.basename(args.evaluate_only))[0]
                visualize_evaluation_results(evaluation_metrics, output_prefix, args.results_dir)
                
                # 保存评估指标到单独的Excel文件
                metrics_file = args.evaluate_only.replace('.xlsx', '_evaluation_metrics.xlsx')
                with pd.ExcelWriter(metrics_file) as writer:
                    # 创建评估指标的DataFrame
                    if args.classification_mode in ["binary", "multiclass"]:
                        unique_labels = evaluation_metrics.get("unique_labels", ["Depression", "Anxiety"])
                    elif args.classification_mode in ["depression_symptom", "anxiety_symptom"]:
                        unique_labels = evaluation_metrics.get("unique_labels", ["有", "没有"])
                    else:
                        unique_labels = evaluation_metrics.get("unique_labels", [])
                    
                    metrics_data = {
                        "Metric": ["Weighted F1-Score", "Balanced Accuracy", "Macro AUPRC", "Weighted AUPRC"],
                        "Value": [
                            evaluation_metrics["weighted_f1"],
                            evaluation_metrics["balanced_accuracy"], 
                            evaluation_metrics["macro_auprc"],
                            evaluation_metrics["weighted_auprc"]
                        ]
                    }
                    
                    # 添加各类别的F1和AUPRC
                    for label in unique_labels:
                        if label in evaluation_metrics["individual_f1"]:
                            metrics_data["Metric"].append(f"{label} F1-Score")
                            metrics_data["Value"].append(evaluation_metrics["individual_f1"][label])
                        if label in evaluation_metrics["individual_auprc"]:
                            metrics_data["Metric"].append(f"{label} AUPRC")
                            metrics_data["Value"].append(evaluation_metrics["individual_auprc"][label])
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                    
                    # 保存分类报告
                    if "classification_report" in evaluation_metrics:
                        report_lines = evaluation_metrics["classification_report"].split('\n')
                        report_data = []
                        for line in report_lines:
                            if line.strip():
                                report_data.append([line])
                        report_df = pd.DataFrame(report_data)
                        if len(report_df.columns) > 0:
                            report_df.columns = ['Classification_Report']
                        report_df.to_excel(writer, sheet_name='Classification_Report', index=False)
                    
                    # 保存混淆矩阵
                    if "confusion_matrix" in evaluation_metrics:
                        cm_df = pd.DataFrame(
                            evaluation_metrics["confusion_matrix"],
                            index=unique_labels,
                            columns=unique_labels
                        )
                        cm_df.to_excel(writer, sheet_name='Confusion_Matrix')
                
                logger.info(f"评估指标已保存至: {metrics_file}")
            
            return
            
        except Exception as e:
            logger.error(f"评估失败: {e}")
            return
    
    OPENAI_API_BASE = args.api
    INPUT_FILE = args.input
    # 确保results目录存在，并构建完整的输出文件路径
    os.makedirs(args.results_dir, exist_ok=True)
    if os.path.isabs(args.output):
        OUTPUT_FILE = args.output  # 如果是绝对路径，直接使用
    else:
        OUTPUT_FILE = os.path.join(args.results_dir, args.output)  # 相对路径，拼接到results_dir
    MODEL_NAME = args.model
    MAX_CONCURRENT = args.max_concurrent
    
    # 从环境变量读取API key
    API_KEY = os.getenv('OPENROUTER_API_KEY')
    if not API_KEY:
        logger.error("错误：未找到OPENROUTER_API_KEY环境变量，请在.env文件中设置或通过环境变量提供")
        return
    
    logger.info(f"配置参数:")
    logger.info(f"  OpenRouter API base URL: {OPENAI_API_BASE}")
    logger.info(f"  OpenRouter API key: {API_KEY[:8] if API_KEY else 'None'}...{API_KEY[-4:] if API_KEY and len(API_KEY) > 12 else '*' * 4}")
    logger.info(f"  输入文件: {INPUT_FILE}")
    logger.info(f"  输出文件: {OUTPUT_FILE}")
    logger.info(f"  模型: {MODEL_NAME}")
    logger.info(f"  最大并发数: {MAX_CONCURRENT}")
    logger.info(f"  Top logprobs: {args.top_logprobs}")
    logger.info(f"  站点URL: {args.site_url}")
    logger.info(f"  站点名称: {args.site_name}")
    logger.info(f"  调试模式: {args.debug}")
    logger.info(f"  可视化: {args.visualize}")
    logger.info(f"  模型评估: {args.evaluate}")
    logger.info(f"  分类模式: {args.classification_mode}")
    logger.info(f"  保存JSON日志: {args.save_json_logs}")
    
    # 初始化JSON日志数据
    global json_log_data
    json_log_data = []
    
    # 检测API类型
    api_type = detect_api_type(OPENAI_API_BASE)
    logger.info(f"检测到API类型: {api_type}")
    
    # 创建OpenAI客户端（支持OpenRouter和本地vLLM）
    client = AsyncOpenAI(
        api_key=API_KEY,
        base_url=OPENAI_API_BASE,
    )
    
    # 加载tokenizer
    if not load_tokenizer(MODEL_NAME):
        logger.warning("无法加载tokenizer，将使用原始token进行匹配")
    
    # 检查文件是否存在
    if not os.path.exists(INPUT_FILE):
        logger.error(f"错误：文件 {INPUT_FILE} 不存在")
        return
    
    # 读取Excel文件
    try:
        logger.info(f"正在读取Excel文件...")
        df = pd.read_excel(INPUT_FILE)
        logger.info(f"成功读取文件，共 {len(df)} 条记录")
        
        # 使用平衡采样策略选择样本
        if args.limit and args.limit > 0 and args.limit < len(df):
            
            
            # logger.info(f"限制处理数量为前 {args.limit} 条记录")
            # # 随机选择args.limit条记录
            # df = df.sample(n=args.limit, random_state=42).reset_index(drop=True)
            
            logger.info(f"=== 开始平衡采样 ===")
            logger.info(f"目标样本数: {args.limit}")
            logger.info(f"总可用样本数: {len(df)}")
            
            # 使用平衡采样函数
            sampled_df, sampling_info = balanced_sampling_with_resume(
                df=df, 
                classification_mode=args.classification_mode,
                target_sample_count=args.limit,
                processed_visits=None,  # 这里暂时为None，后面会在resume逻辑中处理
                min_samples_per_class=args.min_samples_per_class,
                random_state=42
            )
            
            if len(sampled_df) > 0:
                df = sampled_df.reset_index(drop=True)
                logger.info(f"平衡采样完成，实际选择 {len(df)} 个样本")
                logger.info(f"采样信息: {sampling_info}")
            else:
                logger.warning("平衡采样未选择到任何样本，使用原始数据")
        else:
            logger.info("未设置样本限制或样本数不足，使用全部数据")
    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        return
    
    # Resume功能：检查是否存在之前的JSON日志文件（优先使用批次文件）
    output_prefix = os.path.splitext(os.path.basename(OUTPUT_FILE))[0]
    existing_json_file, previous_json_data = find_and_merge_batch_json_logs(output_prefix, MODEL_NAME, args.classification_mode, args.results_dir)
    
    processed_visits = set()
    previous_results_df = pd.DataFrame()
    
    print("------------前序处理文件检测----------------")
    print(f"existing_json_file: {existing_json_file}")
    
    if existing_json_file and previous_json_data:
        logger.info(f"=== Resume功能启用 ===")
        logger.info(f"发现已存在的JSON日志文件: {existing_json_file}")
        
        # 提取已处理的VisitNumber
        processed_visits = extract_processed_visit_numbers(previous_json_data)
        logger.info(f"已处理的样本数量: {len(processed_visits)}")
        
        # 从JSON数据重建之前的结果
        previous_results_df = load_previous_results_from_json(previous_json_data, args.classification_mode)
        
        # 将之前的JSON数据加载到当前的json_log_data中
        json_log_data.extend(previous_json_data)
        
        # 计算需要处理的剩余样本
        target_sample_count = args.limit if args.limit and args.limit > 0 else len(df)
        
        # 使用平衡采样检查是否需要补充样本
        logger.info(f"=== Resume模式下的平衡采样检查 ===")
        additional_sampled_df, sampling_info = balanced_sampling_with_resume(
            df=df, 
            classification_mode=args.classification_mode,
            target_sample_count=target_sample_count,
            processed_visits=processed_visits,
            min_samples_per_class=args.min_samples_per_class,
            random_state=42
        )
        
        if len(additional_sampled_df) == 0:
            logger.info(f"平衡采样检查完成，无需额外样本")
            logger.info(f"已处理样本数: {len(processed_visits)}, 目标样本数: {target_sample_count}")
            logger.info("直接使用已有结果进行评估...")
            
            # 使用之前的结果和目标结果的交集
            results_df = previous_results_df[previous_results_df["VisitNumber"].isin(df["VisitNumber"])]
            
            # 保存结果
            try:
                # 添加日期前缀
                date_prefix = datetime.now().strftime("%Y%m%d")
                basename = os.path.basename(OUTPUT_FILE)
                new_output_file = os.path.join(os.path.dirname(OUTPUT_FILE), date_prefix + "_" + basename)
                
                results_df.to_excel(new_output_file, index=False)
                logger.info(f"结果已保存至: {new_output_file}")
            except Exception as e:
                logger.error(f"保存结果失败: {e}")
            
            # Resume模式下的评估指标计算
            if args.evaluate and len(results_df) > 0:
                logger.info("=== 开始模型评估 ===")
                
                # 准备评估数据
                evaluation_df = results_df.copy()
                should_evaluate = True
                
                # 根据分类模式检查必要的列
                if args.classification_mode == "binary" and "OverallDiagnosis" not in evaluation_df.columns:
                    logger.warning("二分类模式但缺少OverallDiagnosis列，跳过评估")
                    should_evaluate = False
                elif args.classification_mode == "multiclass" and "OverallDiagnosis" not in evaluation_df.columns:
                    logger.warning("多分类模式但缺少OverallDiagnosis列，跳过评估")
                    should_evaluate = False
                elif args.classification_mode == "depression_symptom" and "DepressionDiagnosis" not in evaluation_df.columns:
                    logger.warning("抑郁症状模式但缺少DepressionDiagnosis列，跳过评估")
                    should_evaluate = False
                elif args.classification_mode == "anxiety_symptom" and "AnxietyDiagnosis" not in evaluation_df.columns:
                    logger.warning("焦虑症状模式但缺少AnxietyDiagnosis列，跳过评估")
                    should_evaluate = False
                elif args.classification_mode == "multilabel" and "MultilabelDiagnosis" not in evaluation_df.columns:
                    logger.warning("多标签模式但缺少MultilabelDiagnosis列，跳过评估")
                    should_evaluate = False
                elif args.classification_mode in ["icd10", "recommendation"] and "DiagnosisCode" not in evaluation_df.columns:
                    logger.warning(f"{args.classification_mode}模式但缺少DiagnosisCode列，跳过评估")
                    should_evaluate = False
                
                # 计算评估指标
                evaluation_metrics = {}
                if should_evaluate:
                    logger.info(f"用于评估的样本数: {len(evaluation_df)}")
                    evaluation_metrics = calculate_evaluation_metrics(evaluation_df, args.classification_mode)
                
                # 如果有评估指标，保存到单独的文件
                if evaluation_metrics and should_evaluate:
                    try:
                        # 保存评估指标到单独的Excel文件
                        metrics_file = new_output_file.replace('.xlsx', '_evaluation_metrics.xlsx')
                        with pd.ExcelWriter(metrics_file) as writer:
                            
                            if args.classification_mode == "icd10":
                                # ICD-10模式：保存大类和小类的指标
                                # 大类指标
                                major_metrics_data = {
                                    "Metric": ["Major Class Weighted F1-Score", "Major Class Balanced Accuracy"],
                                    "Value": [
                                        evaluation_metrics.get("major_class_weighted_f1", 0),
                                        evaluation_metrics.get("major_class_balanced_accuracy", 0)
                                    ]
                                }
                                
                                # 小类指标
                                minor_metrics_data = {
                                    "Metric": ["Minor Class Weighted F1-Score", "Minor Class Balanced Accuracy"],
                                    "Value": [
                                        evaluation_metrics.get("minor_class_weighted_f1", 0),
                                        evaluation_metrics.get("minor_class_balanced_accuracy", 0)
                                    ]
                                }
                                
                                # 合并指标
                                combined_metrics_data = {
                                    "Metric": major_metrics_data["Metric"] + minor_metrics_data["Metric"],
                                    "Value": major_metrics_data["Value"] + minor_metrics_data["Value"]
                                }
                                
                                # 添加各大类的F1分数
                                major_unique_labels = evaluation_metrics.get("major_class_unique_labels", [])
                                major_individual_f1 = evaluation_metrics.get("major_class_individual_f1", {})
                                for label in major_unique_labels:
                                    if label in major_individual_f1:
                                        combined_metrics_data["Metric"].append(f"Major Class {label} F1-Score")
                                        combined_metrics_data["Value"].append(major_individual_f1[label])
                                
                                # 添加各小类的F1分数
                                minor_unique_labels = evaluation_metrics.get("minor_class_unique_labels", [])
                                minor_individual_f1 = evaluation_metrics.get("minor_class_individual_f1", {})
                                for label in minor_unique_labels:
                                    if label in minor_individual_f1:
                                        combined_metrics_data["Metric"].append(f"Minor Class {label} F1-Score")
                                        combined_metrics_data["Value"].append(minor_individual_f1[label])
                                
                                metrics_df = pd.DataFrame(combined_metrics_data)
                                metrics_df.to_excel(writer, sheet_name='ICD10_Metrics', index=False)
                                
                                # 保存大类分类报告
                                if "major_class_classification_report" in evaluation_metrics:
                                    report_lines = evaluation_metrics["major_class_classification_report"].split('\n')
                                    report_data = []
                                    for line in report_lines:
                                        if line.strip():
                                            report_data.append([line])
                                    report_df = pd.DataFrame(report_data)
                                    if len(report_df.columns) > 0:
                                        report_df.columns = ['Major_Class_Classification_Report']
                                    report_df.to_excel(writer, sheet_name='Major_Class_Report', index=False)
                                
                                # 保存小类分类报告
                                if "minor_class_classification_report" in evaluation_metrics:
                                    report_lines = evaluation_metrics["minor_class_classification_report"].split('\n')
                                    report_data = []
                                    for line in report_lines:
                                        if line.strip():
                                            report_data.append([line])
                                    report_df = pd.DataFrame(report_data)
                                    if len(report_df.columns) > 0:
                                        report_df.columns = ['Minor_Class_Classification_Report']
                                    report_df.to_excel(writer, sheet_name='Minor_Class_Report', index=False)
                                
                                # 保存大类混淆矩阵
                                if "major_class_confusion_matrix" in evaluation_metrics and major_unique_labels:
                                    major_cm_df = pd.DataFrame(
                                        evaluation_metrics["major_class_confusion_matrix"],
                                        index=major_unique_labels,
                                        columns=major_unique_labels
                                    )
                                    major_cm_df.to_excel(writer, sheet_name='Major_Class_Confusion_Matrix')
                                
                                # 保存小类混淆矩阵
                                if "minor_class_confusion_matrix" in evaluation_metrics and minor_unique_labels:
                                    minor_cm_df = pd.DataFrame(
                                        evaluation_metrics["minor_class_confusion_matrix"],
                                        index=minor_unique_labels,
                                        columns=minor_unique_labels
                                    )
                                    minor_cm_df.to_excel(writer, sheet_name='Minor_Class_Confusion_Matrix')
                                    
                            elif args.classification_mode == "recommendation":
                                # 推荐模式：保存推荐指标
                                metrics_data = {
                                    "Metric": [
                                        "Top1 Accuracy", "Top3 Accuracy", "Exact Match",
                                        "Precision (weighted)", "Recall (weighted)", "F1 Score (weighted)",
                                        "Precision (macro)", "Recall (macro)", "F1 Score (macro)",
                                        "Top1 Precision", "Top1 Recall", "Top1 F1"
                                    ],
                                    "Value": [
                                        evaluation_metrics.get("top1_accuracy", 0),
                                        evaluation_metrics.get("top3_accuracy", 0),
                                        evaluation_metrics.get("exact_match", 0),
                                        evaluation_metrics.get("weighted_precision", 0),
                                        evaluation_metrics.get("weighted_recall", 0),
                                        evaluation_metrics.get("weighted_f1", 0),
                                        evaluation_metrics.get("macro_precision", 0),
                                        evaluation_metrics.get("macro_recall", 0),
                                        evaluation_metrics.get("macro_f1", 0),
                                        evaluation_metrics.get("top1_precision", 0),
                                        evaluation_metrics.get("top1_recall", 0),
                                        evaluation_metrics.get("top1_f1", 0)
                                    ]
                                }
                                
                                metrics_df = pd.DataFrame(metrics_data)
                                metrics_df.to_excel(writer, sheet_name='Recommendation_Metrics', index=False)
                                
                                # 保存分类报告
                                if "classification_report" in evaluation_metrics:
                                    report_lines = evaluation_metrics["classification_report"].split('\n')
                                    report_data = []
                                    for line in report_lines:
                                        if line.strip():
                                            report_data.append([line])
                                    report_df = pd.DataFrame(report_data)
                                    if len(report_df.columns) > 0:
                                        report_df.columns = ['Classification_Report']
                                    report_df.to_excel(writer, sheet_name='Classification_Report', index=False)
                                
                                # 保存混淆矩阵
                                if "confusion_matrix" in evaluation_metrics and "unique_labels" in evaluation_metrics:
                                    unique_labels = evaluation_metrics["unique_labels"]
                                    cm_df = pd.DataFrame(
                                        evaluation_metrics["confusion_matrix"],
                                        index=unique_labels,
                                        columns=unique_labels
                                    )
                                    cm_df.to_excel(writer, sheet_name='Confusion_Matrix')
                            else:
                                # 其他模式：保存通用指标
                                metrics_data = {
                                    "Metric": [],
                                    "Value": []
                                }
                                
                                # 根据不同模式添加相应指标
                                for key, value in evaluation_metrics.items():
                                    if isinstance(value, (int, float)) and not key.endswith('_matrix') and not key.endswith('_report'):
                                        metrics_data["Metric"].append(key)
                                        metrics_data["Value"].append(value)
                                
                                if metrics_data["Metric"]:
                                    metrics_df = pd.DataFrame(metrics_data)
                                    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                                
                                # 保存分类报告
                                if "classification_report" in evaluation_metrics:
                                    report_lines = evaluation_metrics["classification_report"].split('\n')
                                    report_data = []
                                    for line in report_lines:
                                        if line.strip():
                                            report_data.append([line])
                                    report_df = pd.DataFrame(report_data)
                                    if len(report_df.columns) > 0:
                                        report_df.columns = ['Classification_Report']
                                    report_df.to_excel(writer, sheet_name='Classification_Report', index=False)
                        
                        logger.info(f"评估指标已保存至: {metrics_file}")
                        
                    except Exception as e:
                        logger.error(f"保存评估指标失败: {e}")
                        logger.error(f"错误详情: {str(e)}")
                else:
                    logger.warning("评估指标计算失败或无有效数据")
            else:
                if not args.evaluate:
                    logger.info("评估功能未启用（--evaluate参数未设置）")
                else:
                    logger.warning("没有有效的结果数据进行评估")
            
            return
        else:
            logger.info(f"平衡采样检查完成，需要补充 {len(additional_sampled_df)} 个样本")
            logger.info(f"补充采样信息: {sampling_info}")
            
            # 更新df为需要额外处理的样本
            df = additional_sampled_df.reset_index(drop=True)
            logger.info(f"将继续处理这 {len(df)} 个补充样本")
    else:
        logger.info("未发现已存在的JSON日志文件，将从头开始处理")
    
    if "cleaned_text" not in df.columns or "VisitNumber" not in df.columns:
        logger.error(f"错误：文件中缺少 'cleaned_text' 或 'VisitNumber' 列")
        logger.error(f"可用列: {df.columns.tolist()}")
        return
    
    # 根据分类模式过滤数据
    if args.classification_mode == "binary":
        # 二分类模式：只处理OverallDiagnosis为Depression和Anxiety的数据
        if "OverallDiagnosis" in df.columns:
            original_count = len(df)
            df = df[df["OverallDiagnosis"].isin(["Depression", "Anxiety"])].copy()
            filtered_count = len(df)
            logger.info(f"二分类模式：从{original_count}条记录中过滤出{filtered_count}条Depression/Anxiety记录")
            
            if filtered_count == 0:
                logger.error("过滤后没有符合条件的记录，请检查数据中OverallDiagnosis列的值")
                return
        else:
            logger.warning("二分类模式但未找到OverallDiagnosis列，将处理所有数据")
    elif args.classification_mode == "multiclass":
        # 四分类模式：处理所有数据
        logger.info(f"四分类模式：处理所有{len(df)}条记录")
    elif args.classification_mode == "depression_symptom":
        # 抑郁症状检测模式：处理所有数据
        logger.info(f"抑郁症状检测模式：处理所有{len(df)}条记录")
        if "DepressionDiagnosis" in df.columns:
            symptom_counts = df["DepressionDiagnosis"].value_counts()
            logger.info(f"DepressionDiagnosis分布: {symptom_counts.to_dict()}")
        else:
            logger.warning("未找到DepressionDiagnosis列，无法进行模型评估")
    elif args.classification_mode == "anxiety_symptom":
        # 焦虑症状检测模式：处理所有数据
        logger.info(f"焦虑症状检测模式：处理所有{len(df)}条记录")
        if "AnxietyDiagnosis" in df.columns:
            symptom_counts = df["AnxietyDiagnosis"].value_counts()
            logger.info(f"AnxietyDiagnosis分布: {symptom_counts.to_dict()}")
        else:
            logger.warning("未找到AnxietyDiagnosis列，无法进行模型评估")
    elif args.classification_mode == "multilabel":
        # 多标签模式：处理所有数据
        logger.info(f"多标签模式：处理所有{len(df)}条记录")
        if "MultilabelDiagnosis" in df.columns:
            label_counts = df["MultilabelDiagnosis"].value_counts()
            logger.info(f"MultilabelDiagnosis分布: {label_counts.to_dict()}")
        else:
            logger.warning("未找到MultilabelDiagnosis列，无法进行模型评估")
    elif args.classification_mode == "recommendation":
        # 推荐模式：处理所有数据
        logger.info(f"推荐模式：处理所有{len(df)}条记录")
        if "DiagnosisCode" in df.columns:
            # 预处理DiagnosisCode列，支持多个ICD代码
            df['Processed_DiagnosisCode'] = df["DiagnosisCode"].apply(preprocess_diagnosis_codes)
            logger.info("DiagnosisCode列预处理完成，支持多个ICD代码匹配")
            
            # 统计诊断代码分布
            all_codes = []
            for codes_list in df['Processed_DiagnosisCode']:
                all_codes.extend(codes_list)
            from collections import Counter
            code_counts = Counter(all_codes)
            logger.info(f"诊断代码分布（前10个）: {dict(list(code_counts.most_common(10)))}")
        else:
            logger.warning("未找到DiagnosisCode列，无法进行模型评估")
    
    # 检查是否存在用于评估的ground truth列
    has_ground_truth = False
    ground_truth_col = None
    
    if args.classification_mode in ["binary", "multiclass"]:
        ground_truth_col = "OverallDiagnosis"
        has_ground_truth = ground_truth_col in df.columns
        if has_ground_truth:
            logger.info("检测到OverallDiagnosis列，将计算模型评估指标")
        else:
            logger.info("未检测到OverallDiagnosis列，将跳过模型评估")
    elif args.classification_mode == "depression_symptom":
        ground_truth_col = "DepressionDiagnosis"
        has_ground_truth = ground_truth_col in df.columns
        if has_ground_truth:
            logger.info("检测到DepressionDiagnosis列，将计算模型评估指标")
        else:
            logger.info("未检测到DepressionDiagnosis列，将跳过模型评估")
    elif args.classification_mode == "anxiety_symptom":
        ground_truth_col = "AnxietyDiagnosis"
        has_ground_truth = ground_truth_col in df.columns
        if has_ground_truth:
            logger.info("检测到AnxietyDiagnosis列，将计算模型评估指标")
        else:
            logger.info("未检测到AnxietyDiagnosis列，将跳过模型评估")
    elif args.classification_mode == "multilabel":
        # 多标签模式：优先使用OverallDiagnosis列进行4分类评估，如果没有则使用MultilabelDiagnosis列
        if "OverallDiagnosis" in df.columns:
            ground_truth_col = "OverallDiagnosis"
            has_ground_truth = True
            logger.info("检测到OverallDiagnosis列，将使用多标签转4分类评估")
        elif "MultilabelDiagnosis" in df.columns:
            ground_truth_col = "MultilabelDiagnosis"
            has_ground_truth = True
            logger.info("检测到MultilabelDiagnosis列，将计算多标签评估指标")
        else:
            ground_truth_col = "OverallDiagnosis"  # 默认列名
            has_ground_truth = False
            logger.info("未检测到OverallDiagnosis或MultilabelDiagnosis列，将跳过模型评估")
    elif args.classification_mode == "icd10":
        # ICD10模式：使用DiagnosisCode列作为ground truth
        ground_truth_col = "DiagnosisCode"
        has_ground_truth = ground_truth_col in df.columns
        if has_ground_truth:
            logger.info("检测到DiagnosisCode列，将计算ICD10评估指标")
            # 预处理DiagnosisCode列，支持多个ICD代码
            df['Processed_DiagnosisCode'] = df[ground_truth_col].apply(preprocess_diagnosis_codes)
            logger.info("DiagnosisCode列预处理完成，支持多个ICD代码匹配")
        else:
            logger.info("未检测到DiagnosisCode列，将跳过模型评估")
    elif args.classification_mode == "recommendation":
        # Recommendation模式：使用DiagnosisCode列作为ground truth
        ground_truth_col = "DiagnosisCode"
        has_ground_truth = ground_truth_col in df.columns
        if has_ground_truth:
            logger.info("检测到DiagnosisCode列，将计算推荐模式评估指标")
            # 预处理DiagnosisCode列已在上面完成
        else:
            logger.info("未检测到DiagnosisCode列，将跳过模型评估")
    
    should_evaluate = args.evaluate and has_ground_truth
    if should_evaluate:
        logger.info("将进行模型评估")
    elif args.evaluate and not has_ground_truth:
        logger.warning(f"指定了--evaluate参数但数据中缺少{ground_truth_col}列，跳过评估")
    
    logger.info(f"开始并行处理记录...")
    
    # 创建信号量来控制并发数量
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    # 创建所有任务（跳过已处理的样本）
    tasks = []
    skipped_count = 0
    
    for idx, row in df.iterrows():
        visit_number = row["VisitNumber"]
        text = row["cleaned_text"]
        
        # Resume功能：跳过已处理的样本
        if visit_number in processed_visits:
            skipped_count += 1
            continue
        
        # 安全地获取ground truth值
        if args.classification_mode in ["icd10", "recommendation"] and has_ground_truth:
            # 对于ICD10和recommendation模式，获取预处理后的代码列表
            ground_truth_value = row.get('Processed_DiagnosisCode', [])
        else:
            ground_truth_value = row.get(ground_truth_col, None) if has_ground_truth else None
        
        task = process_record_async(
            client, semaphore, visit_number, text, 
            MODEL_NAME, 
            classification_mode=args.classification_mode,
            primary_diagnosis=ground_truth_value,
            debug=args.debug,
            save_full_response=args.save_full_response,
            top_logprobs=args.top_logprobs,
            site_url=args.site_url,
            site_name=args.site_name,
            save_json_logs=args.save_json_logs,
            api_type=api_type
        )
        tasks.append(task)
    
    if skipped_count > 0:
        logger.info(f"跳过了 {skipped_count} 个已处理的样本")
    
    logger.info(f"需要处理的新样本数量: {len(tasks)}")
    
    # 批处理执行所有任务并显示进度
    new_results_list = []
    if len(tasks) > 0:
        batch_size = args.batch_size
        total_batches = math.ceil(len(tasks) / batch_size)
        logger.info(f"将分 {total_batches} 个批次处理，每批 {batch_size} 个样本")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(tasks))
            batch_tasks = tasks[start_idx:end_idx]
            
            logger.info(f"正在处理第 {batch_idx + 1}/{total_batches} 批次 ({len(batch_tasks)} 个样本)")
            
            # 记录批次开始前的日志数量
            batch_start_log_count = len(json_log_data)
            
            # 执行当前批次的任务
            batch_results = []
            for future in tqdm(asyncio.as_completed(batch_tasks), total=len(batch_tasks), 
                             desc=f"批次 {batch_idx + 1}/{total_batches}"):
                result = await future
                batch_results.append(result)
                new_results_list.append(result)
            
            # 每个批次完成后保存JSON日志
            if args.save_json_logs and batch_results:
                logger.info(f"保存第 {batch_idx + 1} 批次的JSON日志...")
                
                # 获取当前批次的日志（从批次开始到现在新增的日志）
                batch_logs = json_log_data[batch_start_log_count:]
                
                # 生成批次特定的输出前缀
                base_dir = os.path.dirname(OUTPUT_FILE)
                base_name = os.path.basename(OUTPUT_FILE)
                base_file_name = os.path.splitext(base_name)[0]
                
                # 移除已有的日期前缀（如果存在）
                if base_file_name[:8].isdigit() and base_file_name[8] == '_':
                    base_file_name = base_file_name[9:]
                
                batch_output_prefix = f"{current_date}_{base_file_name}_batch_{batch_idx + 1:03d}"
                
                # 保存当前批次的JSON日志
                json_log_file = save_batch_json_logs_to_file(batch_output_prefix, MODEL_NAME, args.classification_mode, batch_logs, args.results_dir)
                if json_log_file:
                    logger.info(f"第 {batch_idx + 1} 批次JSON日志已保存至: {json_log_file}")
                else:
                    logger.warning(f"第 {batch_idx + 1} 批次JSON日志保存失败")
            
            logger.info(f"第 {batch_idx + 1} 批次完成，已处理 {len(batch_results)} 个样本")
        
        # 将新结果转换为DataFrame
        new_results_df = pd.DataFrame(new_results_list)
        logger.info(f"所有批次完成，新处理了 {len(new_results_df)} 个样本")
    else:
        new_results_df = pd.DataFrame()
        logger.info("没有新样本需要处理")
    
    # 合并之前的结果和新结果
    if len(previous_results_df) > 0 and len(new_results_df) > 0:
        logger.info("合并之前的结果和新结果...")
        results_df = pd.concat([previous_results_df, new_results_df], ignore_index=True)
        logger.info(f"合并后总样本数: {len(results_df)} (之前: {len(previous_results_df)}, 新增: {len(new_results_df)})")
    elif len(previous_results_df) > 0:
        logger.info("使用之前的结果...")
        results_df = previous_results_df
    elif len(new_results_df) > 0:
        logger.info("使用新结果...")
        results_df = new_results_df
    else:
        logger.warning("没有任何结果数据")
        results_df = pd.DataFrame()
    
    # 按VisitNumber排序以保持原始顺序
    if len(results_df) > 0:
        results_df = results_df.sort_values('VisitNumber').reset_index(drop=True)
    
    # 为评估选择合适的样本子集
    evaluation_df = results_df.copy()  # 默认使用所有样本
    
    # 如果是resume模式且有目标样本数限制，需要选择评估样本
    if existing_json_file and previous_json_data and args.limit and args.limit > 0:
        logger.info(f"=== 选择评估样本 ===")
        
        # 获取原始目标样本数
        target_sample_count = args.limit
        
        # 使用平衡采样逻辑来确定应该评估哪些样本
        evaluation_sampled_df, eval_sampling_info = balanced_sampling_with_resume(
            df=df,  # 使用原始数据
            classification_mode=args.classification_mode,
            target_sample_count=target_sample_count,
            processed_visits=None,  # 不考虑已处理样本，重新选择
            min_samples_per_class=args.min_samples_per_class,
            random_state=42
        )
        
        if len(evaluation_sampled_df) > 0:
            # 获取应该用于评估的VisitNumber
            target_visit_numbers = set(evaluation_sampled_df['VisitNumber'].values)
            
            # 从results_df中筛选出这些样本
            evaluation_df = results_df[results_df['VisitNumber'].isin(target_visit_numbers)].copy()
            
            logger.info(f"评估样本选择完成:")
            logger.info(f"  目标样本数: {target_sample_count}")
            logger.info(f"  实际可评估样本数: {len(evaluation_df)}")
            logger.info(f"  总处理样本数: {len(results_df)}")
            
            # 显示评估样本的分布信息
            if eval_sampling_info.get('method') == 'balanced_sampling_with_resume_icd_major':
                if 'final_major_class_distribution' in eval_sampling_info:
                    logger.info(f"  评估样本ICD-10大类分布: {eval_sampling_info['final_major_class_distribution']}")
        else:
            logger.warning("无法确定评估样本，将使用所有处理的样本进行评估")
    
    # 保存结果
    try:
        # 生成带日期前缀的文件名
        base_dir = os.path.dirname(OUTPUT_FILE)
        base_name = os.path.basename(OUTPUT_FILE)
        base_file_name = os.path.splitext(base_name)[0] + f"_{current_time}"
        base_ext = os.path.splitext(base_name)[1]
        
        # 如果base_name已经包含日期前缀，则移除它
        if base_file_name.startswith(current_date):
            output_base = f"{base_file_name}{base_ext}"
        else:
            # 移除已有的日期前缀（如果存在）
            if base_file_name[:8].isdigit() and base_file_name[8] == '_':
                base_file_name = base_file_name[9:]
            output_base = f"{current_date}_{base_file_name}{base_ext}"
        
        # 构建完整的输出路径
        output_path = os.path.join(base_dir, output_base)
        
        # 如果文件已存在，添加时间戳
        if os.path.exists(output_path):
            base, ext = os.path.splitext(output_path)
            timestamp = datetime.now().strftime("%H%M%S")
            output_file = f"{base}_{timestamp}{ext}"
        else:
            output_file = output_path
            
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        results_df.to_excel(output_file, index=False)
        logger.info(f"结果已保存至: {output_file}")
        
        # 保存完整的JSON日志（包含所有批次）
        if args.save_json_logs:
            output_prefix = os.path.splitext(os.path.basename(output_file))[0]
            # 添加"complete"标识以区分完整日志和批次日志
            complete_output_prefix = f"{output_prefix}_complete"
            json_log_file = save_json_logs_to_file(complete_output_prefix, MODEL_NAME, args.classification_mode, args.results_dir)
            if json_log_file:
                logger.info(f"完整请求响应日志已保存至: {json_log_file}")
                logger.info(f"注意：批次日志已在处理过程中分别保存，此文件包含所有批次的完整日志")
        
        # # 如果启用了可视化，生成结果图表
        # if args.visualize:
        #     output_prefix = os.path.splitext(os.path.basename(output_file))[0]
        #     visualize_results(results_df, output_prefix, args.classification_mode)
            
        # 计算模型评估指标
        evaluation_metrics = {}
        if should_evaluate:
            logger.info(f"用于评估的样本数: {len(evaluation_df)}")
            evaluation_metrics = calculate_evaluation_metrics(evaluation_df, args.classification_mode)
        
        # 如果有评估指标，保存到单独的文件
        if evaluation_metrics and should_evaluate:
            try:
                # 生成评估结果可视化图表
                output_prefix = os.path.splitext(os.path.basename(output_file))[0]
                # visualize_evaluation_results(evaluation_metrics, output_prefix, args.results_dir)
                
                # 保存评估指标到单独的Excel文件
                metrics_file = output_file.replace('.xlsx', '_evaluation_metrics.xlsx')
                with pd.ExcelWriter(metrics_file) as writer:
                    
                    if args.classification_mode == "icd10":
                        # ICD-10模式：保存大类和小类的指标
                        # 大类指标
                        major_metrics_data = {
                            "Metric": ["Major Class Weighted F1-Score", "Major Class Balanced Accuracy"],
                            "Value": [
                                evaluation_metrics.get("major_class_weighted_f1", 0),
                                evaluation_metrics.get("major_class_balanced_accuracy", 0)
                            ]
                        }
                        
                        # 小类指标
                        minor_metrics_data = {
                            "Metric": ["Minor Class Weighted F1-Score", "Minor Class Balanced Accuracy"],
                            "Value": [
                                evaluation_metrics.get("minor_class_weighted_f1", 0),
                                evaluation_metrics.get("minor_class_balanced_accuracy", 0)
                            ]
                        }
                        
                        # 合并指标
                        combined_metrics_data = {
                            "Metric": major_metrics_data["Metric"] + minor_metrics_data["Metric"],
                            "Value": major_metrics_data["Value"] + minor_metrics_data["Value"]
                        }
                        
                        # 添加各大类的F1分数
                        major_unique_labels = evaluation_metrics.get("major_class_unique_labels", [])
                        major_individual_f1 = evaluation_metrics.get("major_class_individual_f1", {})
                        for label in major_unique_labels:
                            if label in major_individual_f1:
                                combined_metrics_data["Metric"].append(f"Major Class {label} F1-Score")
                                combined_metrics_data["Value"].append(major_individual_f1[label])
                        
                        # 添加各小类的F1分数
                        minor_unique_labels = evaluation_metrics.get("minor_class_unique_labels", [])
                        minor_individual_f1 = evaluation_metrics.get("minor_class_individual_f1", {})
                        for label in minor_unique_labels:
                            if label in minor_individual_f1:
                                combined_metrics_data["Metric"].append(f"Minor Class {label} F1-Score")
                                combined_metrics_data["Value"].append(minor_individual_f1[label])
                        
                        metrics_df = pd.DataFrame(combined_metrics_data)
                        metrics_df.to_excel(writer, sheet_name='ICD10_Metrics', index=False)
                        
                        # 保存大类分类报告
                        if "major_class_classification_report" in evaluation_metrics:
                            report_lines = evaluation_metrics["major_class_classification_report"].split('\n')
                            report_data = []
                            for line in report_lines:
                                if line.strip():
                                    report_data.append([line])
                            report_df = pd.DataFrame(report_data)
                            if len(report_df.columns) > 0:
                                report_df.columns = ['Major_Class_Classification_Report']
                            report_df.to_excel(writer, sheet_name='Major_Class_Report', index=False)
                        
                        # 保存小类分类报告
                        if "minor_class_classification_report" in evaluation_metrics:
                            report_lines = evaluation_metrics["minor_class_classification_report"].split('\n')
                            report_data = []
                            for line in report_lines:
                                if line.strip():
                                    report_data.append([line])
                            report_df = pd.DataFrame(report_data)
                            if len(report_df.columns) > 0:
                                report_df.columns = ['Minor_Class_Classification_Report']
                            report_df.to_excel(writer, sheet_name='Minor_Class_Report', index=False)
                        
                        # 保存大类混淆矩阵
                        if "major_class_confusion_matrix" in evaluation_metrics and major_unique_labels:
                            major_cm_df = pd.DataFrame(
                                evaluation_metrics["major_class_confusion_matrix"],
                                index=major_unique_labels,
                                columns=major_unique_labels
                            )
                            major_cm_df.to_excel(writer, sheet_name='Major_Class_Confusion_Matrix')
                        
                        # 保存小类混淆矩阵
                        if "minor_class_confusion_matrix" in evaluation_metrics and minor_unique_labels:
                            minor_cm_df = pd.DataFrame(
                                evaluation_metrics["minor_class_confusion_matrix"],
                                index=minor_unique_labels,
                                columns=minor_unique_labels
                            )
                            minor_cm_df.to_excel(writer, sheet_name='Minor_Class_Confusion_Matrix')
                    
                    elif args.classification_mode == "recommendation":
                        # Recommendation模式：保存推荐指标
                        recommendation_metrics_data = {
                            "Metric": ["Top1 Accuracy", "Top3 Accuracy", "Recall@1", "Recall@3", "Exact Match"],
                            "Value": [
                                evaluation_metrics.get("top1_accuracy", 0),
                                evaluation_metrics.get("top3_accuracy", 0),
                                evaluation_metrics.get("recall_at_1", 0),
                                evaluation_metrics.get("recall_at_3", 0),
                                evaluation_metrics.get("exact_match", 0)
                            ]
                        }
                        
                        metrics_df = pd.DataFrame(recommendation_metrics_data)
                        metrics_df.to_excel(writer, sheet_name='Recommendation_Metrics', index=False)
                    
                    else:
                        # 其他模式：使用原有的保存逻辑
                        if args.classification_mode in ["binary", "multiclass"]:
                            unique_labels = evaluation_metrics.get("unique_labels", ["Depression", "Anxiety"])
                        elif args.classification_mode in ["depression_symptom", "anxiety_symptom"]:
                            unique_labels = evaluation_metrics.get("unique_labels", ["有", "没有"])
                        else:
                            unique_labels = evaluation_metrics.get("unique_labels", [])
                        
                        metrics_data = {
                            "Metric": ["Weighted F1-Score", "Balanced Accuracy", "Macro AUPRC", "Weighted AUPRC"],
                            "Value": [
                                evaluation_metrics.get("weighted_f1", 0),
                                evaluation_metrics.get("balanced_accuracy", 0), 
                                evaluation_metrics.get("macro_auprc", 0),
                                evaluation_metrics.get("weighted_auprc", 0)
                            ]
                        }
                        
                        # 添加各类别的F1和AUPRC
                        for label in unique_labels:
                            if "individual_f1" in evaluation_metrics and label in evaluation_metrics["individual_f1"]:
                                metrics_data["Metric"].append(f"{label} F1-Score")
                                metrics_data["Value"].append(evaluation_metrics["individual_f1"][label])
                            if "individual_auprc" in evaluation_metrics and label in evaluation_metrics["individual_auprc"]:
                                metrics_data["Metric"].append(f"{label} AUPRC")
                                metrics_data["Value"].append(evaluation_metrics["individual_auprc"][label])
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                        
                        # 保存分类报告
                        if "classification_report" in evaluation_metrics:
                            report_lines = evaluation_metrics["classification_report"].split('\n')
                            report_data = []
                            for line in report_lines:
                                if line.strip():
                                    report_data.append([line])
                            report_df = pd.DataFrame(report_data)
                            if len(report_df.columns) > 0:
                                report_df.columns = ['Classification_Report']
                            report_df.to_excel(writer, sheet_name='Classification_Report', index=False)
                        
                        # 保存混淆矩阵
                        if "confusion_matrix" in evaluation_metrics and unique_labels:
                            cm_df = pd.DataFrame(
                                evaluation_metrics["confusion_matrix"],
                                index=unique_labels,
                                columns=unique_labels
                            )
                            cm_df.to_excel(writer, sheet_name='Confusion_Matrix')
                
                logger.info(f"评估指标已保存至: {metrics_file}")
                
            except Exception as e:
                logger.error(f"保存评估指标失败: {e}")
            
    except Exception as e:
        logger.error(f"保存结果失败: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 