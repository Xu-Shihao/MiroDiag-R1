#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评估结果提取工具
从日志文件中提取各模型的评估指标并生成 Excel 报告
"""

import os
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict


class ModelResultsExtractor:
    """模型结果提取器"""
    
    def __init__(self, log_dir: str):
        """
        初始化提取器
        
        Args:
            log_dir: 日志文件目录路径
        """
        self.log_dir = Path(log_dir)
        self.results = []
        
    def extract_model_name(self, filename: str) -> str:
        """
        从文件名中提取模型名称
        
        Args:
            filename: 日志文件名
            
        Returns:
            模型名称
        """
        # 移除日期前缀和文件扩展名
        name = re.sub(r'^\d{8}_', '', filename).replace('.log', '')
        # 移除时间戳后缀
        name = re.sub(r'_\d{6}$', '', name)
        # 移除 "recommendation" 后缀
        name = name.replace('_recommendation', '')
        return name
    
    def parse_filename(self, filename: str) -> Dict[str, str]:
        """
        解析文件名，提取日期、时间戳和模型名称
        
        Args:
            filename: 日志文件名
            
        Returns:
            包含日期、时间戳和模型名称的字典
        """
        # 文件名格式: YYYYMMDD_model_name_recommendation_HHMMSS.log
        pattern = r'^(\d{8})_(.+)_recommendation_(\d{6})\.log$'
        match = re.match(pattern, filename)
        
        if match:
            date_str, model_part, timestamp = match.groups()
            return {
                'date': date_str,
                'timestamp': timestamp,
                'model_name': model_part,
                'full_datetime': f"{date_str}_{timestamp}"
            }
        else:
            # 备用解析方式
            name_without_ext = filename.replace('.log', '')
            parts = name_without_ext.split('_')
            if len(parts) >= 3:
                date_str = parts[0] if parts[0].isdigit() and len(parts[0]) == 8 else '00000000'
                timestamp = parts[-1] if parts[-1].isdigit() and len(parts[-1]) == 6 else '000000'
                model_part = '_'.join(parts[1:-2]) if len(parts) > 3 else parts[1] if len(parts) > 1 else 'unknown'
                return {
                    'date': date_str,
                    'timestamp': timestamp,
                    'model_name': model_part,
                    'full_datetime': f"{date_str}_{timestamp}"
                }
            
        return {
            'date': '00000000',
            'timestamp': '000000',
            'model_name': filename.replace('.log', ''),
            'full_datetime': '00000000_000000'
        }
    
    def parse_log_file(self, filepath: Path) -> Optional[Dict]:
        """
        解析单个日志文件，提取评估指标
        
        Args:
            filepath: 日志文件路径
            
        Returns:
            包含评估指标的字典，如果未找到则返回 None
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找推荐模式评估指标部分
            pattern = r'=== 推荐模式评估指标（限制11种疾病）===.*?(?=DEBUG|\Z)'
            match = re.search(pattern, content, re.DOTALL)
            
            if not match:
                print(f"警告: 在 {filepath.name} 中未找到评估指标")
                return None
            
            metrics_section = match.group(0)
            
            # 提取各个指标
            metrics = {}
            
            # 指标模式定义
            metric_patterns = {
                '有效样本数': r'有效样本数:\s*(\d+)',
                'Top1_Accuracy': r'Top1 Accuracy:\s*([\d.]+)',
                'Top3_Accuracy': r'Top3 Accuracy:\s*([\d.]+)',
                'Exact_Match': r'Exact Match:\s*([\d.]+)',
                'Precision_Weighted': r'Precision \(weighted\):\s*([\d.]+)',
                'Recall_Weighted': r'Recall \(weighted\):\s*([\d.]+)',
                'F1_Score_Weighted': r'F1 Score \(weighted\):\s*([\d.]+)',
                'Precision_Macro': r'Precision \(macro\):\s*([\d.]+)',
                'Recall_Macro': r'Recall \(macro\):\s*([\d.]+)',
                'F1_Score_Macro': r'F1 Score \(macro\):\s*([\d.]+)',
                'Top1_Precision': r'Top1 Precision:\s*([\d.]+)',
                'Top1_Recall': r'Top1 Recall:\s*([\d.]+)',
                'Top1_F1': r'Top1 F1:\s*([\d.]+)'
            }
            
            for metric_name, pattern in metric_patterns.items():
                match = re.search(pattern, metrics_section)
                if match:
                    value = match.group(1)
                    # 将数值转换为适当的类型
                    if metric_name == '有效样本数':
                        metrics[metric_name] = int(value)
                    else:
                        metrics[metric_name] = float(value)
                else:
                    print(f"警告: 在 {filepath.name} 中未找到指标 {metric_name}")
                    metrics[metric_name] = None
            
            # 添加模型名称和文件信息
            file_info = self.parse_filename(filepath.name)
            metrics['模型名称'] = file_info['model_name']
            metrics['日志文件'] = filepath.name
            metrics['日期'] = file_info['date']
            metrics['时间戳'] = file_info['timestamp']
            
            return metrics
            
        except Exception as e:
            print(f"错误: 解析文件 {filepath} 时出错: {e}")
            return None
    
    def get_latest_log_files(self) -> List[Path]:
        """
        获取每个模型的最新日期最新时间戳的日志文件
        
        Returns:
            最新的日志文件列表
        """
        log_files = list(self.log_dir.glob("*.log"))
        
        if not log_files:
            return []
        
        # 按模型分组文件
        model_files = defaultdict(list)
        
        for log_file in log_files:
            file_info = self.parse_filename(log_file.name)
            model_name = file_info['model_name']
            model_files[model_name].append({
                'path': log_file,
                'info': file_info
            })
        
        # 为每个模型选择最新的文件
        latest_files = []
        
        for model_name, files in model_files.items():
            # 按日期和时间戳排序，选择最新的
            latest_file = max(files, key=lambda x: x['info']['full_datetime'])
            latest_files.append(latest_file['path'])
            
            print(f"模型 {model_name}: 选择最新文件 {latest_file['path'].name}")
            if len(files) > 1:
                print(f"  跳过的旧文件: {', '.join([f['path'].name for f in files if f != latest_file])}")
        
        return sorted(latest_files)
    
    def extract_all_results(self) -> List[Dict]:
        """
        提取所有模型的最新评估结果
        
        Returns:
            包含所有模型最新评估结果的列表
        """
        all_log_files = list(self.log_dir.glob("*.log"))
        
        if not all_log_files:
            print(f"警告: 在目录 {self.log_dir} 中未找到日志文件")
            return []
        
        print(f"找到 {len(all_log_files)} 个日志文件")
        
        # 获取最新的日志文件
        latest_files = self.get_latest_log_files()
        
        if not latest_files:
            print("警告: 未找到有效的日志文件")
            return []
        
        print(f"选择了 {len(latest_files)} 个最新的日志文件进行处理")
        print("-" * 60)
        
        results = []
        for log_file in latest_files:
            print(f"正在处理: {log_file.name}")
            result = self.parse_log_file(log_file)
            if result:
                results.append(result)
                print(f"  ✓ 成功提取 {result['模型名称']} 的评估指标")
            else:
                print(f"  ✗ 跳过 {log_file.name}")
        
        self.results = results
        return results
    
    def create_excel_report(self, output_folder: str = None, method_name: str = "") -> str:
        """
        创建 Excel 报告
        
        Args:
            output_folder: 输出文件路径目录
            
        Returns:
            生成的 Excel 文件路径
        """
        if not self.results:
            raise ValueError("没有可用的结果数据，请先运行 extract_all_results()")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 生成输出文件名
        if output_folder:
            output_path = os.path.join(output_folder, f"model_evaluation_results_{method_name}_{timestamp}.xlsx")
        else:
            output_path = f"model_evaluation_results_{method_name}_{timestamp}.xlsx"
        
        # 创建 DataFrame
        df = pd.DataFrame(self.results)
        
        # 只保留指定的指标列
        selected_columns = [
            '模型名称', 'Top1_Accuracy', 'Top3_Accuracy', 'Exact_Match',
            'Precision_Macro', 'Recall_Macro', 'F1_Score_Macro'
        ]
        
        # 确保所有列都存在
        existing_columns = [col for col in selected_columns if col in df.columns]
        df = df[existing_columns]
        
        # 按F1_Score_Macro由高到低排序
        if 'F1_Score_Macro' in df.columns:
            df = df.sort_values('F1_Score_Macro', ascending=False)
        
        # 创建 Excel 文件
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 写入主要结果表
            df.to_excel(writer, sheet_name='模型评估结果', index=False)
            
            # 获取工作表对象来进行格式化
            worksheet = writer.sheets['模型评估结果']
            
            # 自动调整列宽
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 20)  # 限制最大宽度
                worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # 创建性能比较表，只包含指定的指标
            performance_metrics = ['Top1_Accuracy', 'Top3_Accuracy', 'Exact_Match', 
                                 'Precision_Macro', 'Recall_Macro', 'F1_Score_Macro']
            
            # 确保所有指标都存在于数据中
            available_metrics = [metric for metric in performance_metrics if metric in df.columns]
            performance_df = df[['模型名称'] + available_metrics].copy()
            
            # 添加排名
            for metric in available_metrics:
                if metric in performance_df.columns:
                    performance_df[f'{metric}_Rank'] = performance_df[metric].rank(
                        method='min', ascending=False
                    ).astype('Int64')
            
            performance_df.to_excel(writer, sheet_name='性能对比', index=False)
            
            # 格式化性能对比表
            perf_worksheet = writer.sheets['性能对比']
            for column in perf_worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 25)
                perf_worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"Excel 报告已生成: {output_path}")
        return output_path
    
    def print_summary(self):
        """打印结果摘要"""
        if not self.results:
            print("没有可用的结果数据")
            return
        
        print("\n" + "="*80)
        print("模型评估结果摘要")
        print("="*80)
        
        # 按 Top1 Accuracy 排序
        sorted_results = sorted(self.results, 
                              key=lambda x: x.get('Top1_Accuracy', 0), 
                              reverse=True)
        
        print(f"{'排名':<4} {'模型名称':<25} {'Top1准确率':<10} {'Top3准确率':<10} {'精确匹配':<10} {'F1(加权)':<10}")
        print("-" * 80)
        
        for i, result in enumerate(sorted_results, 1):
            model_name = result.get('模型名称', 'Unknown')
            top1_acc = result.get('Top1_Accuracy', 0)
            top3_acc = result.get('Top3_Accuracy', 0)
            exact_match = result.get('Exact_Match', 0)
            f1_weighted = result.get('F1_Score_Weighted', 0)
            
            print(f"{i:<4} {model_name:<25} {top1_acc:<10.4f} {top3_acc:<10.4f} "
                  f"{exact_match:<10.4f} {f1_weighted:<10.4f}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='模型评估结果提取工具')
    parser.add_argument('--dataset', default='MiroDiag-16K', choices=['MDD-5K', 'MiroDiag-16K'], 
                        help='数据集名称')
    parser.add_argument('--classification_mode', default='recommendation', 
                        help='分类模式')
    parser.add_argument('--results_dir', default=None, 
                        help='结果目录路径')
    
    args = parser.parse_args()
    
    # 设置日志目录路径
    method_name = args.dataset
    if args.results_dir:
        base_dir = args.results_dir
    else:
        base_dir = f"/Users/shihaoxu/Desktop/work/document/Dlab/code/baseline_llm/results/{method_name}"
    
    log_dir = f"{base_dir}/logs"
    output_folder = base_dir
    
    print("模型评估结果提取工具")
    print("=" * 50)
    
    # 创建提取器
    extractor = ModelResultsExtractor(log_dir)
    
    # 提取结果
    print(f"正在扫描目录: {log_dir}")
    results = extractor.extract_all_results()
    
    if not results:
        print("未找到任何有效的评估结果")
        return
    
    # 打印摘要
    extractor.print_summary()
    
    # 生成 Excel 报告
    print("\n正在生成 Excel 报告...")
    excel_path = extractor.create_excel_report(output_folder, method_name)
    
    print(f"\n✓ 成功处理了 {len(results)} 个模型的评估结果")
    print(f"✓ Excel 报告已保存至: {excel_path}")


if __name__ == "__main__":
    main()
