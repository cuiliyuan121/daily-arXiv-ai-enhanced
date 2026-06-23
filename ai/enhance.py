import os
import json
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import requests

import dotenv
import argparse
from tqdm import tqdm

import langchain_core.exceptions
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from pydantic import BaseModel, Field
from structure import Structure

if os.path.exists('.env'):
    dotenv.load_dotenv()
template = open("template.txt", "r").read()
system = open("system.txt", "r").read()


class InterestDecision(BaseModel):
    keep: bool = Field(description="Whether this paper should be kept for downstream summarization")
    reason: str = Field(description="Short reason for the decision")
    matched_interests: List[str] = Field(
        default_factory=list,
        description="The interest keywords/themes that match this paper"
    )

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    return parser.parse_args()


def parse_interests(raw_interest: str) -> List[str]:
    """Parse INTEREST env into a clean keyword list."""
    if not raw_interest:
        return []

    raw_interest = raw_interest.strip()
    if raw_interest in {'""', "''"}:
        return []

    if (
        len(raw_interest) >= 2 and
        raw_interest[0] == raw_interest[-1] and
        raw_interest[0] in {'"', "'"}
    ):
        raw_interest = raw_interest[1:-1].strip()

    if not raw_interest:
        return []

    normalized = raw_interest.replace("\n", ";").replace(",", ";")
    return [part.strip() for part in normalized.split(";") if part.strip()]


def normalize_for_keyword_match(text: str) -> str:
    """Normalize text so hyphenated and spaced phrases can be matched consistently."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def get_direct_interest_matches(item: Dict, interests: List[str]) -> List[str]:
    """Return interests that appear directly in title, abstract, or comment."""
    raw_text = " ".join(
        str(item.get(key) or "") for key in ["title", "summary", "comment"]
    ).lower()
    normalized_text = normalize_for_keyword_match(raw_text)

    matches = []
    for interest in interests:
        interest_lower = interest.lower()
        normalized_interest = normalize_for_keyword_match(interest)
        if interest_lower in raw_text or normalized_interest in normalized_text:
            matches.append(interest)
    return matches


def filter_items_by_interest(
    data: List[Dict],
    model_name: str,
    language: str,
    interests: List[str],
) -> List[Dict]:
    """Keep direct keyword matches first, then use the model for semantic matches."""
    if not interests:
        print("INTEREST is empty, skipping interest-based filtering", file=sys.stderr)
        return data

    llm = ChatOpenAI(model=model_name).with_structured_output(
        InterestDecision,
        method="function_calling",
    )
    interest_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are a strict arXiv paper triage assistant. "
                "Decide whether a paper is relevant to the user's research interests. "
                "Keep the paper only when it is clearly related to at least one interest, "
                "including close subtopics, standard synonyms, or direct applications. "
                "Reject papers that are only loosely related."
            ),
        ),
        (
            "human",
            (
                "Write the decision reason in {language}.\n\n"
                "User interests:\n{interests}\n\n"
                "Paper title: {title}\n"
                "Paper categories: {categories}\n"
                "Paper abstract:\n{summary}\n\n"
                "Return whether this paper should be kept."
            ),
        ),
    ])
    chain = interest_prompt | llm

    filtered_data = []
    direct_match_count = 0
    for item in tqdm(data, desc="Filtering by interest"):
        direct_matches = get_direct_interest_matches(item, interests)
        if direct_matches:
            item["interest_filter"] = {
                "matched_interests": direct_matches,
                "reason": f"Direct keyword match: {', '.join(direct_matches)}",
            }
            filtered_data.append(item)
            direct_match_count += 1
            continue

        try:
            decision: InterestDecision = chain.invoke({
                "language": language,
                "interests": "; ".join(interests),
                "title": item.get("title", ""),
                "categories": ", ".join(item.get("categories", [])),
                "summary": item.get("summary", ""),
            })
        except Exception as e:
            print(
                f"Interest filtering failed for {item.get('id', 'unknown')}: {e}. Keeping the paper.",
                file=sys.stderr,
            )
            filtered_data.append(item)
            continue

        if decision.keep:
            item["interest_filter"] = {
                "matched_interests": decision.matched_interests,
                "reason": decision.reason,
            }
            filtered_data.append(item)

    print(
        f"Interest-based filtering kept {len(filtered_data)} / {len(data)} papers "
        f"({direct_match_count} direct keyword matches)",
        file=sys.stderr,
    )
    return filtered_data

def process_single_item(chain, item: Dict, language: str) -> Dict:
    def is_sensitive(content: str) -> bool:
        """
        调用 spam.dw-dengwei.workers.dev 接口检测内容是否包含敏感词。
        返回 True 表示触发敏感词，False 表示未触发。
        """
        try:
            resp = requests.post(
                "https://spam.dw-dengwei.workers.dev",
                json={"text": content},
                timeout=5
            )
            if resp.status_code == 200:
                result = resp.json()
                # 约定接口返回 {"sensitive": true/false, ...}
                return result.get("sensitive", True)
            else:
                # 如果接口异常，默认不触发敏感词
                print(f"Sensitive check failed with status {resp.status_code}", file=sys.stderr)
                return True
        except Exception as e:
            print(f"Sensitive check error: {e}", file=sys.stderr)
            return True

    # 检查 summary 字段
    if is_sensitive(item.get("summary", "")):
        return None

    """处理单个数据项"""
    # Default structure with meaningful fallback values
    default_ai_fields = {
        "tldr": "Summary generation failed",
        "motivation": "Motivation analysis unavailable",
        "method": "Method extraction failed",
        "result": "Result analysis unavailable",
        "conclusion": "Conclusion extraction failed"
    }
    
    try:
        response: Structure = chain.invoke({
            "language": language,
            "content": item['summary']
        })
        item['AI'] = response.model_dump()
    except langchain_core.exceptions.OutputParserException as e:
        # 尝试从错误信息中提取 JSON 字符串并修复
        error_msg = str(e)
        partial_data = {}
        
        if "Function Structure arguments:" in error_msg:
            try:
                # 提取 JSON 字符串
                json_str = error_msg.split("Function Structure arguments:", 1)[1].strip().split('are not valid JSON')[0].strip()
                # 预处理 LaTeX 数学符号 - 使用四个反斜杠来确保正确转义
                json_str = json_str.replace('\\', '\\\\')
                # 尝试解析修复后的 JSON
                partial_data = json.loads(json_str)
            except Exception as json_e:
                print(f"Failed to parse JSON for {item.get('id', 'unknown')}: {json_e}", file=sys.stderr)
        
        # Merge partial data with defaults to ensure all fields exist
        item['AI'] = {**default_ai_fields, **partial_data}
        print(f"Using partial AI data for {item.get('id', 'unknown')}: {list(partial_data.keys())}", file=sys.stderr)
    except Exception as e:
        # Catch any other exceptions and provide default values
        print(f"Unexpected error for {item.get('id', 'unknown')}: {e}", file=sys.stderr)
        item['AI'] = default_ai_fields
    
    # Final validation to ensure all required fields exist
    for field in default_ai_fields.keys():
        if field not in item['AI']:
            item['AI'][field] = default_ai_fields[field]

    # 检查 AI 生成的所有字段
    for v in item.get("AI", {}).values():
        if is_sensitive(str(v)):
            return None
    return item

def process_all_items(data: List[Dict], model_name: str, language: str, max_workers: int) -> List[Dict]:
    """并行处理所有数据项"""
    llm = ChatOpenAI(model=model_name).with_structured_output(Structure, method="function_calling")
    print('Connect to:', model_name, file=sys.stderr)
    
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(template=template)
    ])

    chain = prompt_template | llm
    
    # 使用线程池并行处理
    processed_data = [None] * len(data)  # 预分配结果列表
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(process_single_item, chain, item, language): idx
            for idx, item in enumerate(data)
        }
        
        # 使用tqdm显示进度
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(data),
            desc="Processing items"
        ):
            idx = future_to_idx[future]
            try:
                result = future.result()
                processed_data[idx] = result
            except Exception as e:
                print(f"Item at index {idx} generated an exception: {e}", file=sys.stderr)
                # Add default AI fields to ensure consistency
                processed_data[idx] = data[idx]
                processed_data[idx]['AI'] = {
                    "tldr": "Processing failed",
                    "motivation": "Processing failed",
                    "method": "Processing failed",
                    "result": "Processing failed",
                    "conclusion": "Processing failed"
                }
    
    return processed_data

def main():
    # python enhance.py --data ../data/${today}.jsonl
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    language = os.environ.get("LANGUAGE", "Chinese")
    interests = parse_interests(os.environ.get("INTEREST", ""))

    # 检查并删除目标文件
    target_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f'Removed existing file: {target_file}', file=sys.stderr)

    # 读取数据
    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # 去重
    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data
    print('Open:', args.data, file=sys.stderr)

    data = filter_items_by_interest(
        data=data,
        model_name=model_name,
        language=language,
        interests=interests,
    )
    if not data:
        print("No papers matched INTEREST after model filtering", file=sys.stderr)
        open(target_file, "w").close()
        return
    
    # 并行处理所有数据
    processed_data = process_all_items(
        data,
        model_name,
        language,
        args.max_workers
    )
    
    # 保存结果
    with open(target_file, "w") as f:
        for item in processed_data:
            if item is not None:
                f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()
