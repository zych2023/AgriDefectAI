"""
=============================================================================
查询扩展模块 — LLM 驱动的农业问题多角度扩展
=============================================================================
将农户输入的口语化、碎片化问题自动扩展为 4~5 个语义相近但表达方式
各异的检索子问题，涵盖同义词替换、正规化书面语改写、关键词组合等角度。
"""

import logging
from typing import List, Optional

from openai import OpenAI

from config import config

logger = logging.getLogger(__name__)

# ============================================================================
# 查询扩展专用 System Prompt
# ============================================================================

QUERY_EXPANSION_SYSTEM_PROMPT = """你是一个农业信息检索专家。你的任务是将农户输入的口语化、碎片化的农业问题，扩展为多个语义相近但表达方式各异的检索子问题。

请严格遵循以下规则：
1. 从以下不同角度生成子问题：
   - 同义词/近义词替换：使用农业领域的专业术语替换口语表达
   - 正规化书面语改写：将口语化问题转化为正式的技术描述
   - 关键词组合：使用不同的关键词排列组合
   - 因果/防治角度：从症状原因、防治方法、识别特征等不同侧重点提问
   - 区域性/作物特异性角度：加入地区或作物品种的限定条件
2. 每个子问题必须是独立、完整、可用于检索的自然语言问句
3. 输出格式必须是严格的 JSON 数组，每个元素是一个字符串，不要包含任何其他内容
4. 输出数组长度控制在 4~5 个
5. 全部使用中文"""

QUERY_EXPANSION_USER_TEMPLATE = """农户原始问题：{question}

请将该问题扩展为 4~5 个不同角度的检索子问题，以 JSON 数组格式输出。"""


class QueryExpander:
    """
    基于 LLM 的查询扩展器

    利用大语言模型的语义理解能力，将农户的口语化问题扩展为
    多个正式、多样的检索子问题，提升召回覆盖率和鲁棒性。
    """

    def __init__(self, llm_cfg: Optional[object] = None):
        self.llm_cfg = llm_cfg or config.llm
        self.retrieval_cfg = config.retrieval
        self._client = OpenAI(
            base_url=self.llm_cfg.base_url,
            api_key=self.llm_cfg.api_key,
        )

    def expand(self, question: str) -> List[str]:
        """
        将用户问题扩展为多个检索子问题。

        Args:
            question: 农户原始问题（口语化）

        Returns:
            扩展后的子问题列表（包含原始问题作为第一个元素）
        """
        if not question or not question.strip():
            logger.warning("输入问题为空，跳过扩展")
            return []

        logger.info("正在扩展查询: %s", question[:60])

        try:
            response = self._client.chat.completions.create(
                model=self.llm_cfg.model_name,
                messages=[
                    {"role": "system", "content": QUERY_EXPANSION_SYSTEM_PROMPT},
                    {"role": "user", "content": QUERY_EXPANSION_USER_TEMPLATE.format(
                        question=question
                    )},
                ],
                temperature=0.7,  # 需要一定随机性以保证多样性
                max_tokens=512,
            )

            raw_output = response.choices[0].message.content.strip()
            logger.debug("LLM 原始输出: %s", raw_output)

            # 解析 JSON 数组
            expanded = self._parse_json_array(raw_output)

            if not expanded:
                logger.warning("查询扩展解析为空，回退到仅使用原始问题")
                return [question]

            # 去重并过滤空串，原始问题放在最前
            seen = {question.strip()}
            unique = [question.strip()]
            for q in expanded:
                q_clean = q.strip()
                if q_clean and q_clean not in seen:
                    seen.add(q_clean)
                    unique.append(q_clean)

            # 限制数量
            if len(unique) > 1 + self.retrieval_cfg.num_expanded_queries:
                unique = unique[: 1 + self.retrieval_cfg.num_expanded_queries]

            logger.info("查询扩展完成: 原始 + %d 个子问题", len(unique) - 1)
            for i, q in enumerate(unique):
                logger.info("  [%d] %s", i, q[:80])

            return unique

        except Exception as e:
            logger.error("查询扩展失败: %s", e)
            return [question]

    @staticmethod
    def _parse_json_array(raw: str) -> List[str]:
        """从 LLM 输出中解析 JSON 字符串数组"""
        import json
        import re

        # 尝试直接解析
        try:
            result = json.loads(raw)
            if isinstance(result, list):
                return [str(item) for item in result]
        except json.JSONDecodeError:
            pass

        # 尝试用正则提取 JSON 数组
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return [str(item) for item in result]
            except json.JSONDecodeError:
                pass

        # 最后手段：按行分割
        logger.warning("JSON 解析失败，尝试按行分割")
        lines = [line.strip().lstrip("0123456789.-) ").strip("\"'") for line in raw.split("\n")]
        return [line for line in lines if line]


# ============================================================================
# 便捷函数
# ============================================================================

def expand_query(question: str) -> List[str]:
    """便捷函数：扩展查询"""
    expander = QueryExpander()
    return expander.expand(question)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.log.format)
    test_questions = [
        "玉米叶子黄了怎么回事",
        "水稻稻瘟病怎么防治",
    ]
    for q in test_questions:
        expanded = expand_query(q)
        print(f"\n原始: {q}")
        for i, eq in enumerate(expanded):
            print(f"  [{i}] {eq}")
