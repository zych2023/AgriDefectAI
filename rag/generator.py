"""
=============================================================================
答案生成模块 — 流式 LLM 生成 + 可追溯引用
=============================================================================
将精排后的参考文档片段与系统提示词拼接，调用 LLM 流式生成最终回答，
并返回包含答案文本和引用来源的结构化结果。
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from openai import OpenAI

from config import config

logger = logging.getLogger(__name__)

# ============================================================================
# 农业 AI 助手专用 System Prompt
# ============================================================================

AGRICULTURE_SYSTEM_PROMPT = """你是一位经验丰富的智慧农业AI助手，专门为农户和农业技术人员提供病虫害识别、种植管理和施肥用药方面的专业建议。

你的回答必须严格遵循以下原则：

1. **基于参考知识**：你只能依据下面提供的"参考知识片段"来回答问题。如果参考知识中没有相关信息，请如实告知用户"该问题在目前的知识库中暂未找到相关信息"，绝不能编造或猜测。

2. **知识融合**：当多个参考知识片段涉及同一问题时，综合各片段的信息给出完整回答，避免仅依赖单一片段。

3. **专业且易懂**：使用专业但不晦涩的语言，确保普通农户也能理解。对专业术语应适当解释。

4. **结构清晰**：对于复杂问题，请分点列出，按症状识别→病因分析→防治建议的顺序组织回答。

5. **安全第一**：涉及农药使用的建议，务必强调安全注意事项（稀释比例、施药时间、安全间隔期等）。

6. **信息不完整时**：如果参考知识只覆盖了部分问题，请先回答能回答的部分，再明确指出哪些方面信息不足。

7. **只使用中文回答**。

---

{reference_context}

---

现在，请根据以上参考知识片段，回答用户的问题。"""


@dataclass
class Citation:
    """引用来源"""
    source_file: str
    page: int = -1
    content_snippet: str = ""

    def to_markdown(self) -> str:
        if self.page >= 0:
            return f"【来源：{self.source_file}，第{self.page}页】"
        return f"【来源：{self.source_file}】"


@dataclass
class GenerationResult:
    """生成结果"""
    answer: str                              # 生成的回答文本
    citations: List[Citation] = field(default_factory=list)   # 引用来源列表
    references: List[Dict[str, Any]] = field(default_factory=list)  # 完整的参考文档信息

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [
                {
                    "source_file": c.source_file,
                    "page": c.page,
                    "content_snippet": c.content_snippet,
                }
                for c in self.citations
            ],
            "references": self.references,
        }


class AnswerGenerator:
    """
    基于 LLM 的流式答案生成器

    特点：
    - 流式输出：通过 yield 逐步返回生成的 token
    - 引用追溯：从参考文档中提取来源信息并随结果返回
    - 结构化输出：返回包含答案文本和引用列表的 GenerationResult
    """

    def __init__(self, llm_cfg: Optional[object] = None):
        self.llm_cfg = llm_cfg or config.llm
        self._client = OpenAI(
            base_url=self.llm_cfg.base_url,
            api_key=self.llm_cfg.api_key,
        )

    # ========================================================================
    # 构建上下文
    # ========================================================================

    def _build_reference_context(self, top_docs: List[Dict[str, Any]]) -> str:
        """
        将精排后的 Top-K 文档拼接为参考知识片段文本。

        格式：
        【参考知识片段 1】（来源：xxx，第X页）
        内容...

        【参考知识片段 2】（来源：xxx）
        内容...
        """
        parts = []
        for i, doc in enumerate(top_docs, 1):
            source = doc.get("source_file", "未知来源")
            page = doc.get("page", -1)
            content = doc.get("content", "").strip()

            if page >= 0:
                header = f"【参考知识片段 {i}】（来源：{source}，第{page}页）"
            else:
                header = f"【参考知识片段 {i}】（来源：{source}）"

            parts.append(f"{header}\n{content}")

        return "\n\n".join(parts)

    def _build_citations(self, top_docs: List[Dict[str, Any]]) -> List[Citation]:
        """从参考文档中提取引用信息"""
        citations = []
        for doc in top_docs:
            citations.append(Citation(
                source_file=doc.get("source_file", "未知"),
                page=doc.get("page", -1),
                content_snippet=doc.get("content", "")[:200],
            ))
        return citations

    # ========================================================================
    # 流式生成
    # ========================================================================

    def generate_stream(
        self,
        question: str,
        top_docs: List[Dict[str, Any]],
    ) -> GenerationResult:
        """
        流式生成回答，同时在终端实时打印。

        Yields each token chunk; returns GenerationResult on completion.
        """
        if not top_docs:
            logger.warning("参考文档为空，LLM 可能无法给出有效回答")
            return GenerationResult(
                answer="抱歉，当前知识库中未找到与该问题相关的信息，无法给出准确回答。",
                citations=[],
                references=[],
            )

        reference_context = self._build_reference_context(top_docs)
        system_prompt = AGRICULTURE_SYSTEM_PROMPT.format(
            reference_context=reference_context
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        logger.info("开始流式生成回答...")
        full_answer = ""

        try:
            stream = self._client.chat.completions.create(
                model=self.llm_cfg.model_name,
                messages=messages,
                temperature=self.llm_cfg.temperature,
                max_tokens=self.llm_cfg.max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_answer += token
                    # 不在这里直接 print，由调用方控制

            logger.info("回答生成完成，共 %d 字符", len(full_answer))

        except Exception as e:
            logger.error("LLM 流式生成失败: %s", e)
            return GenerationResult(
                answer=f"生成回答时发生错误: {str(e)}",
                citations=[],
                references=[],
            )

        citations = self._build_citations(top_docs)
        return GenerationResult(
            answer=full_answer.strip(),
            citations=citations,
            references=top_docs,
        )

    def generate(
        self,
        question: str,
        top_docs: List[Dict[str, Any]],
    ) -> GenerationResult:
        """
        非流式生成（内部仍用流式 API 收集完整结果）。
        """
        return self.generate_stream(question, top_docs)


# ============================================================================
# 便捷函数
# ============================================================================

def generate_answer(
    question: str,
    top_docs: List[Dict[str, Any]],
) -> GenerationResult:
    """便捷函数：生成带引用的回答"""
    gen = AnswerGenerator()
    return gen.generate(question, top_docs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=config.log.format)
    test_docs = [
        {
            "content": "玉米大斑病：主要为害叶片，严重时也可为害叶鞘和苞叶。"
                       "发病初期叶片上出现水渍状青灰色斑点，"
                       "后逐渐扩大为大型梭形或不规则形病斑，边缘暗褐色，"
                       "中央淡褐色。在潮湿条件下，病斑上密生黑色霉层。",
            "source_file": "玉米病害列表.txt",
            "page": 3,
        },
        {
            "content": "玉米缺氮：植株生长缓慢，株型矮小；叶片由下而上逐渐变黄，"
                       "首先是叶尖和叶脉间变黄，后扩展至整个叶片。"
                       "补救措施：追施尿素10-15kg/亩，或叶面喷施1%尿素溶液。",
            "source_file": "施肥1.txt",
            "page": 12,
        },
    ]
    gen = AnswerGenerator()
    result = gen.generate("玉米叶子黄了怎么回事", test_docs)
    print("\n=== 生成结果 ===")
    print(result.answer)
    print("\n=== 引用来源 ===")
    for c in result.citations:
        print(f"  {c.to_markdown()}")
