# 智慧农业 RAG 问答接口 — 调用文档

## 一、接口总览

| 接口 | 方法 | 说明 |
|------|------|------|
| `/ask` | POST | 问答（LLM 生成） |
| `/retrieve` | POST | 检索知识文档（无 LLM，供 Agent 用） |
| `/history` | GET | 分页查历史（可选按 fid 过滤） |
| `/history/search` | GET | 关键词搜索历史（可选按 fid 过滤） |
| `/health` | GET | 健康检查 |

---

## 二、问答接口 `POST /ask`

### 请求

```json
{
  "fid": "farmer_001",
  "question": "玉米叶子黄了怎么回事"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `fid` | string | **必填**。由前端传入（登录后缓存），不是用户手输 |
| `question` | string | **必填**。农户输入的自然语言问题 |

### 返回

```json
{
  "answer": "玉米叶子发黄可能由以下几种原因引起...",
  "success": true
}
```

### 三种语言调用示例

**curl：**
```bash
curl -X POST http://10.250.215.164:8899/ask \
  -H "Content-Type: application/json" \
  -d '{"fid":"farmer_001","question":"水稻稻瘟病怎么防治"}'
```

**Vue / JavaScript（前端示例）：**
```javascript
// fid 从登录态取，用户只管输入问题
const fid = localStorage.getItem("fid");
const question = this.inputValue;  // 农户在文本框打的字

const resp = await fetch("http://10.250.215.164:8899/ask", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    fid: fid,   // ← 前端自动带，用户无感
    question: question
  })
});
const data = await resp.json();
console.log(data.answer);
```

**Python：**
```python
import requests
resp = requests.post(
    "http://10.250.215.164:8899/ask",
    json={"fid": "farmer_001", "question": "小麦锈病有什么症状"}
)
print(resp.json()["answer"])
```

---

## 三、检索接口 `POST /retrieve`（供外部 Agent 使用）

对方 Agent 有自己的 LLM 时，调这个接口拿知识文档，自己拼 prompt 生成回答。

### 请求

```json
{
  "fid": "farmer_001",
  "question": "渤海湾地区苹果施肥建议"
}
```

### 返回

```json
{
  "documents": [
    {
      "content": "渤海湾地区苹果施肥建议：基肥以有机肥为主...",
      "source_file": "施肥1.txt",
      "page": 5,
      "score": 0.95
    },
    {
      "content": "苹果树追肥时期：萌芽前追施氮肥...",
      "source_file": "施肥2.txt",
      "page": 12,
      "score": 0.87
    }
  ]
}
```

| 字段 | 说明 |
|------|------|
| `content` | 知识片段文本 |
| `source_file` | 来源文件名 |
| `page` | 页码（非 PDF 为 -1） |
| `score` | 相关性得分，越高越相关 |

### Agent 使用示例

**Python：**
```python
import requests

def get_knowledge(fid: str, question: str) -> list:
    """从农业 RAG 获取相关知识文档"""
    resp = requests.post(
        "http://10.250.215.164:8899/retrieve",
        json={"fid": fid, "question": question},
        timeout=30
    )
    return resp.json()["documents"]

# Agent 拿到文档后用自己的 LLM 生成
docs = get_knowledge("farmer_001", "黄土高原的苹果施肥建议")
context = "\n".join(d["content"] for d in docs)

# 把 context 拼进自己的 prompt 发给自己的 LLM
```

**OpenAI Function Calling：**
```python
# 工具定义
{
  "type": "function",
  "function": {
    "name": "retrieve_agri_knowledge",
    "description": "从农业知识库检索相关知识文档，返回原始文本片段",
    "parameters": {
      "type": "object",
      "properties": {
        "question": {"type": "string", "description": "农业问题"}
      },
      "required": ["question"]
    }
  }
}
```

### /ask vs /retrieve

| | `/ask` | `/retrieve` |
|------|------|------|
| 谁用 | 普通用户 / 前端 | 外部 Agent |
| 返回 | LLM 生成的回答 | 原始知识文档 |
| LLM 调用 | 是（服务端） | 否（Agent 自己来） |
| 耗时 | 3~8s | 1~3s |

---

## 四、健康检查

```
GET http://10.250.215.164:8899/health
```

```json
{"status": "ok", "records": 1115, "qa_history_count": 128}
```

---

## 五、历史记录接口

> 每次调用 `/ask` 后，问题、回答、fid 和时间会自动存入数据库。

### 3.1 分页查历史 `GET /history`

```
GET /history?fid=farmer_001&page=1&page_size=20
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `fid` | string | 否 | 过滤指定农户，**不传则查全部** |
| `page` | int | 否 | 页码，默认 1 |
| `page_size` | int | 否 | 每页条数，默认 20，无上限 |

### 3.2 关键词搜索 `GET /history/search`

```
GET /history/search?keyword=水稻&fid=farmer_001&page=1
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `keyword` | string | **是** | 搜索词，匹配问题或回答内容 |
| `fid` | string | 否 | 过滤指定农户 |
| `page` | int | 否 | 页码 |
| `page_size` | int | 否 | 每页条数 |

### 3.3 返回格式

```json
{
  "total": 128,
  "page": 1,
  "page_size": 20,
  "items": [
    {
      "id": 128,
      "fid": "farmer_001",
      "question": "玉米叶子黄了怎么回事",
      "answer": "玉米叶子发黄可能由...",
      "success": true,
      "created_at": "2026-07-11T14:30:25"
    }
  ]
}
```

### 3.4 调用示例

```bash
# 查 farmer_001 的全部历史
curl "http://10.250.215.164:8899/history?fid=farmer_001&page=1"

# 查全部农户的历史（不传 fid）
curl "http://10.250.215.164:8899/history?page=1"

# 搜索 farmer_001 关于"施肥"的记录
curl "http://10.250.215.164:8899/history/search?keyword=施肥&fid=farmer_001"
```

**Vue 前端示例：**
```javascript
const fid = localStorage.getItem("fid");

// 查当前农户的历史
const resp = await fetch(
  `http://10.250.215.164:8899/history?fid=${fid}&page=1`
);
const data = await resp.json();
// data.total → 总条数
// data.items → 问答列表
```

---

## 六、Agent / LLM 调用方式

如果对方是一个 AI Agent（如 LangChain Agent、Coze、Dify、自定义 LLM Agent 等），通过 Function Calling / Tool 机制调用本接口。

### 5.1 工具定义（OpenAI Function Calling 格式）

```json
{
  "type": "function",
  "function": {
    "name": "query_agriculture_knowledge",
    "description": "查询农业知识库，获取病虫害识别、种植管理、施肥用药等方面的专业建议。当用户提出农业相关问题时调用此工具。",
    "parameters": {
      "type": "object",
      "properties": {
        "fid": {
          "type": "string",
          "description": "农户唯一标识，由系统自动传入"
        },
        "question": {
          "type": "string",
          "description": "农户提出的农业问题，保留原始口语化表述"
        }
      },
      "required": ["fid", "question"]
    }
  }
}
```

### 5.2 Agent 调用示例

**Python（LangChain Agent）：**
```python
import requests

def query_agriculture_knowledge(fid: str, question: str) -> str:
    """农业知识库查询工具"""
    resp = requests.post(
        "http://10.250.215.164:8899/ask",
        json={"fid": fid, "question": question},
        timeout=60
    )
    data = resp.json()
    if data.get("success"):
        return data["answer"]
    return f"查询失败"

# 注册为 LangChain Tool
from langchain.tools import tool

@tool
def agri_qa(fid: str, question: str) -> str:
    """当用户咨询农业病虫害、种植、施肥相关问题时使用"""
    return query_agriculture_knowledge(fid, question)
```

**Python（OpenAI Function Calling）：**
```python
import requests, json
from openai import OpenAI

client = OpenAI(api_key="...")

def call_agri_api(fid: str, question: str) -> str:
    resp = requests.post(
        "http://10.250.215.164:8899/ask",
        json={"fid": fid, "question": question},
        timeout=60
    )
    return resp.json()["answer"]

# Agent 主循环
def agent_run(user_message: str, fid: str):
    messages = [{"role": "user", "content": user_message}]

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=[{
            "type": "function",
            "function": {
                "name": "query_agriculture_knowledge",
                "description": "查询农业知识库获取专业建议",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "农业问题"}
                    },
                    "required": ["question"]
                }
            }
        }]
    )

    # 如果模型决定调用工具
    tool_call = response.choices[0].message.tool_calls[0]
    if tool_call.function.name == "query_agriculture_knowledge":
        args = json.loads(tool_call.function.arguments)
        answer = call_agri_api(fid, args["question"])

        # 把结果返回给模型继续对话
        messages.append({"role": "tool", "content": answer, "tool_call_id": tool_call.id})
        final = client.chat.completions.create(model="gpt-4", messages=messages)
        return final.choices[0].message.content
```

**Dify / Coze 等平台：**
- 添加自定义 API 工具
- Method: `POST`
- URL: `http://10.250.215.164:8899/ask`
- Body: `{"fid": "{{fid}}", "question": "{{query}}"}`
- 提取回答路径: `$.answer`



---

## 七、注意事项

| 事项 | 说明 |
|------|------|
| fid 来源 | 由前端自动传入（如登录后缓存），不是农户手输 |
| fid 过滤 | `/history` 不传则查全部，传了则只看该农户的 |
| 防火墙 | 确保 Windows 防火墙放行 8899 端口 |
| IP 变更 | IP 变了用 `--host` 指定新地址后重启 |
| 首次请求稍慢 | Cross-Encoder 模型首次加载需几秒 |
