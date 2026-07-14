"""
=============================================================================
问答日志存储模块 — SQLite 本地数据库
=============================================================================
存储每次 API 调用的问题和回答（按 fid 归属），支持历史查询。
"""

import sqlite3
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qa_history.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """初始化数据库（不存在则创建；已有表则补 fid 列）"""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS qa_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            fid   TEXT    NOT NULL DEFAULT '',
            question    TEXT    NOT NULL,
            answer      TEXT    NOT NULL,
            success     INTEGER NOT NULL DEFAULT 1,
            error_msg   TEXT,
            created_at  TEXT    NOT NULL
        )
    """)
    # 兼容旧表：尝试添加 fid 列（已存在则忽略）
    try:
        conn.execute("ALTER TABLE qa_logs ADD COLUMN fid TEXT NOT NULL DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()
    logger.info("问答日志库就绪: %s", DB_PATH)


# ==================== 写入 ====================

def save(fid: str, question: str, answer: str,
         success: bool = True, error_msg: Optional[str] = None) -> int:
    """保存一条问答记录，返回记录 ID"""
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO qa_logs (fid, question, answer, success, error_msg, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (fid, question, answer, int(success), error_msg, datetime.now().isoformat()),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


# ==================== 查询 ====================

def _where_farmer(fid: Optional[str] = None) -> tuple:
    """构建 fid 过滤条件"""
    if fid:
        return ("WHERE fid = ?", [fid])
    return ("", [])


def get_list(fid: Optional[str] = None,
             page: int = 1, page_size: int = 20) -> List[Dict[str, Any]]:
    """分页获取记录（可按 fid 过滤）"""
    conn = _get_conn()
    where, params = _where_farmer(fid)
    rows = conn.execute(
        f"SELECT * FROM qa_logs {where} ORDER BY id DESC LIMIT ? OFFSET ?",
        params + [page_size, (page - 1) * page_size],
    ).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]


def total_count(fid: Optional[str] = None) -> int:
    """总记录数（可按 fid 过滤）"""
    conn = _get_conn()
    where, params = _where_farmer(fid)
    row = conn.execute(
        f"SELECT COUNT(*) as cnt FROM qa_logs {where}", params
    ).fetchone()
    conn.close()
    return row["cnt"] if row else 0


def search(keyword: str, fid: Optional[str] = None,
           page: int = 1, page_size: int = 20) -> List[Dict[str, Any]]:
    """按关键词搜索（可按 fid 过滤）"""
    conn = _get_conn()
    kw = f"%{keyword}%"
    where = "WHERE (question LIKE ? OR answer LIKE ?)"
    params = [kw, kw]
    if fid:
        where += " AND fid = ?"
        params.append(fid)
    rows = conn.execute(
        f"SELECT * FROM qa_logs {where} ORDER BY id DESC LIMIT ? OFFSET ?",
        params + [page_size, (page - 1) * page_size],
    ).fetchall()
    conn.close()
    return [dict(r) for r in reversed(rows)]


def search_count(keyword: str, fid: Optional[str] = None) -> int:
    """搜索结果总数（可按 fid 过滤）"""
    conn = _get_conn()
    kw = f"%{keyword}%"
    where = "WHERE (question LIKE ? OR answer LIKE ?)"
    params = [kw, kw]
    if fid:
        where += " AND fid = ?"
        params.append(fid)
    row = conn.execute(
        f"SELECT COUNT(*) as cnt FROM qa_logs {where}", params
    ).fetchone()
    conn.close()
    return row["cnt"] if row else 0
