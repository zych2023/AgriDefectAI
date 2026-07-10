"""Seed pests table with 36 classes from model config."""
import sys
sys.path.insert(0, '.')
import pymysql, yaml, json
from app.core.config import settings

conn = pymysql.connect(
    host=settings.DB_HOST, port=settings.DB_PORT,
    user=settings.DB_USER, password=settings.DB_PASSWORD,
    database=settings.DB_NAME, charset='utf8mb4'
)
cur = conn.cursor()

cur.execute('SELECT COUNT(*) FROM pests')
print(f'Before: {cur.fetchone()[0]} pests')

cfg = yaml.safe_load(open('../model-training/config.yaml', encoding='utf-8'))
kb = json.load(open('../model-training/knowledge_base.json', encoding='utf-8'))

for cid, name in cfg['class_names'].items():
    info = kb.get(str(cid), {})
    cur.execute(
        'INSERT IGNORE INTO pests (id, name, type, symptoms, prevention) VALUES (%s, %s, %s, %s, %s)',
        (int(cid) + 1, name, 'disease', name, info.get('advice', '')[:500])
    )

conn.commit()
cur.execute('SELECT COUNT(*) FROM pests')
print(f'After: {cur.fetchone()[0]} pests')
conn.close()
print('Done')
