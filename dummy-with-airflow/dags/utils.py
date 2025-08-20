import os
from airflow.providers.postgres.hooks.postgres import PostgresHook

def save_file(save_dir: str, file_name: str, data: bytes):
    filepath = os.path.join(save_dir, file_name)
    with open(filepath, 'wb') as f:
        f.write(data)
        
def fetch_keywords_from_db():
    hook = PostgresHook(postgres_conn_id="postgres_image")
    records = hook.get_records("""
        SELECT id, domain, keyword, target_count, current_count, last_page_fetched
        FROM crawl_tasks
        WHERE status IS NULL OR current_count < target_count
        FOR UPDATE SKIP LOCKED
    """)
    return [
        {
            "keyword_id": r[0],
            "domain": r[1],
            "keyword": r[2],
            "target_count": r[3],
            "current_count": r[4],
            "last_page_fetched": r[5],
        }
        for r in records
    ]

def update_crawl_task(kw, new_count, new_page):
    hook = PostgresHook(postgres_conn_id="postgres_image")
    hook.run(
        """
        UPDATE crawl_tasks
        SET current_count = %s,
            last_page_fetched = %s,
            status = 'new',
            updated_at = now()
        WHERE id = %s
        """,
        parameters=(new_count, new_page, kw["keyword_id"])
    )