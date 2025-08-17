

CREATE TABLE crawl_tasks (
    id SERIAL PRIMARY KEY,
    domain TEXT,
    keyword TEXT,
    target_count INT,
    current_count INT,
    last_page_fetched INT,
    status TEXT DEFAULT NULL, -- new, cleaned, processed, done
    batch_id VARCHAR(50) DEFAULT NULL,
    created_at TIMESTAMP DEFAULT now(),
    updated_at TIMESTAMP DEFAULT now(),
)