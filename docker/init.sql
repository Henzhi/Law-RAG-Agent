CREATE EXTENSION IF NOT EXISTS vector;

-- 用户表（username + 密码哈希）
CREATE TABLE IF NOT EXISTS users (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    username VARCHAR(64) UNIQUE NOT NULL,
    password_hash VARCHAR(256) NOT NULL DEFAULT '',
    token_hash VARCHAR(128) NOT NULL DEFAULT '',
    display_name VARCHAR(128),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 内置匿名用户
INSERT INTO users (id, username, password_hash, token_hash, display_name)
VALUES ('00000000-0000-0000-0000-000000000000', '__anonymous__', '', '', '匿名用户')
ON CONFLICT (id) DO NOTHING;

-- 对话表
CREATE TABLE IF NOT EXISTS conversations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000' REFERENCES users(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    messages JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_conv_user_session ON conversations(user_id, session_id);

-- ============================================================
-- v0.5: 知识库表（企业级升级）
-- ============================================================

-- 文档主表（法律/司法解释/案例/法规）
CREATE TABLE IF NOT EXISTS documents (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    doc_type VARCHAR(20) NOT NULL,      -- law|interpretation|case|regulation
    title VARCHAR(500) NOT NULL,
    source VARCHAR(500),
    effective_date DATE,
    version INT DEFAULT 1,
    status VARCHAR(20) DEFAULT 'active', -- active|superseded|draft
    superseded_by UUID REFERENCES documents(id),
    original_filename VARCHAR(500),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_docs_type_status ON documents(doc_type, status);

-- 文档块表（pgvector，统一用 halfvec 减少存储和计算量）
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    doc_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_type VARCHAR(20) NOT NULL,     -- article|judgment|summary|guideline
    content TEXT NOT NULL,
    embedding_model VARCHAR(50) NOT NULL,
    embedding HALFVEC(3072),              -- 按最大维度预留，halfvec 减半存储
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);
-- HNSW 索引（检索速度优先）
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON document_chunks USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 200);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON document_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_model ON document_chunks(embedding_model);

-- FAQ 语义缓存表
CREATE TABLE IF NOT EXISTS faq_cache (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    question TEXT NOT NULL,
    question_embed HALFVEC(3072),
    answer TEXT NOT NULL,
    sources JSONB,
    related_laws TEXT[],
    confidence FLOAT,
    hit_count INT DEFAULT 1,
    status VARCHAR(20) DEFAULT 'active',  -- active|expired|invalidated
    created_at TIMESTAMPTZ DEFAULT now(),
    expires_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_faq_embedding
    ON faq_cache USING hnsw (question_embed halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- 对话记忆表
CREATE TABLE IF NOT EXISTS conversation_memories (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id VARCHAR(128) NOT NULL,
    session_id VARCHAR(128) NOT NULL,
    summary TEXT,
    summary_embed HALFVEC(3072),
    entities JSONB,
    message_count INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now(),
    expires_at TIMESTAMPTZ DEFAULT (now() + INTERVAL '30 days')
);
CREATE INDEX IF NOT EXISTS idx_memory_user ON conversation_memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memory_embedding
    ON conversation_memories USING hnsw (summary_embed halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- 检索质量日志表
CREATE TABLE IF NOT EXISTS query_logs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    request_id UUID NOT NULL,
    user_id VARCHAR(128),
    query TEXT NOT NULL,
    intent VARCHAR(20),
    retrieved_count INT,
    reranked_count INT,
    faq_cache_hit BOOLEAN DEFAULT FALSE,
    memory_docs_used INT DEFAULT 0,
    llm_tokens_used INT,
    total_latency_ms INT,
    stage_timings JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);
