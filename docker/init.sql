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

-- 内置匿名用户，用于兼容旧数据迁移
INSERT INTO users (id, username, password_hash, token_hash, display_name)
VALUES ('00000000-0000-0000-0000-000000000000', '__anonymous__', '', '', '匿名用户')
ON CONFLICT (id) DO NOTHING;

-- 对话表（每个会话一条记录，JSONB 存储全部消息）
CREATE TABLE IF NOT EXISTS conversations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL DEFAULT '00000000-0000-0000-0000-000000000000' REFERENCES users(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    messages JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_conv_user_session ON conversations(user_id, session_id);
