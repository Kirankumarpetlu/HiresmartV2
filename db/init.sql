CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS resumes (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_name     TEXT NOT NULL,
    file_path     TEXT NOT NULL,
    parse_method  TEXT,
    clean_text    TEXT NOT NULL,
    skills        TEXT[] DEFAULT '{}',
    years_exp     FLOAT,
    education     TEXT,
    embed_model   TEXT NOT NULL,
    uploaded_at   TIMESTAMP DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_resumes_years ON resumes (years_exp);
CREATE INDEX IF NOT EXISTS idx_resumes_skills ON resumes USING GIN (skills);

CREATE TABLE IF NOT EXISTS resume_chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resume_id   UUID NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
    section     TEXT NOT NULL,
    label       TEXT,
    content     TEXT NOT NULL,
    token_count INT,
    embedding   VECTOR(768) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_resume ON resume_chunks (resume_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON resume_chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS job_descriptions (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title          TEXT,
    raw_text       TEXT NOT NULL,
    must_have      TEXT[] DEFAULT '{}',
    nice_to_have   TEXT[] DEFAULT '{}',
    min_years      FLOAT,
    responsibilities TEXT[] DEFAULT '{}',
    embedding      VECTOR(768),
    created_at     TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS screening_results (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    jd_id         UUID NOT NULL REFERENCES job_descriptions(id) ON DELETE CASCADE,
    resume_id     UUID NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
    status        TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    similarity    FLOAT,
    final_score   FLOAT,
    rank          INT,
    skill_report  JSONB,
    fit_scores    JSONB,
    red_flags     TEXT[] DEFAULT '{}',
    explanation   TEXT,
    recommendation TEXT,
    created_at    TIMESTAMP DEFAULT now(),
    UNIQUE (jd_id, resume_id)
);
CREATE INDEX IF NOT EXISTS idx_results_jd ON screening_results (jd_id, rank);

ALTER TABLE resumes ADD COLUMN IF NOT EXISTS github_username TEXT;
ALTER TABLE resumes ADD COLUMN IF NOT EXISTS github_repo_links TEXT[];
ALTER TABLE screening_results ADD COLUMN IF NOT EXISTS github_json JSONB;
ALTER TABLE screening_results ADD COLUMN IF NOT EXISTS github_score INT;

CREATE TABLE IF NOT EXISTS github_profiles (
    username    TEXT PRIMARY KEY,
    fetched_at  TIMESTAMPTZ NOT NULL,
    payload     JSONB NOT NULL,
    etag        TEXT
);

CREATE TABLE IF NOT EXISTS jd_parse_cache (
    text_hash       TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    parsed          JSONB NOT NULL,
    query_vectors   JSONB NOT NULL,
    created_at      TIMESTAMP DEFAULT now(),
    PRIMARY KEY (text_hash, model_id, prompt_version)
);

CREATE TABLE IF NOT EXISTS embedding_cache (
    text_hash       TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    embedding       VECTOR(768) NOT NULL,
    created_at      TIMESTAMP DEFAULT now(),
    PRIMARY KEY (text_hash, model_id)
);

CREATE TABLE IF NOT EXISTS evaluation_cache (
    resume_id         UUID NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
    content_hash      TEXT NOT NULL,
    requirements_hash TEXT NOT NULL,
    chunks_hash       TEXT NOT NULL,
    model_id          TEXT NOT NULL,
    prompt_version    TEXT NOT NULL,
    verified          JSONB NOT NULL,
    created_at        TIMESTAMP DEFAULT now(),
    PRIMARY KEY (resume_id, content_hash, requirements_hash, chunks_hash, model_id, prompt_version)
);

CREATE TABLE IF NOT EXISTS report_cache (
    results_hash    TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    summary         TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT now(),
    PRIMARY KEY (results_hash, model_id, prompt_version)
);

ALTER TABLE resumes ADD COLUMN IF NOT EXISTS contact_email TEXT;
ALTER TABLE resumes ADD COLUMN IF NOT EXISTS candidate_name TEXT;
ALTER TABLE job_descriptions ADD COLUMN IF NOT EXISTS email_reviewed_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    jd_id UUID NOT NULL REFERENCES job_descriptions(id) ON DELETE CASCADE,
    resume_id UUID NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
    email_type TEXT NOT NULL CHECK (email_type IN ('selected','rejected')),
    to_address TEXT,
    subject TEXT NOT NULL,
    body_text TEXT NOT NULL,
    body_html TEXT NOT NULL,
    template_version TEXT NOT NULL,
    decision_reason JSONB NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('drafted','approved','sending','sent','failed','skipped','no_contact')),
    attempts INT NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    approved_at TIMESTAMPTZ,
    sent_at TIMESTAMPTZ,
    UNIQUE (jd_id,resume_id,email_type)
);
CREATE INDEX IF NOT EXISTS idx_notifications_resume ON notifications (resume_id);

ALTER TABLE job_descriptions ADD COLUMN IF NOT EXISTS ranking_weights JSONB;
ALTER TABLE job_descriptions ADD COLUMN IF NOT EXISTS github_waiting BOOLEAN NOT NULL DEFAULT false;

CREATE TABLE IF NOT EXISTS github_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_id UUID NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
    username TEXT NOT NULL,
    run_id UUID NOT NULL REFERENCES job_descriptions(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN
        ('pending','running','done','rate_limited','blocked_config','partial','forbidden','not_found','failed')),
    priority DOUBLE PRECISION NOT NULL,
    attempts INT NOT NULL DEFAULT 0,
    max_attempts INT NOT NULL DEFAULT 5,
    retry_after TIMESTAMPTZ,
    last_error_category TEXT,
    evidence_coverage TEXT NOT NULL DEFAULT 'none' CHECK (evidence_coverage IN ('full','partial','none')),
    lease_until TIMESTAMPTZ,
    lease_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, candidate_id)
);
CREATE INDEX IF NOT EXISTS idx_github_jobs_claim ON github_jobs (status, retry_after, priority DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_github_jobs_active_username ON github_jobs (username) WHERE status='running';
ALTER TABLE github_jobs ADD COLUMN IF NOT EXISTS network_calls INT NOT NULL DEFAULT 0;
ALTER TABLE github_jobs ADD COLUMN IF NOT EXISTS cache_hits INT NOT NULL DEFAULT 0;
ALTER TABLE github_jobs ADD COLUMN IF NOT EXISTS secondary_limits INT NOT NULL DEFAULT 0;
ALTER TABLE github_jobs ADD COLUMN IF NOT EXISTS primary_limit_hits INT NOT NULL DEFAULT 0;
ALTER TABLE github_jobs ADD COLUMN IF NOT EXISTS reserve_deferrals INT NOT NULL DEFAULT 0;
ALTER TABLE github_jobs ADD COLUMN IF NOT EXISTS config_blocks INT NOT NULL DEFAULT 0;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_constraint WHERE conrelid='github_jobs'::regclass
        AND conname='github_jobs_status_check' AND pg_get_constraintdef(oid) NOT LIKE '%blocked_config%') THEN
        ALTER TABLE github_jobs DROP CONSTRAINT github_jobs_status_check;
        ALTER TABLE github_jobs ADD CONSTRAINT github_jobs_status_check CHECK (status IN
            ('pending','running','done','rate_limited','blocked_config','partial','forbidden','not_found','failed'));
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS github_quota_state (
    bucket TEXT PRIMARY KEY,
    "limit" INT,
    remaining INT,
    reset_at TIMESTAMPTZ,
    blocked_until TIMESTAMPTZ,
    blocked_category TEXT,
    secondary_failures INT NOT NULL DEFAULT 0,
    next_request_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
