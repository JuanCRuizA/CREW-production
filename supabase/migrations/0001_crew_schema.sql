-- 0001_crew_schema.sql: minimal `crew` schema for the simple deploy (Flask on Render + Supabase Postgres).
--
-- Four tables: clients, client_features (model inputs), scores (model outputs) and score_audit_log.
-- The Flask service connects through the Supavisor pooler as the role `crew_app`; nothing is exposed
-- through PostgREST. Apply once with `supabase db push` or by pasting this file in the SQL Editor.
-- Safe to re-run: every object is created with IF NOT EXISTS or dropped and recreated.

begin;

-- ---------------------------------------------------------------------------
-- Schema. Not in the PostgREST exposed list, and closed to the Supabase API roles
-- (GDPR data minimisation: client data is reachable only by the service that needs it).
-- ---------------------------------------------------------------------------
create schema if not exists crew;
revoke all on schema crew from public, anon, authenticated;

-- ---------------------------------------------------------------------------
-- Clients of the sample (~2,000, stratified by TARGET, fixed seed). sample_seed records which
-- draw produced the row (SR 11-7: the data behind any result must be reproducible).
-- ---------------------------------------------------------------------------
create table if not exists crew.clients (
    sk_id_curr  bigint primary key,
    target      smallint check (target in (0, 1)),
    sample_seed int not null,
    loaded_at   timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- Model inputs: one jsonb object per client with exactly the 216 bundle features, in the
-- POST /api/score input format (NaN as null, +/-inf as "Infinity"/"-Infinity", categoricals as text).
-- Raw values, never encoded: the bundle encodes at scoring time.
-- ---------------------------------------------------------------------------
create table if not exists crew.client_features (
    sk_id_curr  bigint primary key references crew.clients (sk_id_curr),
    feature_set text not null,
    features    jsonb not null check (jsonb_typeof(features) = 'object'),
    loaded_at   timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- Model outputs. Every score carries model_version and bundle_sha256 (SR 11-7 traceability:
-- each PD is tied to the exact model artifact that produced it). result keeps the full API answer.
-- pd_raw and pd_calibrated are both stored; neither is a headline PD for crew-v1.
-- ---------------------------------------------------------------------------
create table if not exists crew.scores (
    score_id          bigint generated always as identity primary key,
    sk_id_curr        bigint not null references crew.clients (sk_id_curr),
    model_version     text not null,
    bundle_sha256     text not null,
    pd_raw            double precision not null,
    pd_calibrated     double precision not null,
    above_statistical boolean not null,
    above_business    boolean not null,
    result            jsonb not null,
    scored_at         timestamptz not null default now()
);
create index if not exists scores_client_recent_idx on crew.scores (sk_id_curr, scored_at desc);

-- ---------------------------------------------------------------------------
-- Audit log, append-only (EU AI Act Art. 12: automatic logging for a high-risk system,
-- credit scoring being Annex III). No UPDATE/DELETE grants, and a trigger rejects them
-- (and TRUNCATE) even for the table owner.
-- ---------------------------------------------------------------------------
create table if not exists crew.score_audit_log (
    audit_id      bigint generated always as identity primary key,
    score_id      bigint not null references crew.scores (score_id),
    sk_id_curr    bigint not null,
    event         text not null check (event in ('scored')),
    source        text not null check (source in ('api', 'ui')),
    model_version text not null,
    bundle_sha256 text not null,
    created_at    timestamptz not null default now()
);

create or replace function crew.reject_audit_change() returns trigger
language plpgsql as $$
begin
    raise exception 'crew.score_audit_log is append-only: % is not allowed', tg_op;
end;
$$;

drop trigger if exists score_audit_log_no_update_delete on crew.score_audit_log;
create trigger score_audit_log_no_update_delete
    before update or delete on crew.score_audit_log
    for each row execute function crew.reject_audit_change();

drop trigger if exists score_audit_log_no_truncate on crew.score_audit_log;
create trigger score_audit_log_no_truncate
    before truncate on crew.score_audit_log
    for each statement execute function crew.reject_audit_change();

-- ---------------------------------------------------------------------------
-- Row Level Security on every table, deny by default. The Supabase API roles get no
-- privileges and no policies, so even an accidental exposure of `crew` returns nothing.
-- ---------------------------------------------------------------------------
alter table crew.clients         enable row level security;
alter table crew.client_features enable row level security;
alter table crew.scores          enable row level security;
alter table crew.score_audit_log enable row level security;

revoke all on all tables in schema crew from public, anon, authenticated;
revoke all on all sequences in schema crew from public, anon, authenticated;
revoke all on function crew.reject_audit_change() from public, anon, authenticated;

-- ---------------------------------------------------------------------------
-- Application role used by the Flask service (least privilege, SR 11-7 / GDPR minimisation):
-- read clients and features, write scores and audit rows, nothing else. No BYPASSRLS:
-- access goes through the explicit policies below.
-- The password is never in this file. Set it once in the SQL Editor:
--   alter role crew_app with password '<generated secret>';
-- ---------------------------------------------------------------------------
do $$
begin
    if not exists (select 1 from pg_roles where rolname = 'crew_app') then
        create role crew_app login noinherit nobypassrls;
    end if;
end;
$$;

grant usage on schema crew to crew_app;
grant select on crew.clients, crew.client_features, crew.scores, crew.score_audit_log to crew_app;
grant insert on crew.scores, crew.score_audit_log to crew_app;

drop policy if exists crew_app_select on crew.clients;
create policy crew_app_select on crew.clients for select to crew_app using (true);

drop policy if exists crew_app_select on crew.client_features;
create policy crew_app_select on crew.client_features for select to crew_app using (true);

-- SELECT on scores is also needed for INSERT ... RETURNING score_id.
drop policy if exists crew_app_select on crew.scores;
create policy crew_app_select on crew.scores for select to crew_app using (true);
drop policy if exists crew_app_insert on crew.scores;
create policy crew_app_insert on crew.scores for insert to crew_app with check (true);

drop policy if exists crew_app_select on crew.score_audit_log;
create policy crew_app_select on crew.score_audit_log for select to crew_app using (true);
drop policy if exists crew_app_insert on crew.score_audit_log;
create policy crew_app_insert on crew.score_audit_log for insert to crew_app with check (true);

commit;
