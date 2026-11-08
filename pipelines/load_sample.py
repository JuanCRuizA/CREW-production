"""Load a stratified client sample from the legacy SQLite into Supabase (crew.clients, crew.client_features).

The sample is stratified by TARGET with a fixed seed, so the same arguments always pick the same clients.
Features are stored raw (categoricals as text, never encoded), in the POST /api/score input format:
NaN as null and +/-inf as "Infinity"/"-Infinity". Every record must carry all 216 bundle features.

Usage:
    python pipelines/load_sample.py --dry-run              # no database: build and validate the rows
    python pipelines/load_sample.py                        # insert; needs DATABASE_URL_ADMIN
    python pipelines/load_sample.py --replace              # empty crew.clients first (only if no scores exist)
"""
import argparse
import json
import math
import os
import pathlib
import sqlite3
from contextlib import closing

import numpy as np
from dotenv import load_dotenv

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = BASE_DIR / "data" / "legacy" / "portfolio_surveillance.db"
PREPROCESSING_PATH = BASE_DIR / "models" / "crew-v1" / "preprocessing.json"
SOURCE_TABLE = "engineered_features"
ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"
DEFAULT_SIZE = 2000
DEFAULT_SEED = 42


class SampleError(Exception):
    """The source data or the sample request cannot produce a valid load."""


def load_feature_spec(path: pathlib.Path = PREPROCESSING_PATH) -> tuple:
    """(feature_set, feature_names) from the bundle: the 216 names in bundle order."""
    pre = json.loads(path.read_text(encoding="utf-8"))
    names = pre["feature_names"]
    if len(names) != pre["n_features"]:
        raise SampleError(f"preprocessing.json lists {len(names)} features but n_features is {pre['n_features']}")
    return pre["model_version"], names


def connect_readonly(db_path: pathlib.Path) -> sqlite3.Connection:
    if not db_path.is_file():
        raise SampleError(f"legacy SQLite not found: {db_path}")
    return sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)


def check_source_columns(conn: sqlite3.Connection, feature_names: list) -> None:
    """Fail loudly if the source table lacks the key, the target or any bundle feature."""
    columns = {row[1] for row in conn.execute(f"pragma table_info({SOURCE_TABLE})")}
    missing = [name for name in [ID_COLUMN, TARGET_COLUMN, *feature_names] if name not in columns]
    if missing:
        raise SampleError(f"{SOURCE_TABLE} is missing {len(missing)} required columns: {missing}")


def allocate(class_counts: dict, size: int) -> dict:
    """Proportional allocation per TARGET class (largest remainder), summing exactly to size."""
    total = sum(class_counts.values())
    if not 0 < size <= total:
        raise SampleError(f"sample size {size} must be between 1 and the population size {total}")
    exact = {k: size * n / total for k, n in class_counts.items()}
    alloc = {k: math.floor(v) for k, v in exact.items()}
    by_remainder = sorted(exact, key=lambda k: (exact[k] - alloc[k], -k), reverse=True)
    for k in by_remainder[: size - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def stratified_ids(population: list, size: int, seed: int) -> list:
    """Pick size (sk_id_curr, target) pairs from population, stratified by target. Deterministic for a seed."""
    by_class = {}
    for sk_id, target in population:
        if target not in (0, 1):
            raise SampleError(f"client {sk_id} has TARGET {target!r}; expected 0 or 1")
        by_class.setdefault(target, []).append(sk_id)
    alloc = allocate({k: len(v) for k, v in by_class.items()}, size)
    rng = np.random.default_rng(seed)
    picked = []
    for target in sorted(by_class):  # fixed class order and sorted ids keep the draw reproducible
        ids = np.array(sorted(by_class[target]), dtype=np.int64)
        chosen = rng.choice(ids, size=alloc[target], replace=False)
        picked.extend((int(i), target) for i in chosen)
    return sorted(picked)


def to_api_value(value):
    """One source value in the POST /api/score input format. Unexpected types are an error."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        raise SampleError(f"unexpected boolean value {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    raise SampleError(f"unexpected value type {type(value).__name__}")


def build_features(row: dict, feature_names: list) -> dict:
    """The 216-key features object for one source row, in bundle order."""
    missing = [name for name in feature_names if name not in row]
    if missing:
        raise SampleError(f"client {row.get(ID_COLUMN)} is missing {len(missing)} features: {missing[:10]}")
    return {name: to_api_value(row[name]) for name in feature_names}


def fetch_rows(conn: sqlite3.Connection, ids: list, feature_names: list) -> list:
    """Source rows (as dicts) for the given ids, in id order. Fails if any id is absent."""
    columns = [ID_COLUMN, *feature_names]
    select = ", ".join(f'"{c}"' for c in columns)
    rows = {}
    for start in range(0, len(ids), 500):  # stay below the SQLite bound-parameter limit
        chunk = ids[start:start + 500]
        marks = ", ".join("?" * len(chunk))
        query = f'select {select} from {SOURCE_TABLE} where "{ID_COLUMN}" in ({marks})'
        for values in conn.execute(query, chunk):
            rows[values[0]] = dict(zip(columns, values))
    absent = [i for i in ids if i not in rows]
    if absent:
        raise SampleError(f"{len(absent)} sampled ids not found in {SOURCE_TABLE}: {absent[:10]}")
    return [rows[i] for i in ids]


def build_sample(conn: sqlite3.Connection, feature_names: list, size: int, seed: int) -> list:
    """[(sk_id_curr, target, features)] for the stratified sample."""
    check_source_columns(conn, feature_names)
    population = list(conn.execute(f'select "{ID_COLUMN}", "{TARGET_COLUMN}" from {SOURCE_TABLE}'))
    picked = stratified_ids(population, size, seed)
    rows = fetch_rows(conn, [sk_id for sk_id, _ in picked], feature_names)
    return [(sk_id, target, build_features(row, feature_names)) for (sk_id, target), row in zip(picked, rows)]


def summarize(sample: list, feature_names: list) -> str:
    targets = {}
    nulls = {}
    for _, target, features in sample:
        targets[target] = targets.get(target, 0) + 1
        if list(features) != feature_names:
            raise SampleError("a features object does not have exactly the bundle features in bundle order")
        for name, value in features.items():
            if value is None:
                nulls[name] = nulls.get(name, 0) + 1
    infinities = sum(1 for _, _, f in sample for v in f.values() if v in ("Infinity", "-Infinity"))
    top_nulls = sorted(nulls.items(), key=lambda kv: -kv[1])[:10]
    return "\n".join([
        f"clients: {len(sample)}  by TARGET: {dict(sorted(targets.items()))}",
        f"features per client: {len(feature_names)} (all present, bundle order)",
        f"columns with nulls: {len(nulls)}  null cells: {sum(nulls.values())}  infinity cells: {infinities}",
        f"top null columns: {top_nulls}",
    ])


def insert_sample(database_url: str, sample: list, feature_set: str, seed: int, replace: bool) -> None:
    """Insert clients and features in one transaction. Refuses to overwrite unless replace is set."""
    import psycopg  # imported here so --dry-run and the unit tests need no database driver
    from psycopg.types.json import Jsonb

    def dumps(obj):
        return json.dumps(obj, allow_nan=False)

    with psycopg.connect(database_url) as conn, conn.transaction():
        existing = conn.execute("select count(*) from crew.clients").fetchone()[0]
        if existing and not replace:
            raise SampleError(f"crew.clients already has {existing} rows; pass --replace to reload")
        if existing:
            scores = conn.execute("select count(*) from crew.scores").fetchone()[0]
            if scores:
                raise SampleError(f"crew.scores has {scores} rows tied to the audit log; --replace is not allowed")
            conn.execute("delete from crew.client_features")
            conn.execute("delete from crew.clients")
        with conn.cursor() as cur:
            cur.executemany(
                "insert into crew.clients (sk_id_curr, target, sample_seed) values (%s, %s, %s)",
                [(sk_id, target, seed) for sk_id, target, _ in sample],
            )
            cur.executemany(
                "insert into crew.client_features (sk_id_curr, feature_set, features) values (%s, %s, %s)",
                [(sk_id, feature_set, Jsonb(features, dumps=dumps)) for sk_id, _, features in sample],
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help=f"clients to sample (default {DEFAULT_SIZE})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"random seed (default {DEFAULT_SEED})")
    parser.add_argument("--db-path", type=pathlib.Path, default=DEFAULT_DB_PATH, help="legacy SQLite (read-only)")
    parser.add_argument("--dry-run", action="store_true", help="build and validate the rows; no database connection")
    parser.add_argument("--replace", action="store_true", help="empty crew.clients first (refused if scores exist)")
    args = parser.parse_args()

    feature_set, feature_names = load_feature_spec()
    with closing(connect_readonly(args.db_path)) as conn:
        sample = build_sample(conn, feature_names, args.size, args.seed)
    print(f"feature_set: {feature_set}  seed: {args.seed}")
    print(summarize(sample, feature_names))
    if args.dry_run:
        print("dry run: nothing written")
        return

    load_dotenv()  # the local .env (git-ignored) may hold DATABASE_URL_ADMIN
    database_url = os.environ.get("DATABASE_URL_ADMIN")
    if not database_url:
        raise SystemExit("DATABASE_URL_ADMIN is not set: export the Supabase Session pooler URL for the postgres role")
    insert_sample(database_url, sample, feature_set, args.seed, args.replace)
    print(f"inserted {len(sample)} clients into crew.clients and crew.client_features")


if __name__ == "__main__":
    main()
