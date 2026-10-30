"""Idempotent loan CSV ingestion and read-only dashboard queries for Cassandra."""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

KEYSPACE = "loan_analytics"
DATASET = "Loan_default 2.csv"


def connect(keyspace: bool = True):
    from cassandra.auth import PlainTextAuthProvider
    from cassandra.cluster import Cluster

    host = os.getenv("CASSANDRA_HOST", "127.0.0.1")
    port = int(os.getenv("CASSANDRA_PORT", "9042"))
    username = os.getenv("CASSANDRA_USERNAME")
    password = os.getenv("CASSANDRA_PASSWORD", "")
    auth = PlainTextAuthProvider(username=username, password=password) if username else None
    cluster = Cluster([host], port=port, auth_provider=auth, connect_timeout=5, control_connection_timeout=5)
    return cluster, cluster.connect(KEYSPACE if keyspace else None)


def create_schema(session):
    session.execute("CREATE KEYSPACE IF NOT EXISTS loan_analytics WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}")
    session.set_keyspace(KEYSPACE)
    session.execute("""CREATE TABLE IF NOT EXISTS loan_records (
        loan_id text PRIMARY KEY, age int, income int, loan_amount int, credit_score int,
        months_employed int, num_credit_lines int, interest_rate double, loan_term int,
        dti_ratio double, education text, employment_type text, marital_status text,
        has_mortgage text, has_dependents text, loan_purpose text, has_cosigner text,
        defaulted int)""")
    session.execute("""CREATE TABLE IF NOT EXISTS dataset_meta (
        dataset text PRIMARY KEY, total int, defaults int, ingested_at timestamp)""")
    session.execute("""CREATE TABLE IF NOT EXISTS category_counts (
        category text, value text, total int, defaults int,
        PRIMARY KEY (category, value))""")


def ingest(path: Path):
    from cassandra.concurrent import execute_concurrent_with_args

    cluster, session = connect(keyspace=False)
    try:
        create_schema(session)
        insert = session.prepare("""INSERT INTO loan_records (
            loan_id, age, income, loan_amount, credit_score, months_employed,
            num_credit_lines, interest_rate, loan_term, dti_ratio, education,
            employment_type, marital_status, has_mortgage, has_dependents,
            loan_purpose, has_cosigner, defaulted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""")
        totals = defaultdict(lambda: [0, 0])
        batch = []
        count = defaults = 0
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                label = int(row["Default"])
                args = (
                    row["LoanID"], int(row["Age"]), int(row["Income"]), int(row["LoanAmount"]),
                    int(row["CreditScore"]), int(row["MonthsEmployed"]), int(row["NumCreditLines"]),
                    float(row["InterestRate"]), int(row["LoanTerm"]), float(row["DTIRatio"]),
                    row["Education"], row["EmploymentType"], row["MaritalStatus"],
                    row["HasMortgage"], row["HasDependents"], row["LoanPurpose"],
                    row["HasCoSigner"], label,
                )
                batch.append(args)
                count += 1
                defaults += label
                for field in ("LoanPurpose", "EmploymentType", "Education"):
                    item = totals[(field, row[field])]
                    item[0] += 1
                    item[1] += label
                if len(batch) >= 1000:
                    execute_concurrent_with_args(session, insert, batch, concurrency=64, raise_on_first_error=True)
                    batch.clear()
                    if count % 25000 == 0:
                        print(f"Ingested {count:,} loans", flush=True)
        if batch:
            execute_concurrent_with_args(session, insert, batch, concurrency=64, raise_on_first_error=True)
        summary_insert = session.prepare("INSERT INTO category_counts (category, value, total, defaults) VALUES (?, ?, ?, ?)")
        for (category, value), (total, category_defaults) in totals.items():
            session.execute(summary_insert, (category, value, total, category_defaults))
        session.execute("INSERT INTO dataset_meta (dataset, total, defaults, ingested_at) VALUES (%s, %s, %s, %s)",
                        (path.name, count, defaults, datetime.now(timezone.utc)))
        return {"rows": count, "defaults": defaults, "keyspace": KEYSPACE}
    finally:
        cluster.shutdown()


def dashboard():
    """Return saved Cassandra aggregates, or a clear unavailable state."""
    try:
        cluster, session = connect()
        try:
            meta = session.execute("SELECT total, defaults, ingested_at FROM dataset_meta WHERE dataset = %s", (DATASET,)).one()
            if meta is None:
                return {"connected": True, "ingested": False, "keyspace": KEYSPACE}
            def category_rates(category):
                rows = session.execute("SELECT value, total, defaults FROM category_counts WHERE category = %s", (category,))
                return sorted([{"name": r.value, "total": r.total, "defaults": r.defaults,
                                "rate": round(r.defaults / r.total * 100, 1)} for r in rows], key=lambda item: item["name"])
            return {"connected": True, "ingested": True, "keyspace": KEYSPACE,
                    "total": meta.total, "defaults": meta.defaults,
                    "rate": round(meta.defaults / meta.total * 100, 1),
                    "purposes": category_rates("LoanPurpose"),
                    "employment": category_rates("EmploymentType"),
                    "education": category_rates("Education")}
        finally:
            cluster.shutdown()
    except Exception:
        return {"connected": False, "ingested": False, "keyspace": KEYSPACE}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(DATASET))
    args = parser.parse_args()
    print(ingest(args.data))
