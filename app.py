"""
Northbridge Bank - Credit Risk Query Engine (Streamlit app)
Converted from the Jupyter notebook. Logic, SQL templates and workflow preserved.
"""

import os
import re
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Credit Risk Query Engine", page_icon="🏦", layout="wide")

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "credit_risk_portfolio.db"
CSV_PATH = BASE_DIR / "credit_test_queries.csv"   # optional (sample questions)

# ---------------------------------------------------------------------------
# Credentials (Streamlit secrets first, then environment variables)
# ---------------------------------------------------------------------------
def _get_secret(name):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)

OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
OPENAI_API_BASE = _get_secret("OPENAI_API_BASE")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not configured. Add it to Streamlit secrets or as an environment variable.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if OPENAI_API_BASE:
    os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE


@st.cache_resource
def get_llms():
    return (
        ChatOpenAI(model="gpt-4o-mini", temperature=0),
        ChatOpenAI(model="gpt-4o", temperature=0),
    )


@st.cache_resource
def get_connection():
    """Read-only SQLite connection."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    return sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, check_same_thread=False)


llm, evaluator_llm = get_llms()

try:
    conn = get_connection()
except Exception as e:
    st.error(f"Could not open the database: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Database schema (given to the LLM)
# ---------------------------------------------------------------------------
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""

# ---------------------------------------------------------------------------
# Verified query templates
# ---------------------------------------------------------------------------
sql_1 = """
SELECT
sm.sector_name,
SUM(total_outstanding) / 1000000 AS total_amt_owed_in_mill,
SUM(CASE WHEN asset_classification IN ('Substandard', 'Doubtful', 'Loss') THEN total_outstanding ELSE 0 END) / 1000000 AS npa_amt_owed_in_mill
FROM loan_master AS lm
JOIN sector_master AS sm ON lm.sector_code = sm.sector_code
GROUP BY sm.sector_name
ORDER BY total_amt_owed_in_mill DESC
"""

sql_2 = """
SELECT
loan_category,
SUM(total_outstanding) / 1000000 AS amt_owed_in_mill,
COUNT(loan_account_number) AS loan_count
FROM loan_master AS lm
GROUP BY loan_category
ORDER BY amt_owed_in_mill DESC
"""

sql_3 = """
SELECT
ifrs9_stage,
reporting_date,
COUNT(loan_account_number) AS loan_count,
SUM(ead_amount) /1000000 AS ead_amt_in_mill,
SUM(ecl_amount) /1000000 AS ecl_amt_in_mill
FROM provisioning AS p
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
"""

sql_4 = """
SELECT
sm.sector_name,
p.reporting_date,
AVG(p.provision_coverage_ratio) AS avg_pro_cov_ratio
FROM provisioning AS p
JOIN loan_master AS lm ON p.loan_account_number = lm.loan_account_number
JOIN sector_master AS sm ON lm.sector_code = sm.sector_code
WHERE reporting_date = '2025-09-30'
GROUP BY sm.sector_name
ORDER BY avg_pro_cov_ratio DESC
"""

sql_5 = """
SELECT
lm.borrower_name,
sm.sector_name,
lm.total_outstanding / 1000000 AS outstanding_amt_in_mill,
lm.asset_classification
FROM loan_master AS lm
JOIN sector_master AS sm ON lm.sector_code = sm.sector_code
ORDER BY outstanding_amt_in_mill DESC
LIMIT 10
"""

sql_6 = """
SELECT
lm.group_name,
COUNT(lm.loan_account_number) AS loan_count,
SUM(lm.total_outstanding) / 1000000 AS total_outstanding_in_mill
FROM loan_master AS lm
WHERE group_name IS NOT NULL
GROUP BY group_name
ORDER BY total_outstanding_in_mill DESC
LIMIT 5
"""

sql_7 = """
SELECT
lm.loan_account_number,
lm.borrower_name,
sm.sector_name,
lm.total_outstanding / 1000000 AS outstanding_amt_in_mill,
lm.days_past_due,
lm.asset_classification
FROM loan_master AS lm
JOIN sector_master AS sm ON lm.sector_code = sm.sector_code
WHERE days_past_due > 0
ORDER BY days_past_due DESC
"""

# NOTE: ordering changed from "ORDER BY days_past_due_column DESC" (alphabetical,
# gave Current, 90+, 61-90...) to MIN(days_past_due) so buckets run Current -> 90+
# as the VQ8 specification requires. Bucket logic and counts are unchanged.
sql_8 = """
SELECT
    CASE
        WHEN lm.days_past_due = 0 THEN 'Current'
        WHEN lm.days_past_due > 0 AND lm.days_past_due <= 30 THEN '1-30'
        WHEN lm.days_past_due > 30 AND lm.days_past_due <= 60 THEN '31-60'
        WHEN lm.days_past_due > 60 AND lm.days_past_due <= 90 THEN '61-90'
        ELSE '90+'
    END AS days_past_due_column,
    COUNT(lm.days_past_due) AS loan_count,
    SUM(lm.total_outstanding) / 1000000 AS total_outstanding_in_mill
FROM loan_master AS lm
GROUP BY days_past_due_column
ORDER BY MIN(lm.days_past_due)
"""

sql_9 = """
SELECT
borrower_id,
previous_rating,
internal_rating,
pd_estimate
FROM borrower_rating AS br
WHERE rating_date = '2025-09-30' AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC
"""

sql_10 = """
SELECT
reporting_date,
SUM(ecl_amount) / 1000000 AS total_ecl_in_mill
FROM provisioning AS p
GROUP BY reporting_date
ORDER BY reporting_date
"""

verified_query_library = {
    "VQ1": {"description": "Sector-wise total outstanding and NPA amount breakdown across all sectors.", "sql": sql_1},
    "VQ2": {"description": "Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)", "sql": sql_2},
    "VQ3": {"description": "IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter", "sql": sql_3},
    "VQ4": {"description": "Average provision coverage ratio by sector for the latest reporting quarter", "sql": sql_4},
    "VQ5": {"description": "Top 10 largest loan exposures by outstanding amount at the borrower level", "sql": sql_5},
    "VQ6": {"description": "Top 5 largest exposures aggregated at the business group level", "sql": sql_6},
    "VQ7": {"description": "All overdue loan accounts with their days past due and asset classification", "sql": sql_7},
    "VQ8": {"description": "Distribution of loans across days-past-due buckets showing aging profile of the portfolio", "sql": sql_8},
    "VQ9": {"description": "Borrowers whose internal rating was downgraded in the latest rating cycle", "sql": sql_9},
    "VQ10": {"description": "Expected credit loss trend across all reporting quarters showing provisioning movement over time", "sql": sql_10},
}

# ---------------------------------------------------------------------------
# Tool 1: Intent classification
# ---------------------------------------------------------------------------
def classify_intent(user_question, query_library):
    library_descriptions = "\n".join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""

### ROLE
You are a data analysis for medium to large bank. Your bank has a multibillion dollar lending portfolio with
2,000 to 8,000 borrowers accounts across 8–12 industry sectors. I have created 10 SQL query tempaltes. You job is read and understand a
user's query and see if it matches one of the 10 existing templates. If one of the templates do not match, please feel free to have new SQL code generated.
The database we will be working with is credit_risk_portfolio database.


### INPUT
User Question:
{user_question}

Available Verified Query Templates:
{library_descriptions}

### INSTRUCTIONS
1. Read the user's question and identify the key aspects of the question.
2. Identify which tables and fields will be needed to answer the question.
3. Compare the intent of the user's question against the 10 SQL templates I have provided.
4. Match on semantic meaning, not exact wording.
Examples:
"How much does a borrower owe" refers to 'total_outstanding' in the loan_master table.
"What is the the borrower's in house rating" refers to 'internal_rating' in the borrower_rating table.
"What is the borrower's business sector" refers to 'sector_code' in the sector_master table.
5. Use one the predefined templates if it matches the intent of the user's question otherwise use the 'generate' approach.
8. If the question requires filters, calculations, or grouping not used in the SQL templates, please generate new SQL code.

### Restrictions
1. Do not change the existing 10 SQL queries.
2. Do not invent new fields, tables, or database relationships.

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# ---------------------------------------------------------------------------
# Tool 2: Query generation
# ---------------------------------------------------------------------------
def generate_query(user_question, schema_context):
    generation_prompt = f"""

### ROLE
You are a senior data analyst who has been asked to generate a SQL lite query because none of the existing templates will answer a user's question.
You currently have access to the user's original question, the database and its schema, and input from the intent classification tool.

### INPUT
User Question:
{user_question}

Database Schema (single source of truth):
{schema_context}

### INSTRUCTIONS
1. Write ONLY ONE executable SQL statement that answers the user question using only the provided schema and this database credit_risk_portfolio.db.
2. You MUST return exactly one query. Never generate multiple queries, alternative queries, or additional SQL statements.
3. Maintain the intent of the question. Do not change requested calculations, groupings, or filters.
4. The query must be read-only. Use SELECT (or WITH ... SELECT). Never use DROP, DELETE, UPDATE, INSERT, ALTER, or TRUNCATE. Do not alter the database in any way.
5. Use the intent classification to determine the output.
   For row-level requests your query should generate an individual record.
   For aggregated requests you can use COUNT(), SUM(), AVG(), MIN(), and MAX() or other SQLite functions.
   If the question is asking for a breakdown by category or group use the GROUP BY clause.
   Do not perform an aggregation unless it is required.
6. If you are certain there is a connection between tables you can use JOIN to connect them.
7. Use only the tables and columns listed in the schema. Do not invent new fields, columns, or relationships.
8. Use only names or relevant data from the tables. Do not invent names.
9. Ensure the query is SQLite compatible.
10. In SQLite, never subtract DATE() or date columns directly (e.g. DATE(a)-DATE(b)) — it silently returns 0; always use julianday(a)-julianday(b) for day differences.
11. There are several fields that have to be converted to millions of dollars.
12. Please include the unit on any number that you return.

### OUTPUT
Return exactly ONE complete, executable SQL statement.
The entire response must consist of that single SQL statement.
Do not return alternative queries or additional SQL statements.
Do not include a trailing semicolon.
Do not include Markdown code blocks, comments, headings, or explanations.
Never repeat the query.
Never provide a second version of the query.

"""

    sql = llm.invoke(generation_prompt).content.strip()
    sql = re.sub(r"^```sql\s*|\s*```$", "", sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r"^```\s*|\s*```$", "", sql, flags=re.MULTILINE).strip()
    return sql


# ---------------------------------------------------------------------------
# Tool 3: Query validation (five checks)
# ---------------------------------------------------------------------------
def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    result = {"passed": False, "failed_check": None, "details": "", "relevance_confidence": None}

    # Check 1: Read-only shape
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "REPLACE", "ATTACH"]

    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
        result["failed_check"] = "read_only_shape"
        result["details"] = "Query must start with SELECT or WITH"
        return result

    for kw in forbidden_keywords:
        if re.search(r"\b" + kw + r"\b", sql_upper):
            result["failed_check"] = "read_only_shape"
            result["details"] = f"Forbidden keyword detected: {kw}"
            return result

    statements = [stmt.strip() for stmt in sqlparse.split(candidate_sql) if stmt.strip()]
    if len(statements) != 1:
        result["failed_check"] = "read_only_shape"
        result["details"] = f"Exactly one SQL statement is required. Found {len(statements)} statements."
        return result

    candidate_sql = statements[0].rstrip().rstrip(";").strip()

    # Check 2: Schema conformance (identifiers compared with real tables/columns)
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())

    referenced_identifiers = re.findall(r"\b[a-z_][a-z0-9_]*\b", candidate_sql.lower())
    sql_keywords = {
        "select", "from", "where", "and", "or", "group", "by", "order", "having",
        "limit", "join", "on", "as", "case", "when", "then", "else", "end", "sum",
        "count", "avg", "min", "max", "round", "desc", "asc", "left", "right",
        "inner", "outer", "distinct", "null", "is", "not", "in", "like", "with",
        "union", "all", "between", "coalesce",
    }
    unknown = [
        tok for tok in referenced_identifiers
        if tok not in sql_keywords and tok not in real_columns
        and tok not in real_tables and not tok.isdigit()
        and tok not in ("s", "l", "p", "r", "e6")
    ]
    # Same as the notebook: 'unknown' is computed (aliases/functions make it noisy)
    # and surfaced in the audit log rather than used as a hard failure.
    result["unrecognized_identifiers"] = sorted(set(unknown))

    # Check 3: Parse-and-plan dry run
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result["failed_check"] = "parse_plan_dry_run"
        result["details"] = f"SQL failed to parse or plan: {str(e)}"
        return result

    # Check 4: LLM relevance
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""

### ROLE
You are a senior data analyst. Your job is to review SQL queries to ensure they answer the question being asked, that they are executable, that they are safe, and that they will not alter the database.

### CONTEXT
{track_context}

### INPUT
User Question: {user_question}

Candidate SQL:
{candidate_sql}

### INSTRUCTIONS
Your job is to ensure that the SQL query is technically correct and that it will return the desired results.

1. Only use tables and columns from the schema that will help answer the question.
2. Check that the shape of the data will be returned correctly.
3. Check to see if the code will return a correct answer.
4. Check that the query will run in the database.
5. Check that the query will not alter the database.
6. Check the filters and groupings.

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}

"""

    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r"\{.*\}", relevance_response, re.DOTALL)
    if json_match:
        try:
            relevance_json = json.loads(json_match.group())
        except json.JSONDecodeError:
            relevance_json = None
        if relevance_json:
            result["relevance_confidence"] = relevance_json.get("confidence", 0.0)
            if relevance_json.get("verdict") == "no" or relevance_json.get("confidence", 0.0) < 0.6:
                result["failed_check"] = "llm_relevance"
                result["details"] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
                return result

    # Check 5: Template integrity (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]["sql"]
        try:
            expected_cols = [d[0] for d in cur.execute(f"{expected_sql} LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"{candidate_sql} LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result["failed_check"] = "template_integrity"
                result["details"] = f"Expected {len(expected_cols)} columns, got {len(actual_cols)}"
                return result
        except sqlite3.Error as e:
            result["failed_check"] = "template_integrity"
            result["details"] = f"Template integrity check failed: {str(e)}"
            return result

    result["passed"] = True
    result["details"] = "All validation checks passed"
    return result


# ---------------------------------------------------------------------------
# Tool 4: Retry generation
# ---------------------------------------------------------------------------
def retry_generation(user_question, failed_sql, error_message, schema_context):
    retry_prompt = f"""
### ROLE
You are a senior SQL developer fixing a query that failed validation.

### INPUT
User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

### INSTRUCTIONS
1. Fix only the specific issue identified by the validation error.
2. Preserve the original intent of the query.
3. Maintain the original output structure. If the user was expecting a single value only return a single value.
4. The revised SQL must be read-only SELECT (or WITH ... SELECT).
5. Do assume the problem that was flagged is the only problem.
6. Do not invent any information. Use only tables and columns from the schema.
7. Ensure the query is SQLite compatible.
8. If you are unable to correct the query, escalate the question for human review.

### OUTPUT
Return ONLY the corrected SQL, with no markdown code blocks, no comments, and no explanation.

"""
    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r"^```sql\s*|\s*```$", "", revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r"^```\s*|\s*```$", "", revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


# ---------------------------------------------------------------------------
# Tool 5: Query execution
# ---------------------------------------------------------------------------
def execute_query(validated_sql, db_connection):
    result = {"dataframe": None, "reasonable": True, "warnings": []}

    df = pd.read_sql_query(validated_sql, db_connection)
    result["dataframe"] = df

    if df.empty:
        result["warnings"].append("Query returned an empty result")

    for col in df.select_dtypes(include="number").columns:
        if (df[col] < 0).any() and "deviation" not in col.lower() and "change" not in col.lower():
            result["warnings"].append(f"Column {col} contains negative values")
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result["warnings"].append(f"Column {col} has {null_count} null values")

    if len(result["warnings"]) > 2:
        result["reasonable"] = False

    return result


# ---------------------------------------------------------------------------
# Tool 6: Response generation
# ---------------------------------------------------------------------------
def generate_response(user_question, dataframe, route, query_id=None):
    response_prompt = f"""

### ROLE
You are a senior data analyst with experience in the banking industry. You have been tasked wiht translating database query results
into accurate and meaninful answers. The important thing would be to pass along meaningful information that could be used by another
data analyst for the bank.
### INPUT
User Question: {user_question}

Query Result Data:
{dataframe.to_string()}

### INSTRUCTIONS
1. Answer the specific question being asked. Make sure to answer the question completely. Do not return data that is not relevant to the question.
2. Only return relevant data. Your answer should be useful to an analyst without further filtering or other work.
3. You are allowed to bring in data from other tables that you think might relate to the question. However, please answer the initial question fully
   first and then add other relevant information if needed.
4. Many of the values are being reported in the millions, so please mention if the field name includes the word millions.
5. Remember that you want the bank to do well. If you do notice a potential issue, please mention that in your answer.
6. Use language that is easy to understand for an entry level data analyst.
7. Ideally your response will be brief. If your answer is going to be longer than four sentences, please answer the posed question as clearly as
   as possible early in your answer. Information that is important but not specific to the question should come later in your reponse.
8. Most of the numbers will be in dollars. Please include units and make sure users know if the dollar amount is in millions of dollars.
9. If the query does not return an answer please say, "No matching records were found."
10. Do not invent any new databases, tables, or fields to answer a question. Only use the database that was provided.

### OUTPUT
Return ONLY the natural language response text, with no markdown headers or bullet points unless truly needed.
{user_question}

{dataframe.to_string()}

"""
    return llm.invoke(response_prompt).content.strip()


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------
def run_pipeline(user_question, db_connection, query_library, schema_context, status=None):
    def note(msg):
        if status is not None:
            status.write(msg)

    log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user_question": user_question,
        "route": None,
        "query_id": None,
        "match_reason": None,
        "candidate_sql": None,
        "gate_result": None,
        "retry_used": False,
        "escalated": False,
        "executed_sql": None,
        "row_count": None,
        "confidence": None,
        "narrative": None,
        "warnings": [],
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log["route"] = classification.get("route", "generated")
    log["query_id"] = classification.get("query_id")
    log["match_reason"] = classification.get("match_reason")
    note(f"**[1] Intent classification:** route=`{log['route']}`, query_id=`{log['query_id']}`")

    # Step 2: Query construction
    if log["route"] == "verified" and log["query_id"] in query_library:
        candidate_sql = query_library[log["query_id"]]["sql"]
        note("**[2] Query construction:** loaded from verified library")
    else:
        log["route"] = "generated"
        log["query_id"] = None
        candidate_sql = generate_query(user_question, schema_context)
        note("**[2] Query construction:** generated fresh SQL")
    log["candidate_sql"] = candidate_sql

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log["query_id"])
    log["gate_result"] = gate
    note(f"**[3] Validation gate:** passed=`{gate['passed']}`, relevance_confidence=`{gate.get('relevance_confidence')}`")

    # Step 4: Retry once on generated track
    if not gate["passed"] and log["route"] == "generated":
        note(f"Retrying after failure: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate["details"], schema_context)
        log["candidate_sql"] = candidate_sql
        log["retry_used"] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log["gate_result"] = gate
        note(f"**[3b] Retry validation:** passed=`{gate['passed']}`, relevance_confidence=`{gate.get('relevance_confidence')}`")

    # Step 5: Escalate if still failing
    if not gate["passed"]:
        log["escalated"] = True
        log["narrative"] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log["confidence"] = "ESCALATED"
        note(f"**[!] Escalated to human analyst:** {gate['details']}")
        return {"log": log, "dataframe": None, **log}

    # Step 6: Execute
    log["executed_sql"] = candidate_sql
    try:
        exec_result = execute_query(candidate_sql, db_connection)
    except Exception as e:
        log["escalated"] = True
        log["narrative"] = f"Query failed during execution. Escalated to human analyst. Error: {e}"
        log["confidence"] = "ESCALATED"
        note(f"**[!] Execution error, escalated:** {e}")
        return {"log": log, "dataframe": None, **log}

    df = exec_result["dataframe"]
    log["row_count"] = len(df)
    log["warnings"] = exec_result["warnings"]
    note(f"**[4] Execute:** {len(df)} rows returned")
    if exec_result["warnings"]:
        note(f"Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    log["narrative"] = generate_response(user_question, df, log["route"], log["query_id"])
    log["confidence"] = gate.get("relevance_confidence")
    note(f"**[5] Response generated:** confidence=`{log['confidence']}`")

    return {"log": log, "dataframe": df, **log}


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
if "audit_trail" not in st.session_state:
    st.session_state.audit_trail = []
if "question_text" not in st.session_state:
    st.session_state.question_text = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None

st.title("🏦 Northbridge Bank — Credit Risk Query Engine")
st.caption(
    "Ask routine commercial lending portfolio questions in plain English. "
    "Every answer shows the SQL used, the raw data, a confidence score, and is logged to an audit trail. "
    "Read-only access. Complex or ambiguous questions are escalated to a human analyst."
)

with st.sidebar:
    st.header("Verified query library")
    for qid, entry in verified_query_library.items():
        st.markdown(f"**{qid}**: {entry['description']}")
    st.divider()
    st.markdown("**Data notes**")
    st.markdown(
        "- Latest provisioning/rating date: 2025-09-30\n"
        "- NPA = Substandard, Doubtful, Loss\n"
        "- Amounts in USD; `_in_mill` / `_millions` columns are in millions of dollars"
    )

# Sample questions (from the test CSV if it is deployed alongside the app)
sample_questions = []
if CSV_PATH.exists():
    try:
        sample_questions = pd.read_csv(CSV_PATH)["User Query"].dropna().tolist()
    except Exception:
        sample_questions = []

if sample_questions:
    choice = st.selectbox("Sample questions (optional)", [""] + sample_questions)
    if choice:
        st.session_state.question_text = choice

user_question = st.text_area(
    "Your question",
    value=st.session_state.question_text,
    height=90,
    placeholder="e.g. How much of our book is in real estate, and how much of that is non-performing?",
)

run_clicked = st.button("Run query", type="primary")

if run_clicked:
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        with st.status("Running pipeline...", expanded=True) as status:
            try:
                out = run_pipeline(user_question.strip(), conn, verified_query_library, database_schema, status=status)
                status.update(label="Pipeline complete", state="complete", expanded=False)
                st.session_state.last_result = out
                st.session_state.audit_trail.append(out["log"])
            except Exception as e:
                status.update(label="Pipeline error", state="error")
                st.error(f"Unexpected error: {e}")
                st.session_state.last_result = None

out = st.session_state.last_result
if out:
    log = out["log"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Route", log["route"])
    c2.metric("Query ID", log["query_id"] or "—")
    conf = log["confidence"]
    c3.metric("Confidence", f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf))
    c4.metric("Rows returned", log["row_count"] if log["row_count"] is not None else "—")

    if log["escalated"]:
        st.error("⚠️ Escalated to human analyst")
        st.write(log["narrative"])
    else:
        st.subheader("Answer")
        st.write(log["narrative"])
        if isinstance(conf, (int, float)) and conf < 0.8:
            st.warning("Confidence is below 0.80. Please have an analyst review this answer.")
        if log["warnings"]:
            st.warning("Result warnings: " + "; ".join(log["warnings"]))

        st.subheader("SQL used")
        st.code(sqlparse.format(log["executed_sql"], reindent=True, keyword_case="upper"), language="sql")

        st.subheader("Raw data")
        st.dataframe(out["dataframe"], use_container_width=True)
        st.download_button(
            "Download results (CSV)",
            out["dataframe"].to_csv(index=False).encode("utf-8"),
            file_name="query_results.csv",
            mime="text/csv",
        )

    with st.expander("Classification and validation details"):
        st.write(f"**Match reason:** {log['match_reason']}")
        st.write(f"**Retry used:** {log['retry_used']}")
        st.json(log["gate_result"])

st.divider()
st.subheader("Audit trail (this session)")
if st.session_state.audit_trail:
    audit_df = pd.DataFrame(
        [
            {
                "timestamp": a["timestamp"],
                "question": a["user_question"],
                "route": a["route"],
                "query_id": a["query_id"],
                "retry_used": a["retry_used"],
                "escalated": a["escalated"],
                "confidence": a["confidence"],
                "rows": a["row_count"],
                "sql": a["executed_sql"] or a["candidate_sql"],
            }
            for a in st.session_state.audit_trail
        ]
    )
    st.dataframe(audit_df, use_container_width=True)
    st.download_button(
        "Download full audit log (JSON)",
        json.dumps(st.session_state.audit_trail, indent=2, default=str).encode("utf-8"),
        file_name="audit_log.json",
        mime="application/json",
    )
else:
    st.info("No queries run yet.")
