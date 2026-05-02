import os
import sqlite3
import time
import json
import fitz  # PyMuPDF
from datetime import datetime, timedelta
from curl_cffi import requests as cffi_requests
from google import genai 

# ==============================
# API KEYS & CONFIGURATION
# ==============================
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL       = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

DB_NAME            = "nse_pipeline.db"
BASE_URL           = "https://www.nseindia.com"
ARCHIVE_URL        = "https://nsearchives.nseindia.com"
TARGET_STOCKS_FILE = "stocks.txt"

# Configure New Gemini SDK
client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("[!] WARNING: GEMINI_API_KEY is not set. AI Analysis will fail automatically.")

# ==========================================
# 1. DATABASE & MIGRATION 
# ==========================================

def init_and_migrate_db():
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        
        # 1. Base Table Creation (Historical structure)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS report_pipeline (
                filepath TEXT PRIMARY KEY,
                symbol TEXT,
                download_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                pdf_text TEXT, 
                llm_status TEXT DEFAULT 'PENDING',
                llm_summary TEXT
            )
        ''')
        
        # 2. Database Migration: Check if batch_job_id column exists
        cursor.execute("PRAGMA table_info(report_pipeline)")
        columns = [info[1] for info in cursor.fetchall()]
        
        if 'batch_job_id' not in columns:
            print("[MIGRATION] Applying structural update: Adding 'batch_job_id' to 'report_pipeline'...")
            cursor.execute("ALTER TABLE report_pipeline ADD COLUMN batch_job_id TEXT")
            
        # 3. Create the new Batch tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS active_batches (
                job_id TEXT PRIMARY KEY,
                status TEXT DEFAULT 'POLLING',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

def cleanup_database(days_to_keep=5):
    if not os.path.exists(DB_NAME): return
    print(f"\n[SYSTEM] Cleaning up database (keeping last {days_to_keep} days)...")
    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM report_pipeline WHERE download_time <= datetime('now', '-{days_to_keep} days')")
            conn.commit()
            
        with sqlite3.connect(DB_NAME, isolation_level=None) as conn:
            conn.execute("VACUUM") # Vital to minimize Git file bloat
        print("  [+] Database vacuumed and optimized.")
    except Exception as e:
        print(f"  [X] Database cleanup failed: {e}")

# ==========================================
# 2. FILE EXTRACTION & FILTERING
# ==========================================

def load_target_stocks(filename=TARGET_STOCKS_FILE):
    if not os.path.exists(filename): return set()
    with open(filename, "r", encoding="utf-8") as f:
        return set(line.strip().upper() for line in f if line.strip())

def extract_pdf_text(filepath):
    try:
        with fitz.open(filepath) as doc:
            return "".join(page.get_text() for page in doc)
    except Exception:
        return None

# ==========================================
# 3. BATCH SUBMISSION LOGIC
# ==========================================

def prepare_and_submit_batch(conn):
    if not client: return
    
    cursor = conn.cursor()
    # Fetch only documents that haven't been assigned to a batch yet
    cursor.execute('''
        SELECT filepath, symbol, pdf_text FROM report_pipeline 
        WHERE llm_status = 'PENDING' AND pdf_text IS NOT NULL AND batch_job_id IS NULL
    ''')
    unprocessed = cursor.fetchall()
    
    if not unprocessed:
        print("[SYSTEM] No new documents pending AI analysis.")
        return

    print(f"\n[SYSTEM] Found {len(unprocessed)} files ready for AI analysis. Generating payload...")
    batch_filename = "requests.jsonl"
    
    with open(batch_filename, "w", encoding="utf-8") as f:
        for filepath, symbol, pdf_text in unprocessed:
            
            prompt = f"""You are a professional equity research analyst specializing in event-driven trading strategies (non-earnings based).
Your task is to analyze a corporate announcement and determine whether it creates a short-term trading opportunity based ONLY on strategic/business events, NOT on earnings or financial performance.
---
### You MUST extract and infer the following fields:
1. Company Name
2. Reason to Trade
3. Date
4. Potential (VERYHIGH | HIGH | LOW | IGNORE | NA)
---
## Input Announcement:
<<<
{pdf_text}
>>>"""
            
            # Formatted exactly to the Google GenAI Batch API spec
            batch_request = {
                "key": filepath, 
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}]
                }
            }
            f.write(json.dumps(batch_request) + "\n")
            
    try:
        print("[BATCH] Uploading JSONL payload to Google servers...")
        
        # MIME type fix to prevent GitHub Actions crash
        uploaded_file = client.files.upload(
            file=batch_filename,
            config={'mime_type': 'application/jsonl'}
        )
        
        print(f"[BATCH] Triggering execution for {len(unprocessed)} documents...")
        batch_job = client.batches.create(
            model=GEMINI_MODEL,
            src=uploaded_file.name,
        )
        
        job_id = batch_job.name
        print(f"[SUCCESS] Batch Job started. API ID: {job_id}")
        
        # Lock these records in the database to this specific job ID
        cursor.execute("INSERT INTO active_batches (job_id, status) VALUES (?, 'POLLING')", (job_id,))
        for filepath, _, _ in unprocessed:
            cursor.execute("UPDATE report_pipeline SET batch_job_id = ? WHERE filepath = ?", (job_id, filepath))
            
        conn.commit()
        
    except Exception as e:
        print(f"[X] Batch Submission Error: {e}")
    finally:
        if os.path.exists(batch_filename):
            os.remove(batch_filename)

# ==========================================
# 4. CORE INGESTION PIPELINE
# ==========================================

def run_ingestion():
    init_and_migrate_db()
    target_stocks = load_target_stocks()
    
    # Set up a 3-day lookback window to catch Friday/Weekend announcements
    today = datetime.now()
    start_date = today - timedelta(days=3)
    
    today_str = today.strftime("%d-%m-%Y")
    start_date_str = start_date.strftime("%d-%m-%Y")
    
    download_dir = f"NSE_Reports_{today_str}"
    os.makedirs(download_dir, exist_ok=True)
    
    session = cffi_requests.Session(impersonate="chrome120")
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": f"{BASE_URL}"})
    
    print(f"\n[SYSTEM] Fetching announcements from {start_date_str} to {today_str}...")
    try:
        session.get(BASE_URL, timeout=15)
        time.sleep(2)
        # Use the widened date range in the API URL
        api_url = f"{BASE_URL}/api/corporate-announcements?index=equities&from_date={start_date_str}&to_date={today_str}"
        data = session.get(api_url, timeout=15).json()
    except Exception as e:
        print(f"[ERROR] API or Fetch failed: {e}")
        return

    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        for item in data:
            symbol = item.get("symbol", "UNKNOWN").upper()
            if target_stocks and symbol not in target_stocks: continue

            att_path_str = str(item.get("attchmntFile") or item.get("attchmntText") or item.get("att")).strip()
            if not att_path_str or '.zip' in att_path_str.lower(): continue
                
            if '.pdf' in att_path_str.lower() or '/corporate/' in att_path_str.lower():
                pdf_url = att_path_str if att_path_str.startswith("http") else f"{ARCHIVE_URL}{att_path_str}"
                safe_filename = att_path_str.split('/')[-1].split('?')[0] 
                if not safe_filename.lower().endswith('.pdf'): safe_filename += ".pdf"
                filepath = os.path.join(download_dir, f"{symbol}_{safe_filename}")
                
                # Check DB to prevent re-downloading files we already got 1 or 2 days ago
                cursor.execute('SELECT 1 FROM report_pipeline WHERE filepath = ?', (filepath,))
                if cursor.fetchone(): continue 
                
                try:
                    pdf_response = session.get(pdf_url, timeout=20)
                    if pdf_response.status_code == 200:
                        with open(filepath, 'wb') as f: f.write(pdf_response.content)
                        time.sleep(1) 
                    else: continue
                except Exception: continue
                
                extracted_text = extract_pdf_text(filepath)
                if extracted_text:
                    cursor.execute('''
                        INSERT OR IGNORE INTO report_pipeline (filepath, symbol, pdf_text, llm_status)
                        VALUES (?, ?, ?, 'PENDING')
                    ''', (filepath, symbol, extracted_text))
                    conn.commit()
                
                if os.path.exists(filepath): os.remove(filepath)

        prepare_and_submit_batch(conn)

    cleanup_database(days_to_keep=5)

if __name__ == "__main__":
    run_ingestion()
