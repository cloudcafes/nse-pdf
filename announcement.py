import os
import time
import sqlite3
import fitz  # PyMuPDF
from datetime import datetime
from curl_cffi import requests as cffi_requests
import requests  # Standard requests for Telegram API
from google import genai 

# ==============================
# API KEYS & CONFIGURATION
# ==============================
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL       = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

DB_NAME = "nse_pipeline.db"
BASE_URL = "https://www.nseindia.com"
ARCHIVE_URL = "https://nsearchives.nseindia.com"
ANALYSIS_FILE = "latest-analysis.txt"
TARGET_STOCKS_FILE = "stocks.txt"

# Configure New Gemini SDK
client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("[!] WARNING: GEMINI_API_KEY is not set. AI Analysis will fail automatically.")

# ==========================================
# 1. DATABASE, FILTERING, & EXTRACTION 
# ==========================================

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
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
    conn.commit()
    conn.close()

def load_target_stocks(filename=TARGET_STOCKS_FILE):
    """Loads the allowed stocks from a text file into an efficient Set."""
    if not os.path.exists(filename):
        print(f"[!] WARNING: '{filename}' not found. No AI analysis will run.")
        return set()
        
    with open(filename, "r", encoding="utf-8") as f:
        # Reads lines, removes spaces, converts to UPPERCASE, ignores blank lines
        return set(line.strip().upper() for line in f if line.strip())

def extract_pdf_text(filepath):
    print(f"    [PROCESS] Extracting text from {os.path.basename(filepath)}...")
    text_content = ""
    try:
        with fitz.open(filepath) as doc:
            for page in doc:
                text_content += page.get_text()
        return text_content
    except Exception as e:
        print(f"    [X] PDF Extraction Failed: {e}")
        return None

def register_download_and_text(filepath, symbol, pdf_text):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR IGNORE INTO report_pipeline (filepath, symbol, pdf_text, llm_status)
        VALUES (?, ?, ?, 'PENDING')
    ''', (filepath, symbol, pdf_text))
    conn.commit()
    conn.close()

def get_pending_text_for_llm():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT filepath, symbol, pdf_text FROM report_pipeline 
        WHERE llm_status IN ('PENDING', 'FAILED') AND pdf_text IS NOT NULL
    ''')
    results = cursor.fetchall()
    conn.close()
    return results

def update_llm_status(filepath, status, summary=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    if summary:
        cursor.execute('UPDATE report_pipeline SET llm_status = ?, llm_summary = ? WHERE filepath = ?', (status, summary, filepath))
    else:
        cursor.execute('UPDATE report_pipeline SET llm_status = ? WHERE filepath = ?', (status, filepath))
    conn.commit()
    conn.close()

def print_db_summary():
    if not os.path.exists(DB_NAME): return
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    print("\n" + "="*50 + "\n📊 DATABASE PIPELINE SUMMARY\n" + "="*50)
    try:
        cursor.execute("SELECT COUNT(*) FROM report_pipeline")
        print(f"Total Documents Tracked: {cursor.fetchone()[0]}")
        cursor.execute("SELECT COUNT(*) FROM report_pipeline WHERE pdf_text IS NOT NULL")
        print(f"Documents with Extracted Text: {cursor.fetchone()[0]}")
        
        print("\n--- LLM Processing Status ---")
        cursor.execute("SELECT llm_status, COUNT(*) FROM report_pipeline GROUP BY llm_status")
        status_dict = {status: count for status, count in cursor.fetchall()}
        print(f"  ✅ SUCCESS (Analyzed):      {status_dict.get('SUCCESS', 0)}")
        print(f"  ⏭️ SKIPPED (Not in Target): {status_dict.get('SKIPPED', 0)}")
        print(f"  ⏳ PENDING (Waiting):       {status_dict.get('PENDING', 0)}")
        print(f"  ❌ FAILED  (Errors):        {status_dict.get('FAILED', 0)}")
    except Exception as e:
        pass
    print("="*50 + "\n")
    conn.close()

# ==========================================
# 2. AI & TELEGRAM INTEGRATION
# ==========================================

def analyze_with_gemini(symbol, pdf_text, max_retries=2):
    """Sends extracted text to Gemini, with automatic retry logic on failure."""
    if not client:
        return False, "API Key Missing"
        
    prompt = f"""You are a professional equity research analyst specializing in event-driven trading strategies (non-earnings based).
Your task is to analyze a corporate announcement and determine whether it creates a short-term trading opportunity based ONLY on strategic/business events, NOT on earnings or financial performance.
---
### You MUST extract and infer the following fields:
1. Company Name
2. Reason to Trade:
    Explain in 1–2 concise lines WHY this announcement may impact stock price
    Focus ONLY on event-driven triggers
    If no actionable insight → return "NA"
3. Date:
    Extract from the announcement or metadata
    Format: DD-MM-YYYY
4. Potential:
   Classify into ONE of the following based on expected market impact of the event:
    VERYHIGH → Transformational or large-scale event
     (e.g., major acquisition/takeover, very large order win, significant capex, strategic partnership with high impact)
    HIGH → Strong positive/negative signal but not transformational
     (e.g., moderate order win, small acquisition, expansion announcement, new business segment entry)
    LOW → Minor or limited impact
     (e.g., small contracts, routine expansion, non-material updates)
    IGNORE → Routine/non-actionable disclosures
     (e.g., compliance filings, board meeting notices, procedural updates, general clarifications)
    NA → If unclear or insufficient information
---
### ⚠️ STRICT ANALYSIS GUIDELINES
#### ✅ Focus ONLY on these event categories:
 Capex / Expansion plans
 Acquisition / Takeover / Merger
 Order wins / Contracts
 Strategic partnerships / JV
 Business restructuring / divestment
 New product / segment entry
 Government approvals / licenses (business-impacting)
 Large deal pipeline announcements
 Regulatory approvals with business impact
#### ❌ Explicitly IGNORE:
 Earnings / results / financial performance
 Revenue, profit, margins, EBITDA
 Guidance based on financial metrics
 Any ratio-based or valuation-based commentary
 If the announcement is primarily financial → Potential = IGNORE
---
### ⚙️ Evaluation Logic (Important)
 Evaluate size, strategic importance, and scalability
 Check if the event:
   Changes future revenue visibility
   Signals institutional interest
   Indicates business expansion or consolidation
 Avoid overrating vague MoUs or non-binding announcements
 Penalize lack of numbers / deal size clarity
---
### 🚫 Anti-Hallucination Rule
 Do NOT assume deal size or impact if not explicitly mentioned
 If key details are missing → downgrade to LOW or NA
---
### 📤 Output Format (STRICT — no extra text)
Company Name: <value>
Reason to Trade: <value or NA>
Date: <DD-MM-YYYY or NA>
Potential: <VERYHIGH | HIGH | LOW | IGNORE | NA>
---
## Input Announcement:
<<<
{pdf_text}
>>>"""

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            return True, response.text.strip()
        except Exception as e:
            print(f"    [!] AI Warning: API Error on attempt {attempt} for {symbol} -> {e}")
            if attempt < max_retries:
                print("    [*] Retrying in 3 seconds...")
                time.sleep(3)
            else:
                print(f"    [X] Max retries reached for {symbol}. Continuing to next file.")
                return False, str(e)

def send_telegram_message(content):
    if not content: return
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
        
    print("    [TELEGRAM] Dispatching alert...")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    chunks = [content[i:i+4000] for i in range(0, len(content), 4000)]
    for chunk in chunks:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": chunk}
        try:
            requests.post(url, json=payload)
        except Exception as e:
            print(f"    [X] Telegram request error: {e}")
        time.sleep(1)

# ==========================================
# 3. CORE PIPELINE WORKFLOW
# ==========================================

def run_pipeline():
    init_db()
    
    # Load the targeted stocks list
    target_stocks = load_target_stocks()
    print(f"[SYSTEM] Loaded {len(target_stocks)} target stocks from {TARGET_STOCKS_FILE}.")

    today_str = datetime.now().strftime("%d-%m-%Y")
    download_dir = f"NSE_Reports_{today_str}"
    os.makedirs(download_dir, exist_ok=True)
    
    session = cffi_requests.Session(impersonate="chrome120")
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Referer": f"{BASE_URL}/companies-listing/corporate-filings-announcements"
    })
    
    print(f"\n[SYSTEM] Starting pipeline for {today_str}...")
    try:
        session.get(BASE_URL, timeout=15)
        time.sleep(2)
        api_url = f"{BASE_URL}/api/corporate-announcements?index=equities&from_date={today_str}&to_date={today_str}"
        response = session.get(api_url, timeout=15)
        data = response.json()
    except Exception as e:
        print(f"[ERROR] API or Fetch failed: {e}")
        return

    # --- PHASE 1: DOWNLOAD & EXTRACT ---
    for item in data:
        att_path = item.get("attchmntFile") or item.get("attchmntText") or item.get("att")
        symbol = item.get("symbol", "UNKNOWN")
        
        if att_path:
            att_path_str = str(att_path).strip()
            if '.pdf' in att_path_str.lower() or '/corporate/' in att_path_str.lower():
                
                if '.zip' in att_path_str.lower():
                    continue

                pdf_url = att_path_str if att_path_str.startswith("http") else f"{ARCHIVE_URL}{att_path_str}"
                safe_filename = att_path_str.split('/')[-1].split('?')[0] 
                if not safe_filename.lower().endswith('.pdf'): safe_filename += ".pdf"
                filepath = os.path.join(download_dir, f"{symbol}_{safe_filename}")
                
                if not os.path.exists(filepath):
                    print(f"\n  [DOWNLOAD] Fetching {symbol} -> {safe_filename}...")
                    try:
                        pdf_response = session.get(pdf_url, timeout=20)
                        if pdf_response.status_code == 200:
                            with open(filepath, 'wb') as f:
                                f.write(pdf_response.content)
                            time.sleep(2) 
                    except Exception as e:
                        print(f"    [X] Request Error: {e}")
                        continue
                
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute('SELECT 1 FROM report_pipeline WHERE filepath = ?', (filepath,))
                already_in_db = cursor.fetchone()
                conn.close()

                if not already_in_db:
                    extracted_text = extract_pdf_text(filepath)
                    if extracted_text:
                        register_download_and_text(filepath, symbol, extracted_text)

    # --- PHASE 2: LLM ANALYSIS & TELEGRAM ---
    unprocessed = get_pending_text_for_llm()
    
    if unprocessed:
        print(f"\n[SYSTEM] Found {len(unprocessed)} files ready for AI analysis...")
        
        if os.path.exists(ANALYSIS_FILE):
            os.remove(ANALYSIS_FILE)
            
        for filepath, symbol, pdf_text in unprocessed:
            
            # --- NEW LOGIC: Target Stock Filter ---
            if symbol.upper() not in target_stocks:
                print(f"  [SKIP] {symbol} is not in {TARGET_STOCKS_FILE}. Ignoring AI Analysis.")
                # Mark as SKIPPED so it leaves the pending queue forever
                update_llm_status(filepath, 'SKIPPED')
                continue
            # --------------------------------------

            print(f"  [AI] Querying Gemini for {symbol}...")
            
            success, result = analyze_with_gemini(symbol, pdf_text)
            
            if success:
                with open(ANALYSIS_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{result}\n\n{'='*40}\n\n")
                    
                update_llm_status(filepath, 'SUCCESS', result)
                
                result_upper = result.upper()
                if "POTENTIAL: HIGH" in result_upper or "POTENTIAL: VERYHIGH" in result_upper or "POTENTIAL: VERY HIGH" in result_upper:
                    send_telegram_message(result)
                else:
                    print("    [SKIP] Telegram alert bypassed (Potential not High/VeryHigh).")
                    
            else:
                update_llm_status(filepath, 'FAILED')
                
            time.sleep(3) 
            
    else:
        print("\n[SYSTEM] No new documents pending AI analysis.")

    print_db_summary()

if __name__ == "__main__":
    run_pipeline()
