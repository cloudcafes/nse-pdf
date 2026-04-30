import os
import sqlite3
import json
import time
import requests
import urllib.request
from google import genai 

# ==============================
# API KEYS & CONFIGURATION
# ==============================
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")
DB_NAME            = "nse_pipeline.db"

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# ==========================================
# TELEGRAM ALERTS
# ==========================================

def send_telegram_message(content):
    if not content or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    print("    [TELEGRAM] Dispatching alert...")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    chunks = [content[i:i+4000] for i in range(0, len(content), 4000)]
    for chunk in chunks:
        try:
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk})
        except Exception as e:
            print(f"    [X] Telegram request error: {e}")
        time.sleep(1)

# ==========================================
# BATCH PROCESSING
# ==========================================

def process_batch_results(job_info, conn):
    cursor = conn.cursor()
    output_uri = job_info.output_uri
    
    print(f"[POLL] Downloading results from {output_uri}...")
    try:
        # Handles generic HTTP URIs or native Developer API File identifiers
        if output_uri.startswith("http"):
            response = urllib.request.urlopen(output_uri)
            results_text = response.read().decode("utf-8")
        else:
            response_bytes = client.files.download(name=output_uri)
            results_text = response_bytes.decode("utf-8")
            
        for line in results_text.splitlines():
            if not line.strip(): continue
            data = json.loads(line)
            
            filepath = data.get("key")
            response_payload = data.get("response", {})
            
            # Robust JSON extraction to prevent GitHub Action runner crashes
            if "candidates" in response_payload and response_payload["candidates"]:
                try:
                    result_text = response_payload["candidates"][0]["content"]["parts"][0]["text"].strip()
                    cursor.execute("UPDATE report_pipeline SET llm_status = 'SUCCESS', llm_summary = ? WHERE filepath = ?", (result_text, filepath))
                    
                    result_upper = result_text.upper()
                    if "POTENTIAL: HIGH" in result_upper or "POTENTIAL: VERYHIGH" in result_upper:
                        send_telegram_message(result_text)
                        
                except (KeyError, IndexError):
                    cursor.execute("UPDATE report_pipeline SET llm_status = 'FAILED', llm_summary = 'Parsing Error' WHERE filepath = ?", (filepath,))
            else:
                # Typically hits here if the prompt was flagged by API safety filters
                cursor.execute("UPDATE report_pipeline SET llm_status = 'FAILED', llm_summary = 'Blocked/Empty Response' WHERE filepath = ?", (filepath,))
                
        # Close out the job loop
        cursor.execute("UPDATE active_batches SET status = 'COMPLETED' WHERE job_id = ?", (job_info.name,))
        conn.commit()
        print("[SUCCESS] Database updated and relevant alerts dispatched.")
        
    except Exception as e:
        print(f"[X] Result processing failed: {e}")

def check_active_batches():
    if not client or not os.path.exists(DB_NAME): return

    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        
        # Verify migration actually occurred
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='active_batches'")
        if not cursor.fetchone(): return

        cursor.execute("SELECT job_id FROM active_batches WHERE status = 'POLLING'")
        active_jobs = cursor.fetchall()
        
        if not active_jobs:
            print("[SYSTEM] No active batch jobs to poll.")
            return
            
        for (job_id,) in active_jobs:
            print(f"\n[POLL] Checking status for job: {job_id}")
            try:
                job_info = client.batches.get(name=job_id)
                current_state = job_info.state
                print(f"  -> API State: {current_state}")
                
                if current_state == "JOB_STATE_SUCCEEDED":
                    process_batch_results(job_info, conn)
                elif current_state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
                    print(f"  -> [!] Job {job_id} terminated unproductively.")
                    cursor.execute("UPDATE active_batches SET status = ? WHERE job_id = ?", (current_state, job_id))
                    cursor.execute("UPDATE report_pipeline SET llm_status = ? WHERE batch_job_id = ?", (current_state, job_id))
                    conn.commit()
                    
            except Exception as e:
                print(f"[X] API Error polling {job_id}: {e}")

if __name__ == "__main__":
    check_active_batches()
