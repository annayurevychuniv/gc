#!/usr/bin/env python3
"""
review.py — Vertex AI Code Review bot (REST-based)

Підтримує Gemini 1.5 Flash (gemini-1.5-flash-002) та інші моделі.
"""

import os
import sys
import json
import requests
from typing import List

# -------------------
# Config from env
# -------------------
GCP_PROJECT = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GCP_MODEL = os.getenv("GCP_MODEL", "gemini-1.5-flash-002")  # Gemini або text-bison@001
SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY")
PR_NUMBER_ENV = os.getenv("PR_NUMBER")

GH_HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"token {GITHUB_TOKEN}" if GITHUB_TOKEN else ""
}

# -------------------
# Helpers: GitHub / PR
# -------------------
def get_event_json():
    ev_path = os.getenv("GITHUB_EVENT_PATH")
    if ev_path and os.path.exists(ev_path):
        with open(ev_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def get_pr_info():
    ev = get_event_json()
    if ev and "pull_request" in ev:
        owner = ev["repository"]["owner"]["login"]
        repo = ev["repository"]["name"]
        pr_number = ev["pull_request"]["number"]
        return owner, repo, pr_number
    if GITHUB_REPOSITORY and PR_NUMBER_ENV:
        owner, repo = GITHUB_REPOSITORY.split("/")
        return owner, repo, int(PR_NUMBER_ENV)
    print("ERROR: Could not determine PR info.")
    sys.exit(1)

def list_pr_files(owner: str, repo: str, pr_number: int) -> List[dict]:
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
    files = []
    page = 1
    while True:
        r = requests.get(url, headers=GH_HEADERS, params={"per_page": 100, "page": page}, timeout=30)
        if r.status_code != 200:
            print("Failed to fetch PR files:", r.status_code, r.text)
            break
        data = r.json()
        if not data:
            break
        files.extend(data)
        if len(data) < 100:
            break
        page += 1
    return files

def fetch_raw_content(raw_url: str) -> str:
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    try:
        r = requests.get(raw_url, headers=headers, timeout=30)
        if r.status_code == 200:
            return r.text
    except Exception as e:
        print("Exception fetching raw content:", e)
    return ""

# -------------------
# Helpers: GCP auth + Vertex REST
# -------------------
from google.oauth2 import service_account
from google.auth.transport.requests import Request

def get_access_token(sa_path: str) -> str:
    creds = service_account.Credentials.from_service_account_file(
        sa_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(Request())
    return creds.token

def is_gemini_model(model_id: str) -> bool:
    return "gemini" in model_id.lower()

def vertex_endpoint(project: str, location: str, model: str) -> str:
    if is_gemini_model(model):
        return f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:predict"
    else:
        return f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:predict"

def call_vertex_predict(prompt: str, access_token: str) -> str:
    endpoint = vertex_endpoint(GCP_PROJECT, GCP_LOCATION, GCP_MODEL)
    body = {
        "instances": [{"content": prompt}],
        "parameters": {"temperature": 0.2, "maxOutputTokens": 512}
    }
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

    try:
        r = requests.post(endpoint, headers=headers, json=body, timeout=90)
    except Exception as e:
        return f"(Vertex request failed: {e})"
    
    if r.status_code not in (200, 201):
        print("Vertex error:", r.status_code, r.text)
        return f"(Vertex error {r.status_code}: {r.text})"
    
    resp_json = r.json()
    # Extract first content from response
    preds = resp_json.get("predictions", [])
    if preds and isinstance(preds[0], dict) and "content" in preds[0]:
        return preds[0]["content"]
    return json.dumps(resp_json, ensure_ascii=False, indent=2)

# -------------------
# Comment building / posting
# -------------------
def build_comment(reviews: List[dict]) -> str:
    lines = ["## Vertex AI — Automated Code Review (PoC)\n",
             "Automated review suggestions:\n"]
    for r in reviews:
        lines.append("---")
        lines.append(f"**File:** `{r['path']}`\n")
        lines.append(r["review"] + "\n")
    lines.append("*This is an automated comment.*")
    return "\n".join(lines)

def post_pr_comment(owner: str, repo: str, pr_number: int, body: str):
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    r = requests.post(url, headers=GH_HEADERS, json={"body": body})
    if r.status_code in (200, 201):
        print("Comment posted to PR")
    else:
        print("Failed to post comment:", r.status_code, r.text)

# -------------------
# File utility
# -------------------
TEXT_FILE_EXTENSIONS = {".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".md", ".txt", ".json", ".yaml", ".yml", ".html", ".css", ".sh", ".rb", ".php"}

def is_text_file(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in TEXT_FILE_EXTENSIONS)

# -------------------
# Main
# -------------------
def main():
    owner, repo, pr_number = get_pr_info()
    print(f"Reviewing PR: {owner}/{repo}#{pr_number}")

    files = list_pr_files(owner, repo, pr_number)
    if not files:
        print("No files to review")
        return

    try:
        token = get_access_token(SERVICE_ACCOUNT_JSON)
    except Exception as e:
        print("Failed to get access token:", e)
        sys.exit(1)

    reviews = []
    for f in files:
        filename = f.get("filename")
        raw_url = f.get("raw_url")
        if not is_text_file(filename):
            continue
        content = fetch_raw_content(raw_url)
        if not content:
            continue
        MAX_CHARS = 25000
        if len(content) > MAX_CHARS:
            content = content[:MAX_CHARS] + "\n\n...file truncated..."
        prompt = (
            "You are a senior software engineer and code reviewer.\n"
            "Provide concise, actionable review comments as bullet points.\n"
            f"Review file: {filename}\n\n```{content}```"
        )
        review_text = call_vertex_predict(prompt, token)
        reviews.append({"path": filename, "review": review_text})

    if reviews:
        comment = build_comment(reviews)
        post_pr_comment(owner, repo, pr_number, comment)
    else:
        print("No reviews generated.")

if __name__ == "__main__":
    main()
