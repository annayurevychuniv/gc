#!/usr/bin/env python3
"""
review.py — Vertex AI Code Review bot (REST-based)

- Отримує PR info з GITHUB_EVENT_PATH або з REPO+PR_NUMBER (локально).
- Лістає файли PR через GitHub API.
- Для кожного текстового файлу робить запит до Vertex AI Publisher predict endpoint:
  https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:predict
- Формує один коментар у PR з усіма оглядами.

Env vars (в workflow встановлюються автоматично або через secrets):
- GCP_PROJECT_ID
- GCP_LOCATION (наприклад us-central1)
- GCP_MODEL (наприклад text-bison@001 або gemini-1.5-flash)
- GOOGLE_APPLICATION_CREDENTIALS -> шлях до service account JSON (в workflow пишемо /tmp/gcp-key.json)
- GITHUB_TOKEN (в workflow підставляється як ${{ github.token }})
- GITHUB_REPOSITORY (owner/repo) (в workflow підставляється автоматично)
- PR_NUMBER (в workflow підставляється автоматично)
"""

import os
import sys
import json
import requests
from typing import List, Optional
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# -------------------
# Config from env
# -------------------
GCP_PROJECT = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GCP_MODEL = os.getenv("GCP_MODEL", "text-bison@001")  # publisher model id (Model Garden)
SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY")
PR_NUMBER_ENV = os.getenv("PR_NUMBER")

# GitHub headers
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
    # fallback for local testing
    if GITHUB_REPOSITORY and PR_NUMBER_ENV:
        owner, repo = GITHUB_REPOSITORY.split("/")
        return owner, repo, int(PR_NUMBER_ENV)
    print("ERROR: Could not determine PR info. Provide GITHUB_EVENT_PATH or set GITHUB_REPOSITORY+PR_NUMBER.")
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
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    try:
        r = requests.get(raw_url, headers=headers, timeout=30)
        if r.status_code == 200:
            return r.text
        print(f"Failed to fetch raw content ({raw_url}): {r.status_code}")
    except Exception as e:
        print("Exception fetching raw content:", e)
    return ""

# -------------------
# Helpers: GCP auth + Vertex predict
# -------------------
def get_access_token_from_service_account(sa_path: str) -> str:
    if not sa_path or not os.path.exists(sa_path):
        raise RuntimeError("Service account JSON not found. Set GOOGLE_APPLICATION_CREDENTIALS to valid path.")
    creds = service_account.Credentials.from_service_account_file(
        sa_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    auth_req = Request()
    creds.refresh(auth_req)
    return creds.token

def vertex_predict_endpoint(project: str, location: str, model: str) -> str:
    # Accept model in forms like "text-bison@001" or "models/text-bison@001"
    model_id = model
    if model_id.startswith("models/"):
        model_id = model_id.split("/", 1)[1]
    # publisher path for Model Garden
    return f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model_id}:predict"

def extract_text_from_vertex_response(resp_json) -> str:
    """
    Try multiple common shapes returned by Vertex:
    - {'predictions': [{'content': '...'}]}
    - {'predictions': [{'candidates': [{'content':'...'}]}]}
    - {'predictions': [{'output': [{'content':'...'}]}]}
    Fall back to stringified JSON.
    """
    if not isinstance(resp_json, dict):
        return str(resp_json)
    preds = resp_json.get("predictions")
    if isinstance(preds, list) and len(preds) > 0:
        p0 = preds[0]
        # candidate style
        if isinstance(p0, dict):
            # common keys
            for k in ("content", "text", "output"):
                if k in p0 and isinstance(p0[k], str):
                    return p0[k]
            # nested candidates
            if "candidates" in p0 and isinstance(p0["candidates"], list) and len(p0["candidates"])>0:
                c0 = p0["candidates"][0]
                if isinstance(c0, dict):
                    if "content" in c0 and isinstance(c0["content"], str):
                        return c0["content"]
                    # some responses place text under 'output' list
                    if "output" in c0 and isinstance(c0["output"], list) and len(c0["output"])>0:
                        out0 = c0["output"][0]
                        if isinstance(out0, dict) and "content" in out0:
                            return out0["content"]
            # output list style
            if "output" in p0 and isinstance(p0["output"], list) and len(p0["output"])>0:
                out0 = p0["output"][0]
                if isinstance(out0, dict):
                    if "content" in out0 and isinstance(out0["content"], str):
                        return out0["content"]
    # fallback: try top-level content
    if "content" in resp_json and isinstance(resp_json["content"], str):
        return resp_json["content"]
    return json.dumps(resp_json, ensure_ascii=False, indent=2)

def call_vertex_predict(prompt: str, access_token: str) -> str:
    if not GCP_PROJECT:
        raise RuntimeError("GCP_PROJECT_ID not set")
    endpoint = vertex_predict_endpoint(GCP_PROJECT, GCP_LOCATION, GCP_MODEL)
    body = {
        "instances": [{"content": prompt}],
        "parameters": {
            "maxOutputTokens": 512
        }
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    try:
        r = requests.post(endpoint, headers=headers, json=body, timeout=60)
    except Exception as e:
        print("Vertex request exception:", e)
        return f"(Vertex request failed: {e})"
    if r.status_code not in (200, 201):
        print("Vertex error:", r.status_code, r.text)
        return f"(Vertex error {r.status_code}: {r.text})"
    resp_json = r.json()
    return extract_text_from_vertex_response(resp_json)

# -------------------
# Comment building / posting
# -------------------
def build_comment(reviews: List[dict]) -> str:
    lines = []
    lines.append("## Vertex AI — Automated Code Review (PoC)\n")
    lines.append("I am an automated reviewer. Suggestions below:\n")
    for r in reviews:
        lines.append("---")
        lines.append(f"**File:** `{r['path']}`")
        lines.append("")
        lines.append(r["review"])
        lines.append("")
    lines.append("\n*This is an automated comment.*")
    return "\n".join(lines)

def post_pr_comment(owner: str, repo: str, pr_number: int, body: str):
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    r = requests.post(url, headers=GH_HEADERS, json={"body": body})
    if r.status_code in (200, 201):
        print("Comment posted to PR")
    else:
        print("Failed to post comment:", r.status_code, r.text)

# -------------------
# Utility: detect likely text file
# -------------------
TEXT_FILE_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".md", ".txt", ".json", ".yaml", ".yml", ".html", ".css", ".sh", ".rb", ".php"
}

def is_text_file(filename: str) -> bool:
    low = filename.lower()
    for ext in TEXT_FILE_EXTENSIONS:
        if low.endswith(ext):
            return True
    # fallback: treat common scripts as text
    return True

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

    # get access token for Vertex
    try:
        token = get_access_token_from_service_account(SERVICE_ACCOUNT_JSON)
    except Exception as e:
        print("Failed to get access token:", e)
        sys.exit(1)

    reviews = []
    for f in files:
        filename = f.get("filename")
        raw_url = f.get("raw_url")
        # skip large binary-ish files by name
        if not is_text_file(filename):
            print("Skipping non-text file:", filename)
            continue
        print("Processing", filename)
        content = fetch_raw_content(raw_url)
        if not content:
            print("No content for", filename)
            continue
        # truncate large files
        MAX_CHARS = 25000
        if len(content) > MAX_CHARS:
            content = content[:MAX_CHARS] + "\n\n...file truncated for brevity..."
        prompt = (
            "You are a senior software engineer and code reviewer.\n"
            "Provide concise, actionable review comments as bullet points. "
            "Mention potential bugs, security issues, and style improvements. "
            "If you suggest code changes, provide short examples.\n\n"
            f"Review file: {filename}\n\n```{content}```\n"
        )
        review_text = call_vertex_predict(prompt, token)
        reviews.append({"path": filename, "review": review_text})

    if not reviews:
        print("No reviews generated.")
        return

    comment = build_comment(reviews)
    post_pr_comment(owner, repo, pr_number, comment)

if __name__ == "__main__":
    main()
