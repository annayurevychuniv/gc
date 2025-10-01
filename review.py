#!/usr/bin/env python3 
"""
review.py — Vertex AI Code Review bot (Google GenAI SDK, works with Gemini)

Вимоги:
- requirements.txt містить google-genai та requests
- Workflow записує сервісний ключ у /tmp/gcp-key.json і встановлює
  GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-key.json
- Secrets: GCP_PROJECT_ID, GCP_LOCATION, GCP_MODEL, GCP_KEY_JSON
"""

import os
import sys
import json
import requests
from typing import List
from google import genai

# -------------------
# Config from env
# -------------------
GCP_PROJECT = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GCP_MODEL = os.getenv("GCP_MODEL", "gemini-2.5-flash")
SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY")
PR_NUMBER_ENV = os.getenv("PR_NUMBER")

# GitHub headers
GH_HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"token {GITHUB_TOKEN}" if GITHUB_TOKEN else ""
}

# debug -> бачимо в логах Actions
print("DEBUG: GCP_PROJECT =", GCP_PROJECT)
print("DEBUG: GCP_LOCATION =", GCP_LOCATION)
print("DEBUG: GCP_MODEL =", GCP_MODEL)
print("DEBUG: GOOGLE_APPLICATION_CREDENTIALS exists:", os.path.exists(SERVICE_ACCOUNT_JSON or ""))

# -------------------
# GitHub helpers
# -------------------
def get_pr_info():
    if GITHUB_REPOSITORY and PR_NUMBER_ENV:
        owner, repo = GITHUB_REPOSITORY.split("/")
        return owner, repo, int(PR_NUMBER_ENV)
    print("ERROR: GITHUB_REPOSITORY or PR_NUMBER not set")
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
        print(f"Failed to fetch raw content ({raw_url}): {r.status_code}")
    except Exception as e:
        print("Exception fetching raw content:", e)
    return ""

# -------------------
# GenAI client init and call
# -------------------
def init_genai_client():
    if not GCP_PROJECT:
        raise RuntimeError("GCP_PROJECT_ID not set")
    # genai uses Application Default Credentials when vertexai=True and GOOGLE_APPLICATION_CREDENTIALS set
    client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    return client

def call_genai(client, prompt: str) -> str:
    """
    Use 'contents' parameter (string or list) per google-genai examples.
    Return response.text (convenient aggregated string).
    """
    try:
        resp = client.models.generate_content(model=GCP_MODEL, contents=prompt)
        # many SDK versions expose resp.text
        text = getattr(resp, "text", None)
        if text:
            return text
        # fallback: try output_text or output
        text2 = getattr(resp, "output_text", None)
        if text2:
            return text2
        # fallback stringify
        return json.dumps(resp.__dict__, default=str, ensure_ascii=False)
    except Exception as e:
        print("GenAI model call failed:", e)
        return f"(GenAI call failed: {e})"

# -------------------
# Build + post comment
# -------------------
def build_comment(reviews: List[dict]) -> str:
    lines = ["## Vertex AI — Automated Code Review (Gemini)\n",
             "I am an automated reviewer. Suggestions below:\n"]
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
# Utility
# -------------------
TEXT_FILE_EXTENSIONS = {".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".md", ".txt", ".json", ".yaml", ".yml", ".html", ".css", ".sh", ".rb", ".php"}
def is_text_file(filename: str) -> bool:
    low = filename.lower()
    return any(low.endswith(ext) for ext in TEXT_FILE_EXTENSIONS)

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

    client = init_genai_client()
    reviews = []
    for f in files:
        filename = f.get("filename")
        raw_url = f.get("raw_url")
        if not is_text_file(filename):
            print("Skipping non-text file:", filename)
            continue
        print("Processing", filename)
        content = fetch_raw_content(raw_url)
        if not content:
            print("No content for", filename)
            continue
        if len(content) > 25000:
            content = content[:25000] + "\n\n...file truncated..."
        prompt = (
            "You are a senior software engineer and code reviewer.\n"
            "Provide concise, actionable review comments as bullet points. Mention bugs, security issues, and style fixes.\n\n"
            f"Review file: {filename}\n\n```{content}```\n"
        )
        review_text = call_genai(client, prompt)
        reviews.append({"path": filename, "review": review_text})

    if not reviews:
        print("No reviews generated.")
        return

    comment = build_comment(reviews)
    post_pr_comment(owner, repo, pr_number, comment)

if __name__ == "__main__":
    main()
