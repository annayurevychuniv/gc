import os
import json
import requests
from github import Github
from google.cloud import aiplatform
from google.oauth2 import service_account

# ==========================
# Конфігурація
# ==========================
GCP_PROJECT = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GCP_MODEL = os.getenv("GCP_MODEL", "text-bison@001")
SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = os.getenv("GITHUB_REPOSITORY")  # owner/repo
PR_NUMBER = int(os.getenv("PR_NUMBER"))

# ==========================
# Ініціалізація Vertex AI
# ==========================
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON)
aiplatform.init(project=GCP_PROJECT, location=GCP_LOCATION, credentials=credentials)
model = aiplatform.ChatModel.from_pretrained(GCP_MODEL)

# ==========================
# Функції
# ==========================
def analyze_code(file_name: str, content: str) -> str:
    chat = model.start_chat()
    prompt = (
        f"You are a senior software engineer. "
        f"Provide concise, actionable review for the following code file `{file_name}`:\n\n"
        f"```{content}```\n\n"
        f"Focus on potential bugs, security issues, and style improvements."
    )
    response = chat.send_message(prompt, temperature=0.2, max_output_tokens=512)
    return response.text

def get_pr_files(owner: str, repo: str, pr_number: int):
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}/files"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.json()

def fetch_file_content(raw_url: str):
    headers = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    r = requests.get(raw_url, headers=headers)
    if r.status_code == 200:
        return r.text
    return ""

def post_comment(owner: str, repo: str, pr_number: int, body: str):
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    r = requests.post(url, headers=headers, json={"body": body})
    if r.status_code in (200, 201):
        print(f"Comment posted to PR #{pr_number}")
    else:
        print("Failed to post comment:", r.status_code, r.text)

# ==========================
# Основна логіка
# ==========================
def main():
    owner, repo = REPO_NAME.split("/")
    pr_number = PR_NUMBER

    files = get_pr_files(owner, repo, pr_number)
    if not files:
        print("No files found in PR.")
        return

    reviews = []
    for f in files:
        file_name = f["filename"]
        raw_url = f["raw_url"]
        print(f"Processing {file_name}...")
        content = fetch_file_content(raw_url)
        if not content:
            continue
        # Обрізаємо великі файли для Vertex AI
        MAX_CHARS = 25000
        if len(content) > MAX_CHARS:
            content = content[:MAX_CHARS] + "\n\n...truncated..."
        review_text = analyze_code(file_name, content)
        reviews.append(f"**{file_name}**:\n{review_text}\n")

    comment_body = "## Vertex AI — Automated Code Review\n\n" + "\n---\n".join(reviews)
    post_comment(owner, repo, pr_number, comment_body)

if __name__ == "__main__":
    main()
