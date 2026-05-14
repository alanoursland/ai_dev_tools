import requests

def list_branches(owner, repo, token=None):
    """
    List all branches in a GitHub repository.
    
    :param owner: GitHub username or organization name
    :param repo: Repository name
    :param token: Personal Access Token (optional but recommended)
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/branches"

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
        return

    branches = response.json()
    print(f"Branches in {owner}/{repo}:")
    for b in branches:
        print(" -", b["name"])


# -------------------------------
# Example usage:
# -------------------------------

if __name__ == "__main__":
    owner = "YOUR_GITHUB_USERNAME"
    repo = "YOUR_REPOSITORY_NAME"
    token = "YOUR_GITHUB_PAT"  # optional; remove if not using

    list_branches(owner, repo, token)
