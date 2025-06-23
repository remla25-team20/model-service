import requests

def fetch_releases():
    url = "https://api.github.com/repos/remla25-team20/model-training/releases"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise Exception(f"Failed to fetch releases: {resp.status_code}")
    releases = resp.json()
    version = [release['tag_name'] for release in releases]
    url_models = [release['assets'][0]['browser_download_url'] for release in releases if release['assets']]
    url_cvs = [release['assets'][1]['browser_download_url'] for release in releases if len(release['assets']) > 1]
    return zip(version, url_models, url_cvs)

models  = list(fetch_releases())
for version, url_model, url_cv in models:
    print(f"Version: {version}")
    print(f"Model URL: {url_model}")
    print(f"CV URL: {url_cv}")
    print("-" * 40)

print(models[0][0])  # Print the first version for reference