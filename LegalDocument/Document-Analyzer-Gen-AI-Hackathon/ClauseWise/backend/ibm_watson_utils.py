import os
import requests

# Optional IBM Granite wrapper - only used if user sets GRANITE_URL and IBM_APIKEY env vars
IBM_GRANITE_URL = os.getenv("IBM_GRANITE_URL", "")
IBM_APIKEY = os.getenv("IBM_APIKEY", "")

def granite_analyze(prompt: str, model="granite-20b-multilingual", max_tokens=512):
    if not IBM_GRANITE_URL or not IBM_APIKEY:
        raise EnvironmentError("IBM Granite not configured. Set IBM_GRANITE_URL and IBM_APIKEY environment variables if you want to use Granite.")
    headers = {
        "Authorization": f"Bearer {IBM_APIKEY}",
        "Content-Type": "application/json"
    }
    payload = {"model": model, "input": prompt, "max_tokens": max_tokens}
    resp = requests.post(IBM_GRANITE_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()
