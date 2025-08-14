# backend/granite_client.py
import os
import requests

IBM_APIKEY = os.getenv("IBM_APIKEY")
IBM_GRANITE_URL = os.getenv("IBM_GRANITE_URL")  # e.g., watsonx endpoint

def call_granite(prompt: str, model="granite-20b-multilingual", max_tokens=512):
    if not IBM_GRANITE_URL or not IBM_APIKEY:
        raise EnvironmentError("Set IBM_GRANITE_URL and IBM_APIKEY for Granite access.")
    headers = {"Authorization": f"Bearer {IBM_APIKEY}", "Content-Type":"application/json"}
    payload = {"model": model, "input": prompt, "max_tokens": max_tokens}
    resp = requests.post(IBM_GRANITE_URL, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # adjust based on IBM response format
    return data.get("output_text") or data.get("result") or data
