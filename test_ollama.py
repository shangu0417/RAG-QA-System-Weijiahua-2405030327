import requests

url = "http://localhost:11434/api/generate"
payload = {"model": "deepseek-r1:7b", "prompt": "你好", "stream": False}

try:
    response = requests.post(url, json=payload)
    print("Ollama连接成功！")
    print("回复:", response.json()["response"][:100])
except:
    print("连接失败，请确保Ollama已启动")