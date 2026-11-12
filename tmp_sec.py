import subprocess, re
from pathlib import Path
# 工作区
pat = re.compile(r"(sk-[A-Za-z0-9]{16,}|Bearer\s+[A-Za-z0-9\-_]{20,}|AKIA[0-9A-Z]{16}|ms-[0-9a-f]{8}-)")
hits = []
for p in Path(".").rglob("*"):
    if ".git" in p.parts or not p.is_file(): continue
    if p.suffix.lower() in (".pyc",".png",".jpg",".mp4",".pdf",".zip",".json"): continue
    try: t = p.read_text(encoding="utf-8", errors="ignore")
    except Exception: continue
    for m in pat.finditer(t):
        hits.append((str(p), m.group(0)[:12]))
print("工作区疑似密钥:", hits[:10] or "无")
# 历史提交
log = subprocess.run(["git","log","-p","--all"], capture_output=True, text=True, encoding="utf-8", errors="ignore").stdout
found = set(re.findall(r"sk-[A-Za-z0-9]{16,}|ms-[0-9a-f]{8}-[0-9a-f]{4}", log))
print("历史里出现的密钥片段:", [x[:14] for x in list(found)[:5]] or "无")
# 本地绝对路径
paths = subprocess.run(["git","grep","-n","D:\\桌面","--",":"], capture_output=True, text=True, encoding="utf-8", errors="ignore")
print("提交文件里含本机绝对路径的行数:", len(paths.stdout.splitlines()))
for line in paths.stdout.splitlines()[:6]: print("   ", line[:110])
