from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def extract_text(content):
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            parts.append(block.get("text", ""))
    return "".join(parts)

candidates = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

for m in candidates:
    try:
        llm = ChatGoogleGenerativeAI(model=m)
        resp = llm.invoke("Reply with exactly: OK")
        print(f"✅ {m} -> {extract_text(resp.content).strip()!r}")
    except Exception as e:
        print(f"❌ {m} -> {type(e).__name__}: {str(e)[:120]}")