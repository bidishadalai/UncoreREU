import sys
import json
from pathlib import Path
from transformers import AutoTokenizer

path = Path(sys.argv[1])
jinja_file = path / "chat_template.jinja"

if jinja_file.exists():
    template = jinja_file.read_text()
else:
    src = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    template = src.chat_template

cfg_file = path / "tokenizer_config.json"
with open(cfg_file) as f:
    cfg = json.load(f)
cfg["chat_template"] = template

with open(cfg_file, "w") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)

t = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
if t.chat_template is None:
    print(f"FAILED to inject chat template into {path}")
    sys.exit(1)
print(f"Chat template injected and verified in {path}")
