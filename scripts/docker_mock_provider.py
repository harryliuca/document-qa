"""Deterministic OpenAI-shaped test server; used only by the Docker smoke test."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/embeddings":
            result = {
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [
                    {"object": "embedding", "index": i, "embedding": [1.0, 0.0]}
                    for i, _ in enumerate(body["input"])
                ],
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            }
        else:
            messages = body["messages"]
            data = json.loads(messages[-1]["content"])
            schema = body["response_format"]["json_schema"]["name"]
            if schema == "ReviewOutput":
                content = {
                    "results": [
                        {"question_id": c["question_id"], "verdict": "supported"}
                        for c in data["candidates"]
                    ]
                }
            else:
                if schema == "BatchOutput":
                    source = json.loads(messages[1]["content"])["untrusted_document"][0]
                else:
                    source = data["sources"][0]
                answer = {
                    "status": "answered",
                    "answer": source["text"],
                    "evidence": [{"chunk_id": source["chunk_id"], "quote": source["text"]}],
                }
                content = (
                    {
                        "results": [
                            dict(answer, question_id=q["question_id"]) for q in data["questions"]
                        ]
                    }
                    if schema == "BatchOutput"
                    else answer
                )
            result = {
                "id": "docker-smoke",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(content)},
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            }
        payload = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 9000), Handler).serve_forever()
