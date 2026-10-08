const form = document.querySelector("#qa-form");
const submit = document.querySelector("#submit");
const progress = document.querySelector("#progress");
const errorBox = document.querySelector("#error");
const results = document.querySelector("#results");
const summary = document.querySelector("#summary");
const download = document.querySelector("#download");
const rawResult = document.querySelector("#raw-result");
const rawJson = document.querySelector("#raw-json");
let downloadUrl;
const strategy = document.querySelector("#strategy");
strategy.addEventListener("change", () => {
  const full = strategy.value === "full_context_batch";
  document.querySelector("#batch-options").hidden = !full;
  document.querySelector("#question-limit").textContent =
    `JSON · 1–${full ? 75 : 20} questions`;
});
function element(tag, text, className) {
  const el = document.createElement(tag);
  if (text !== undefined) el.textContent = text;
  if (className) el.className = className;
  return el;
}
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;
  summary.hidden = true;
  download.hidden = true;
  rawResult.hidden = true;
  rawJson.textContent = "";
  results.replaceChildren();
  if (downloadUrl) URL.revokeObjectURL(downloadUrl);
  downloadUrl = undefined;
  download.removeAttribute("href");
  const documentFile = form.elements.document.files[0];
  const questionFile = form.elements.questions.files[0];
  if (documentFile.size > 10 * 1024 * 1024 || questionFile.size > 64 * 1024) {
    errorBox.textContent = "Source limit: 10 MB. Questions limit: 64 KB.";
    errorBox.hidden = false;
    return;
  }
  const started = Date.now();
  const full = strategy.value === "full_context_batch";
  submit.disabled = true;
  const tick = () => {
    progress.textContent = `${full ? "Reading, batching, and reviewing evidence" : "Reading, retrieving, and checking quotes"}… ${Math.floor((Date.now() - started) / 1000)}s`;
  };
  tick();
  const timer = setInterval(tick, 1000);
  try {
    const data = new FormData();
    data.append("document", documentFile);
    data.append("questions", questionFile);
    const token = document.querySelector("#token").value.trim();
    const params = new URLSearchParams({
      strategy: strategy.value,
      cache_mode: document.querySelector("#cache-mode").value,
    });
    const response = await fetch(`/api/answers?${params}`, {
      method: "POST",
      body: data,
      signal: AbortSignal.timeout(full ? 330000 : 150000),
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    });
    const payload = await response.json();
    if (!response.ok)
      throw new Error(
        `${payload.error?.message || "Request failed."} (Reference: ${payload.request_id || "unavailable"})`,
      );
    downloadUrl = URL.createObjectURL(
      new Blob([JSON.stringify(payload, null, 2)], {
        type: "application/json",
      }),
    );
    download.href = downloadUrl;
    rawJson.textContent = JSON.stringify(payload, null, 2);
    rawResult.hidden = false;
    const answered = payload.results.filter(
      (r) => r.status === "answered",
    ).length;
    summary.textContent = `${answered}/${payload.results.length} answered · ${(payload.duration_ms / 1000).toFixed(1)}s · ${payload.document_chunks} source chunks`;
    const usage = payload.usage;
    summary.textContent += ` · ${usage.llm_calls} AI calls · ${usage.cached_input_tokens} cached input tokens · ${usage.retried_questions} questions retried`;
    if (!usage.usage_complete)
      summary.textContent += " · token usage incomplete";
    summary.hidden = false;
    for (const [i, answer] of payload.results.entries()) {
      const card = element("article", undefined, "answer");
      card.append(
        element(
          "span",
          answer.status.replace("_", " "),
          `badge ${answer.status}`,
        ),
      );
      card.append(element("h3", `${i + 1}. ${answer.question}`));
      card.append(element("p", answer.answer));
      if (answer.status === "answered")
        card.append(
          element(
            "p",
            answer.evidence_check === "model_checked"
              ? "Quotes matched and claim support reviewed by AI. Verify the evidence."
              : "Quotes matched to the source. Verify claim support.",
            "source",
          ),
        );
      if (answer.citations.length) {
        const details = element("details");
        details.append(
          element("summary", `View evidence (${answer.citations.length})`),
        );
        for (const citation of answer.citations) {
          details.append(
            element(
              "p",
              `${citation.location} · ${citation.chunk_id}`,
              "source",
            ),
          );
          details.append(element("blockquote", citation.quote));
        }
        card.append(details);
      }
      results.append(card);
    }
    download.hidden = false;
    progress.textContent =
      "Complete. Review the evidence or save the full JSON result.";
  } catch (error) {
    errorBox.textContent =
      error.name === "TimeoutError"
        ? "Request timed out. Try fewer questions."
        : error.message;
    errorBox.hidden = false;
    progress.textContent = "";
  } finally {
    clearInterval(timer);
    submit.disabled = false;
  }
});
