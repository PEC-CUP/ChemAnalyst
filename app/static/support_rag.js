import { bindManual } from "./support_manual.js";

const pretty = value => JSON.stringify(value, null, 2);
const boolOrNull = value => value === "" ? null : value === "true";
const numberOrNull = value => value === "" ? null : Number(value);

async function json(url, payload) {
  const response = await fetch(url, payload ? {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  } : {});
  const data = await response.json();
  if (!response.ok) throw new Error(pretty(data));
  return data;
}

window.ragStatus = async function ragStatus() {
  try {
    ragOut.textContent = pretty(await json("/support-layer/rag/status"));
  } catch (error) {
    ragOut.textContent = String(error);
  }
};

window.ragTest = async function ragTest() {
  try {
    ragOut.textContent = "Running...";
    ragOut.textContent = pretty(await json("/support-layer/rag/test", {
      query: ragQuery.value,
      rag_mode: ragMode.value,
      enable_refinement_agent: boolOrNull(refinementAgent.value),
      enable_bm25: boolOrNull(bm25.value),
      enable_reranker: boolOrNull(reranker.value)
    }));
  } catch (error) {
    ragOut.textContent = String(error);
  }
};

window.ragCompare = async function ragCompare() {
  const modes = [
    {value: "naive", label: "Naive RAG"},
    {value: "qa_oriented", label: "QA-oriented RAG"},
    {value: "iterative_review", label: "Iterative review RAG"}
  ];
  const results = {};
  ragOut.textContent = "Running comparison: naive RAG -> QA-oriented RAG -> iterative review RAG";
  for (const mode of modes) {
    try {
      results[mode.label] = await json("/support-layer/rag/test", {
        query: ragQuery.value,
        rag_mode: mode.value,
        enable_refinement_agent: boolOrNull(refinementAgent.value),
        enable_bm25: boolOrNull(bm25.value),
        enable_reranker: boolOrNull(reranker.value)
      });
    } catch (error) {
      results[mode.label] = {status: "error", message: String(error)};
    }
    ragOut.textContent = pretty({
      status: "running",
      completed_modes: Object.keys(results),
      results
    });
  }
  ragOut.textContent = pretty({status: "completed", results});
};

window.ragPipeline = async function ragPipeline() {
  try {
    kbOut.textContent = "Running KB pipeline action...";
    kbOut.textContent = pretty(await json("/support-layer/rag/pipeline", {
      action: kbAction.value,
      label_mode: kbLabel.value,
      chunk_mode: kbChunk.value || null,
      chunk_token_size: numberOrNull(kbChunkSize.value),
      chunk_sentence_overlap: numberOrNull(kbOverlap.value),
      embedding_model: kbEmbedding.value || null,
      append: kbAppend.checked,
      embedding_local_files_only: kbLocal.checked
    }));
    await window.ragStatus();
  } catch (error) {
    kbOut.textContent = String(error);
  }
};

bindManual("ragManualBtn", "ragManual");
window.ragStatus();

