import { bindManual } from "./support_manual.js";

const pretty = value => JSON.stringify(value, null, 2);
let currentBenchmarkJobId = null;
let benchmarkPollTimer = null;
let currentKbJobId = null;
let kbPollTimer = null;

async function parseJsonResponse(response, options = {}) {
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = {
      status: "error",
      message: text || response.statusText || "Non-JSON server response"
    };
  }
  if (!response.ok || (!options.allowApplicationError && data.status === "error")) {
    throw new Error(pretty(data));
  }
  return data;
}

async function jsonRequest(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  return parseJsonResponse(response);
}

function show(id, text) {
  const node = document.getElementById(id);
  node.textContent = text;
  node.style.display = "block";
}

window.loadOptions = async function loadOptions() {
  const data = await (await fetch("/benchmark-workbench/options")).json();
  datasets.textContent = data.benchmark_datasets.length
    ? "Existing JSONL benchmarks:\n" + data.benchmark_datasets
        .map(item => `- ${item.name} (${Math.round((item.size || 0) / 1024)} KB)`)
        .join("\n")
    : "No JSONL benchmark dataset listed yet.";
};

window.loadKbStatus = async function loadKbStatus() {
  try {
    const data = await (await fetch("/benchmark-workbench/kb/status")).json();
    kbOutput.textContent = pretty(data);
    const cleanCount = data?.active_kb?.clean_chunk_count;
    const graphStatus = data?.active_kb?.keyword_graph_exists ? "found" : "not found";
    const buildCount = Array.isArray(data?.staged_builds) ? data.staged_builds.length : 0;
    show("kbStatus", `Active clean chunks: ${cleanCount || "not found"}; active keyword graph: ${graphStatus}; staged builds listed: ${buildCount}.`);
  } catch (error) {
    kbOutput.textContent = String(error);
    show("kbStatus", "KB status request failed.");
  }
};

window.chooseKbDirectory = function chooseKbDirectory() {
  kbDirPicker.click();
};

window.usePickedKbDirectory = function usePickedKbDirectory(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  const relative = file.webkitRelativePath || "";
  const folder = relative.includes("/") ? relative.split("/")[0] : "";
  if (file.path) {
    kbSourceDir.value = file.path;
    show("kbStatus", "Selected an absolute PDF folder path.");
    return;
  }
  if (!kbSourceDir.value && folder) {
    kbSourceDir.value = folder;
  }
  show("kbStatus", "Browser folder picker only provided a folder name. Paste the absolute folder path before building.");
};

window.startKbConstruction = async function startKbConstruction() {
  show("kbStatus", "Starting staged KB construction job.");
  try {
    const data = await jsonRequest("/benchmark-workbench/kb/start", {
      source_dir: kbSourceDir.value,
      build_name: kbBuildName.value || "kb.paper.staged",
      pdf_parser_backend: kbPdfParserBackend.value,
      chunk_mode: kbChunkMode.value,
      chunk_token_size: Number(kbChunkTokenSize.value),
      chunk_sentence_overlap: Number(kbSentenceOverlap.value),
      label_mode: "none",
      build_index: kbBuildIndex.checked,
      build_keyword_graph: kbBuildKeywordGraph.checked,
      embedding_model: kbEmbeddingModel.value || null,
      embedding_local_files_only: kbEmbeddingLocal.checked,
      allow_hash_fallback: kbAllowHashFallback.checked
    });
    currentKbJobId = data.job_id;
    kbBuildBtn.disabled = true;
    kbOutput.textContent = pretty(data);
    show("kbStatus", `Staged KB job started: ${currentKbJobId}`);
    pollKbConstructionJob();
  } catch (error) {
    kbOutput.textContent = String(error);
    show("kbStatus", "Staged KB construction request failed.");
  }
};

window.startKbConstructionWithGraph = function startKbConstructionWithGraph() {
  kbBuildKeywordGraph.checked = true;
  return window.startKbConstruction();
};

window.openActiveKeywordGraph = function openActiveKeywordGraph() {
  window.open("/benchmark-workbench/kb/active-keyword-graph", "_blank", "noopener");
};

async function pollKbConstructionJob() {
  if (!currentKbJobId) return;
  try {
    const response = await fetch(`/benchmark-workbench/kb/jobs/${currentKbJobId}`);
    const data = await parseJsonResponse(response, {allowApplicationError: true});
    kbOutput.textContent = pretty(data);
    show("kbStatus", data.message || data.status || "Staged KB job running.");
    if (["success", "error"].includes(data.status)) {
      kbBuildBtn.disabled = false;
      currentKbJobId = null;
      if (kbPollTimer) clearTimeout(kbPollTimer);
      kbPollTimer = null;
      await window.loadKbStatus();
      return;
    }
    kbPollTimer = setTimeout(pollKbConstructionJob, 2000);
  } catch (error) {
    kbOutput.textContent = String(error);
    show("kbStatus", "Staged KB job polling failed.");
    kbBuildBtn.disabled = false;
    currentKbJobId = null;
  }
}

window.applyBenchmarkPreset = function applyBenchmarkPreset(preset) {
  const cleanChunks = "petroleum_kb/petroleum_kb/data/processed/chunks_clean.jsonl";
  const presets = {
    reviewed_single: {
      benchmarkName: "benchmark.reviewed.single.200.en.final",
      benchmarkMode: "reviewed",
      benchmarkStyle: "semantic",
      answerFormat: "open",
      questionCount: 200,
      ratio1: 1,
      ratio2: 0,
      quality: 0.7
    },
    reviewed_twodoc: {
      benchmarkName: "benchmark.reviewed.twodoc.200.en.final",
      benchmarkMode: "reviewed",
      benchmarkStyle: "semantic",
      answerFormat: "open",
      questionCount: 200,
      ratio1: 0,
      ratio2: 1,
      quality: 0.7
    },
    specific_open: {
      benchmarkName: "benchmark.reviewed.specific_fact.single.open.200.en",
      benchmarkMode: "reviewed",
      benchmarkStyle: "specific_fact",
      answerFormat: "open",
      questionCount: 200,
      ratio1: 1,
      ratio2: 0,
      quality: 0.8
    },
    specific_mcq: {
      benchmarkName: "benchmark.reviewed.specific_fact.single.mcq.100.en",
      benchmarkMode: "reviewed",
      benchmarkStyle: "specific_fact",
      answerFormat: "mcq",
      questionCount: 100,
      ratio1: 1,
      ratio2: 0,
      quality: 0.8
    },
    naive_single: {
      benchmarkName: "benchmark.naive_baseline.single.200.en.final",
      benchmarkMode: "naive_baseline",
      benchmarkStyle: "semantic",
      answerFormat: "open",
      questionCount: 200,
      ratio1: 1,
      ratio2: 0,
      quality: 0.7
    },
    naive_twodoc: {
      benchmarkName: "benchmark.naive_baseline.twodoc.200.en.final",
      benchmarkMode: "naive_baseline",
      benchmarkStyle: "semantic",
      answerFormat: "open",
      questionCount: 200,
      ratio1: 0,
      ratio2: 1,
      quality: 0.7
    }
  };
  const cfg = presets[preset];
  if (!cfg) return;
  benchmarkName.value = cfg.benchmarkName;
  benchmarkMode.value = cfg.benchmarkMode;
  benchmarkStyle.value = cfg.benchmarkStyle;
  answerFormat.value = cfg.answerFormat;
  questionCount.value = cfg.questionCount;
  ratio1.value = cfg.ratio1.toFixed(2);
  ratio2.value = cfg.ratio2.toFixed(2);
  quality.value = cfg.quality;
  language.value = "en";
  writeMode.value = "new";
  sourceMode.value = "existing_chunks";
  chunksPath.value = cleanChunks;
  window.syncSourceMode();
  show("generationStatus", `Preset loaded: ${cfg.benchmarkName}`);
};

window.generateBenchmark = async function generateBenchmark() {
  show("generationStatus", "Starting benchmark generation job.");
  try {
    const data = await jsonRequest("/benchmark-workbench/benchmark/start", {
      benchmark_name: benchmarkName.value,
      write_mode: writeMode.value,
      benchmark_mode: benchmarkMode.value,
      source_mode: sourceMode.value,
      chunks_path: chunksPath.value || null,
      pdf_dir: pdfDir.value || null,
      pdf_parser_backend: pdfParserBackend.value,
      chunk_mode: chunkMode.value,
      benchmark_style: benchmarkStyle.value,
      answer_format: answerFormat.value,
      question_count: Number(questionCount.value),
      language: language.value,
      quality_min_score: Number(quality.value),
      single_doc_ratio: Number(ratio1.value),
      two_doc_ratio: Number(ratio2.value)
    });
    currentBenchmarkJobId = data.job_id;
    generateBtn.disabled = true;
    stopBtn.disabled = false;
    benchmarkOutput.textContent = pretty(data);
    show("generationStatus", `Benchmark job started: ${currentBenchmarkJobId}`);
    pollBenchmarkJob();
  } catch (error) {
    benchmarkOutput.textContent = String(error);
    show("generationStatus", "Benchmark generation request failed.");
  }
};

async function pollBenchmarkJob() {
  if (!currentBenchmarkJobId) return;
  try {
    const response = await fetch(`/benchmark-workbench/benchmark/jobs/${currentBenchmarkJobId}`);
    const data = await parseJsonResponse(response, {allowApplicationError: true});
    benchmarkOutput.textContent = pretty(data);
    show("generationStatus", data.message || data.status || "Benchmark job running.");
    if (["success", "error", "stopped"].includes(data.status)) {
      generateBtn.disabled = false;
      stopBtn.disabled = true;
      currentBenchmarkJobId = null;
      if (benchmarkPollTimer) clearTimeout(benchmarkPollTimer);
      benchmarkPollTimer = null;
      await window.loadOptions();
      return;
    }
    benchmarkPollTimer = setTimeout(pollBenchmarkJob, 1500);
  } catch (error) {
    benchmarkOutput.textContent = String(error);
    show("generationStatus", "Benchmark job polling failed.");
    generateBtn.disabled = false;
    stopBtn.disabled = true;
    currentBenchmarkJobId = null;
  }
}

window.stopBenchmark = async function stopBenchmark() {
  if (!currentBenchmarkJobId) return;
  try {
    const data = await jsonRequest(`/benchmark-workbench/benchmark/jobs/${currentBenchmarkJobId}/stop`, {});
    benchmarkOutput.textContent = pretty(data);
    show("generationStatus", "Stop requested for benchmark generation.");
  } catch (error) {
    benchmarkOutput.textContent = String(error);
    show("generationStatus", "Stop request failed.");
  }
};

window.chooseChunksFile = function chooseChunksFile() {
  chunksFilePicker.click();
};

window.usePickedChunksFile = function usePickedChunksFile(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  chunksPath.value = file.path || file.webkitRelativePath || file.name;
  show("generationStatus", "Selected chunks file name. If the backend cannot access it, paste the absolute path manually.");
};

window.choosePdfDirectory = function choosePdfDirectory() {
  pdfDirPicker.click();
};

window.usePickedPdfDirectory = function usePickedPdfDirectory(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  const relative = file.webkitRelativePath || "";
  const folder = relative.includes("/") ? relative.split("/")[0] : "";
  if (file.path) {
    pdfDir.value = file.path;
    show("generationStatus", "Selected an absolute PDF folder path.");
    return;
  }
  if (!pdfDir.value && folder) {
    pdfDir.value = folder;
  }
  show("generationStatus", "Browser folder picker only provided a folder name. Paste the absolute folder path, e.g. outputs/benchmark_generation, before generation.");
};

window.syncSourceMode = function syncSourceMode() {
  const pdfMode = sourceMode.value === "pdf_directory";
  chunksPath.disabled = pdfMode;
  chunksChooseBtn.disabled = pdfMode;
  if (pdfMode) {
    chunksPath.value = "";
    show("generationStatus", "PDF directory mode selected. Existing chunks JSONL path will be ignored.");
  }
};

bindManual("manualBtn", "manualModal");
window.loadKbStatus();
window.loadOptions();
window.syncSourceMode();
