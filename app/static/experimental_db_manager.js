const pretty = value => JSON.stringify(value, null, 2);

let allSamples = [];
let sampleDetailMap = new Map();
const managerSearch = document.getElementById("managerSearch");
const managerLimit = document.getElementById("managerLimit");
const managerRefreshBtn = document.getElementById("managerRefreshBtn");
const managerSampleTypeFilter = document.getElementById("managerSampleTypeFilter");
const managerAnalysisTypeFilter = document.getElementById("managerAnalysisTypeFilter");
const managerArtifactFilter = document.getElementById("managerArtifactFilter");
const managerTableBody = document.getElementById("managerTableBody");
const managerSummary = document.getElementById("managerSummary");
const managerInfo = document.getElementById("managerInfo");
const managerAnalysisSummary = document.getElementById("managerAnalysisSummary");
const managerPropertiesBody = document.getElementById("managerPropertiesBody");
const managerAnalysesBody = document.getElementById("managerAnalysesBody");
const managerRelationsBody = document.getElementById("managerRelationsBody");
const managerRecord = document.getElementById("managerRecord");
const managerDeleteBtn = document.getElementById("managerDeleteBtn");
let selectedSampleId = null;

async function json(url) {
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(pretty(data));
  }
  return data;
}

function renderInfo(info) {
  const rows = Object.entries(info || {}).map(([key, value]) => {
    const row = document.createElement("div");
    row.className = "kv-row";
    row.innerHTML = `<b>${formatLabel(key)}</b><span>${formatValue(value)}</span>`;
    return row;
  });
  managerInfo.replaceChildren(...rows);
}

function formatLabel(key) {
  return String(key || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, ch => ch.toUpperCase());
}

function renderAnalysisSummary(sample) {
  const bulkProperties = sample.analysis_data?.bulk_properties || [];
  const analyses = sample.analysis_data?.analyses || [];
  const rows = [
    ["Bulk property rows", bulkProperties.length],
    ["Analysis records", analyses.length],
    ["Analysis types", [...new Set(analyses.map(item => item.analysis_type).filter(Boolean))].join(", ") || "-"],
    ["Artifacts attached", analyses.filter(hasArtifact).length],
    ["Last updated", sample.updated_at || "-"],
  ].map(([label, value]) => {
    const row = document.createElement("div");
    row.className = "kv-row";
    row.innerHTML = `<b>${label}</b><span>${formatValue(value)}</span>`;
    return row;
  });
  managerAnalysisSummary.replaceChildren(...rows);
}

function hasArtifact(item) {
  return Boolean(item?.artifact && Object.keys(item.artifact).length);
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") return "-";
  if (Array.isArray(value)) return value.length ? value.join(", ") : "-";
  if (typeof value === "object") return pretty(value);
  return String(value);
}

function renderProperties(sample) {
  const bulkProperties = sample.analysis_data?.bulk_properties || [];
  const rows = bulkProperties.map(item => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${item.property_name || "-"}</td>
      <td>${item.value || "-"}</td>
      <td>${item.unit || "-"}</td>
      <td>${item.method || "-"}</td>
    `;
    return tr;
  });
  if (!rows.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="4">No bulk-property rows recorded.</td>`;
    rows.push(tr);
  }
  managerPropertiesBody.replaceChildren(...rows);
}

function renderAnalyses(sample) {
  const analyses = sample.analysis_data?.analyses || [];
  const rows = analyses.map(item => {
    const artifact = item.artifact && Object.keys(item.artifact).length ? (item.artifact.filename || item.artifact.object_id || "yes") : "-";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${item.analysis_type || "-"}</td>
      <td>${item.method || "-"}</td>
      <td>${Array.isArray(item.rows) ? item.rows.length : 0}</td>
      <td>${artifact}</td>
    `;
    return tr;
  });
  if (!rows.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="4">No analysis records recorded.</td>`;
    rows.push(tr);
  }
  managerAnalysesBody.replaceChildren(...rows);
}

function renderRelations(sample) {
  const relations = sample.relations || [];
  const rows = relations.map(item => {
    const description = item.metadata?.description || item.description || "-";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${item.related_sample_id || "-"}</td>
      <td>${item.relation_type || "-"}</td>
      <td>${description}</td>
    `;
    return tr;
  });
  if (!rows.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="3">No relations recorded.</td>`;
    rows.push(tr);
  }
  managerRelationsBody.replaceChildren(...rows);
}

async function inspectSample(sampleId) {
  const sample = sampleDetailMap.get(sampleId) || (await json(`/support-layer/experimental-db/samples/${encodeURIComponent(sampleId)}`)).sample;
  selectedSampleId = sampleId;
  managerSummary.textContent = `Inspecting ${sampleId}`;
  renderInfo(sample.sample_information || {});
  renderAnalysisSummary(sample);
  renderProperties(sample);
  renderAnalyses(sample);
  renderRelations(sample);
  managerRecord.textContent = pretty(sample);
}

async function deleteSelectedSample() {
  if (!selectedSampleId) {
    managerSummary.textContent = "Select a sample before deleting.";
    return;
  }
  if (!window.confirm(`Delete sample ${selectedSampleId}? This removes the stored sample record and attached objects.`)) {
    return;
  }
  try {
    managerSummary.textContent = `Deleting ${selectedSampleId}...`;
    const response = await fetch(`/support-layer/experimental-db/samples/${encodeURIComponent(selectedSampleId)}`, {method: "DELETE"});
    const result = await response.json();
    if (!response.ok) throw new Error(pretty(result));
    selectedSampleId = null;
    await refreshSamples();
  } catch (error) {
    managerSummary.textContent = String(error);
  }
}

function filteredSamples() {
  const query = (managerSearch.value || "").trim().toLowerCase();
  const selectedSampleType = managerSampleTypeFilter.value;
  const selectedAnalysisType = managerAnalysisTypeFilter.value;
  const selectedArtifactFilter = managerArtifactFilter.value;
  return allSamples.filter(item => {
    const detail = sampleDetailMap.get(item.sample_id);
    const queryMatched = !query || [item.sample_id, item.sample_name, item.sample_type, item.origin]
      .filter(Boolean)
      .some(value => String(value).toLowerCase().includes(query));
    const sampleTypeMatched = !selectedSampleType || item.sample_type === selectedSampleType;
    const analysisTypes = (detail?.analysis_data?.analyses || []).map(entry => entry.analysis_type).filter(Boolean);
    const analysisTypeMatched = !selectedAnalysisType || analysisTypes.includes(selectedAnalysisType);
    const artifactCount = (detail?.analysis_data?.analyses || []).filter(hasArtifact).length;
    const artifactMatched =
      !selectedArtifactFilter ||
      (selectedArtifactFilter === "with_artifact" && artifactCount > 0) ||
      (selectedArtifactFilter === "without_artifact" && artifactCount === 0);
    return queryMatched && sampleTypeMatched && analysisTypeMatched && artifactMatched;
  });
}

function renderFilterOptions() {
  const sampleTypes = [...new Set(allSamples.map(item => item.sample_type).filter(Boolean))].sort();
  const analysisTypes = [...new Set(
    [...sampleDetailMap.values()].flatMap(sample =>
      (sample.analysis_data?.analyses || []).map(entry => entry.analysis_type).filter(Boolean)
    )
  )].sort();
  fillSelect(managerSampleTypeFilter, "All sample types", sampleTypes);
  fillSelect(managerAnalysisTypeFilter, "All analysis types", analysisTypes);
}

function fillSelect(select, placeholder, values) {
  const selectedValue = select.value;
  const options = [
    `<option value="">${placeholder}</option>`,
    ...values.map(value => `<option value="${escapeAttribute(value)}">${value}</option>`)
  ];
  select.innerHTML = options.join("");
  if (values.includes(selectedValue)) {
    select.value = selectedValue;
  }
}

function escapeAttribute(value) {
  return String(value).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
}

function renderTable() {
  const rows = filteredSamples().map(item => {
    const tr = document.createElement("tr");
    const updated = item.updated_at ? String(item.updated_at).replace("T", " ").replace("+00:00", " UTC") : "-";
    tr.innerHTML = `
      <td><button type="button" class="table-link" data-detail-sample-id="${item.sample_id}">${item.sample_id}</button></td>
      <td>${item.sample_name || "-"}</td>
      <td>${item.sample_type || "-"}</td>
      <td>${item.origin || "-"}</td>
      <td>${updated}</td>
      <td>${item.property_count || 0}</td>
      <td>${item.analysis_count || 0}</td>
      <td><a class="button-link table-action-link" href="/experimental-db-workbench?sample_id=${encodeURIComponent(item.sample_id)}#sample-update">Open update tools</a></td>
    `;
    return tr;
  });
  if (!rows.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="8">No samples match the current filters.</td>`;
    rows.push(tr);
  }
  managerTableBody.replaceChildren(...rows);
}

async function refreshSamples() {
  const limit = Math.max(1, Number(managerLimit.value) || 200);
  const result = await json(`/support-layer/experimental-db/samples?limit=${limit}`);
  allSamples = result.samples || [];
  const details = await Promise.all(
    allSamples.map(async item => {
      const sampleResult = await json(`/support-layer/experimental-db/samples/${encodeURIComponent(item.sample_id)}`);
      return [item.sample_id, sampleResult.sample];
    })
  );
  sampleDetailMap = new Map(details);
  renderFilterOptions();
  renderTable();
  const selectedSampleId = new URLSearchParams(window.location.search).get("sample_id");
  const firstVisibleSample = filteredSamples()[0];
  if (selectedSampleId && sampleDetailMap.has(selectedSampleId)) {
    await inspectSample(selectedSampleId);
  } else if (firstVisibleSample) {
    await inspectSample(firstVisibleSample.sample_id);
  }
}

managerRefreshBtn.addEventListener("click", refreshSamples);
managerSearch.addEventListener("input", renderTable);
managerSampleTypeFilter.addEventListener("change", renderTable);
managerAnalysisTypeFilter.addEventListener("change", renderTable);
managerArtifactFilter.addEventListener("change", renderTable);
managerDeleteBtn.addEventListener("click", deleteSelectedSample);
managerTableBody.addEventListener("click", event => {
  const button = event.target.closest("button[data-detail-sample-id]");
  if (!button) return;
  inspectSample(button.dataset.detailSampleId).catch(error => {
    managerSummary.textContent = String(error);
  });
});

refreshSamples().catch(error => {
  managerSummary.textContent = String(error);
  managerRecord.textContent = String(error);
});

