import { bindManual } from "./support_manual.js";

const pretty = value => JSON.stringify(value, null, 2);
const dbWorkbook = document.getElementById("dbWorkbook");
const dbWorkbookButton = document.getElementById("dbWorkbookButton");
const dbWorkbookName = document.getElementById("dbWorkbookName");
const dbWorkbookFolder = document.getElementById("dbWorkbookFolder");
const dbWorkbookFolderButton = document.getElementById("dbWorkbookFolderButton");
const dbArtifactFile = document.getElementById("dbArtifactFile");
const dbArtifactButton = document.getElementById("dbArtifactButton");
const dbArtifactName = document.getElementById("dbArtifactName");
const dbTemplateWorkbook = document.getElementById("dbTemplateWorkbook");
const dbTemplateWorkbookButton = document.getElementById("dbTemplateWorkbookButton");
const dbTemplateWorkbookName = document.getElementById("dbTemplateWorkbookName");
const dbEvidenceWorkbook = document.getElementById("dbEvidenceWorkbook");
const dbEvidenceWorkbookButton = document.getElementById("dbEvidenceWorkbookButton");
const dbEvidenceWorkbookName = document.getElementById("dbEvidenceWorkbookName");
const workbenchParams = new URLSearchParams(window.location.search);

function numericValue(value) {
  if (value === null || value === undefined || value === "") return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function propertyDigits(propertyName, isError = false) {
  const key = String(propertyName || "").toLowerCase();
  if (key.startsWith("bp_")) return 2;
  if (["saturates", "aromatics", "wax_content", "sat_ar"].includes(key)) return 2;
  return isError ? 3 : 2;
}

function formatPropertyValue(value, propertyName, isError = false) {
  const numeric = numericValue(value);
  if (numeric === null) return "-";
  if (String(propertyName || "").toLowerCase().includes("density")) return String(value);
  return numeric.toFixed(propertyDigits(propertyName, isError));
}

function formatPercent(value) {
  const numeric = numericValue(value);
  if (numeric === null) return "-";
  return numeric.toFixed(2);
}

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

window.dbStatus = async function dbStatus() {
  try {
    const data = await json("/support-layer/experimental-db/status");
    dbOut.textContent = pretty(data);
    const selectedSampleId = workbenchParams.get("sample_id");
    const fallbackSampleId = data.sample_id_examples && data.sample_id_examples[0];
    if (!dbInspectSample.value && (selectedSampleId || fallbackSampleId)) {
      dbInspectSample.value = selectedSampleId || fallbackSampleId;
    }
    if (selectedSampleId) {
      await dbInspect();
    }
  } catch (error) {
    dbOut.textContent = String(error);
  }
};

function bindFilePicker(input, button, label, emptyText) {
  if (!input || !button || !label) return;
  button.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (!input.files || input.files.length === 0) {
      label.textContent = emptyText;
    } else if (input.files.length === 1) {
      label.textContent = input.files[0].name;
    } else {
      label.textContent = `${input.files.length} files selected`;
    }
  });
}

window.dbTest = async function dbTest() {
  const file = dbEvidenceWorkbook && dbEvidenceWorkbook.files && dbEvidenceWorkbook.files[0];
  if (!file) {
    dbOut.textContent = "Select a query sample workbook first.";
    renderEvidencePresentation(null, dbOut.textContent);
    return;
  }
  if (!/\.(xlsx|xls)$/i.test(file.name)) {
    dbOut.textContent = "Query sample must be an Excel workbook.";
    renderEvidencePresentation(null, dbOut.textContent);
    return;
  }
  try {
    dbOut.textContent = "Building evidence...";
    const form = new FormData();
    form.append("file", file);
    form.append("query", dbQuery.value);
    form.append("top_k", String(Number(dbTopK.value)));
    form.append("use_llm_reasoning", String(Boolean(dbUseLlmReasoning && dbUseLlmReasoning.checked)));
    const response = await fetch("/support-layer/experimental-db/test-workbook", {method: "POST", body: form});
    const result = await response.json();
    if (!response.ok) throw new Error(pretty(result));
    dbOut.textContent = pretty(result);
    renderEvidencePresentation(result);
  } catch (error) {
    dbOut.textContent = String(error);
    renderEvidencePresentation(null, String(error));
  }
};

function importStatus(message, kind = "ok") {
  dbImportStatus.className = `status ${kind}`;
  dbImportStatus.textContent = message;
}

window.dbImportJson = async function dbImportJson() {
  try {
    importStatus("Importing sample JSON...");
    const payload = JSON.parse(dbSampleJson.value);
    const result = await json("/support-layer/experimental-db/samples/import", {payload});
    importStatus(`Imported sample ${result.sample_id}.`);
    dbSampleOut.textContent = pretty(result);
    dbInspectSample.value = result.sample_id;
    await dbStatus();
  } catch (error) {
    importStatus(String(error), "error");
  }
};

window.dbImportWorkbook = async function dbImportWorkbook() {
  const folderFiles = dbWorkbookFolder.files ? Array.from(dbWorkbookFolder.files) : [];
  const selectedFiles = dbWorkbook.files ? Array.from(dbWorkbook.files) : [];
  const files = folderFiles.length ? folderFiles : selectedFiles;
  if (files.length > 1 || folderFiles.length) {
    await importWorkbookFiles(files, folderFiles.length ? "selected folder" : "selected files");
    return;
  }
  await importSingleWorkbook(files[0]);
};

async function importSingleWorkbook(file) {
  if (!file) {
    importStatus("Select an Excel workbook first.", "error");
    return;
  }
  if (!/\.(xlsx|xls)$/i.test(file.name)) {
    importStatus("Selected file must be an Excel workbook.", "error");
    return;
  }
  try {
    importStatus("Importing workbook...");
    const form = new FormData();
    form.append("file", file);
    const response = await fetch("/support-layer/experimental-db/workbook/import", {method: "POST", body: form});
    const result = await response.json();
    if (!response.ok) throw new Error(pretty(result));
    importStatus(`Imported workbook sample ${result.sample_id}.`);
    dbSampleOut.textContent = pretty(result);
    dbInspectSample.value = result.sample_id;
    await dbStatus();
  } catch (error) {
    importStatus(String(error), "error");
  }
}

async function importWorkbookFiles(files, sourceLabel) {
  files = files.filter(file => /\.(xlsx|xls)$/i.test(file.name));
  if (!files.length) {
    importStatus(`No Excel workbooks found in ${sourceLabel}.`, "error");
    return;
  }
  try {
    importStatus(`Importing ${files.length} workbooks...`);
    const form = new FormData();
    for (const file of files) {
      form.append("files", file);
    }
    const response = await fetch("/support-layer/experimental-db/workbooks/import-batch", {method: "POST", body: form});
    const result = await response.json();
    if (!response.ok) throw new Error(pretty(result));
    const kind = result.failed_count ? "error" : "ok";
    importStatus(`Batch import finished: ${result.imported_count} imported, ${result.failed_count} failed.`, kind);
    dbSampleOut.textContent = pretty(result);
    const lastImported = result.imported && result.imported[result.imported.length - 1];
    if (lastImported && lastImported.sample_id) {
      dbInspectSample.value = lastImported.sample_id;
    }
    await dbStatus();
  } catch (error) {
    importStatus(String(error), "error");
  }
}

window.dbUpdateTemplate = async function dbUpdateTemplate() {
  const file = dbTemplateWorkbook.files && dbTemplateWorkbook.files[0];
  if (!file) {
    dbTemplateOut.textContent = "Select a template workbook first.";
    return;
  }
  try {
    dbTemplateOut.textContent = "Updating active template...";
    const form = new FormData();
    form.append("file", file);
    const response = await fetch("/support-layer/experimental-db/template/update", {method: "POST", body: form});
    const result = await response.json();
    if (!response.ok) throw new Error(pretty(result));
    dbTemplateOut.textContent = pretty(result);
    await dbStatus();
  } catch (error) {
    dbTemplateOut.textContent = String(error);
  }
};

window.dbInspect = async function dbInspect() {
  try {
    dbSampleOut.textContent = pretty(await json(`/support-layer/experimental-db/samples/${encodeURIComponent(dbInspectSample.value)}`));
  } catch (error) {
    dbSampleOut.textContent = String(error);
  }
};

window.dbAttachArtifact = async function dbAttachArtifact() {
  const file = dbArtifactFile.files && dbArtifactFile.files[0];
  if (!dbInspectSample.value || !dbArtifactType.value || !file) {
    dbSampleOut.textContent = "Sample id, analysis type, and artifact file are required.";
    return;
  }
  try {
    const form = new FormData();
    form.append("analysis_type", dbArtifactType.value);
    form.append("file", file);
    const response = await fetch(
      `/support-layer/experimental-db/samples/${encodeURIComponent(dbInspectSample.value)}/artifact`,
      {method: "POST", body: form}
    );
    const result = await response.json();
    if (!response.ok) throw new Error(pretty(result));
    dbSampleOut.textContent = pretty(result);
  } catch (error) {
    dbSampleOut.textContent = String(error);
  }
};

window.dbAddRelation = async function dbAddRelation() {
  if (!dbInspectSample.value || !dbRelatedSample.value || !dbRelationType.value) {
    dbSampleOut.textContent = "Sample id, related sample id, and relation type are required.";
    return;
  }
  try {
    const relation = {
      related_sample_id: dbRelatedSample.value,
      relation_type: dbRelationType.value
    };
    if (dbRelationDescription.value) relation.description = dbRelationDescription.value;
    const result = await json("/support-layer/experimental-db/samples/import", {
      payload: {
        sample: {sample_id: dbInspectSample.value},
        relations: [relation]
      }
    });
    dbSampleOut.textContent = pretty(result);
    await dbInspect();
  } catch (error) {
    dbSampleOut.textContent = String(error);
  }
};

bindManual("dbManualBtn", "dbManual");
bindFilePicker(dbWorkbook, dbWorkbookButton, dbWorkbookName, "No workbook selected");
bindFilePicker(dbWorkbookFolder, dbWorkbookFolderButton, dbWorkbookName, "No workbook selected");
bindFilePicker(dbArtifactFile, dbArtifactButton, dbArtifactName, "No file selected");
bindFilePicker(dbTemplateWorkbook, dbTemplateWorkbookButton, dbTemplateWorkbookName, "No file selected");
bindFilePicker(dbEvidenceWorkbook, dbEvidenceWorkbookButton, dbEvidenceWorkbookName, "No query workbook selected");

function renderEvidencePresentation(result, errorText = "") {
  if (!dbEvidenceView) return;
  if (errorText) {
    dbEvidenceView.innerHTML = `<div class="evidence-empty">${errorText}</div>`;
    return;
  }
  if (!result || result.status !== "success" || !result.presentation) {
    dbEvidenceView.innerHTML = `<div class="evidence-empty">Run an Experimental Database evidence test to render a manuscript-ready evidence view.</div>`;
    return;
  }
  const presentation = result.presentation;
  const workflow = (presentation.workflow || []).map(
    item => `<div class="evidence-step"><b>Step ${item.step}. ${item.title}</b><span>${item.detail}</span></div>`
  ).join("");
  const querySnapshot = presentation.query_snapshot || {};
  const neighborRows = (presentation.neighbor_table || []).map(
    item => `
      <tr>
        <td>${item.rank}</td>
        <td>${item.sample_id || "-"}</td>
        <td>${item.sample_type || "-"}</td>
        <td>${item.origin || "-"}</td>
        <td>${item.distance || "-"}</td>
        <td>${item.shared_feature_count || "-"}</td>
        <td>${(item.analysis_channels || []).join(", ") || "-"}</td>
      </tr>
    `
  ).join("") || `<tr><td colspan="7">No neighbors returned.</td></tr>`;
  const hasValidationColumns = (presentation.property_table || []).some(
    item => item.actual_for_validation !== undefined || item.absolute_error !== undefined || item.absolute_percentage_error !== undefined
  );
  const validationHeaders = hasValidationColumns
    ? "<th>Validation actual</th><th>Abs error</th><th>Abs. rel. error/%</th>"
    : "";
  const propertyRows = (presentation.property_table || []).map(
    item => `
      <tr>
        <td>${item.property_name || "-"}</td>
        <td>${formatPropertyValue(item.weighted_estimate, item.property_name)}</td>
        <td>${formatPropertyValue(item.range_min, item.property_name)} to ${formatPropertyValue(item.range_max, item.property_name)}</td>
        <td>${item.neighbor_count || "-"}</td>
        <td>${(item.supporting_samples || []).join(", ") || "-"}</td>
        ${hasValidationColumns ? `<td>${formatPropertyValue(item.actual_for_validation, item.property_name)}</td><td>${formatPropertyValue(item.absolute_error, item.property_name, true)}</td><td>${formatPercent(item.absolute_percentage_error)}</td>` : ""}
      </tr>
    `
  ).join("") || `<tr><td colspan="${hasValidationColumns ? 8 : 5}">No property evidence summarized.</td></tr>`;
  const notes = (presentation.reasoning_notes || []).map(item => `<li>${item}</li>`).join("");
  const constrained = presentation.constrained_reasoning || {};
  const diagnostics = constrained.diagnostics || {};
  const expertReasoning = constrained.expert_reasoning || {};
  const promptPolicy = (constrained.prompt_policy || []).map(item => `<li>${item}</li>`).join("");
  const propertyClaims = (constrained.property_claims || []).map(
    item => `<li><b>${item.property_name || "-"}</b>: ${item.interpretation || "-"}</li>`
  ).join("");
  const supportCounts = Object.entries(diagnostics.property_support_counts || {}).map(
    ([name, count]) => `${name}: ${count}`
  ).join(", ");
  dbEvidenceView.innerHTML = `
    <div class="detail-grid">
      <section class="detail-card">
        <h3>Evidence Workflow</h3>
        <div class="evidence-steps">${workflow}</div>
      </section>
      <section class="detail-card">
        <h3>Query Sample Snapshot</h3>
        <div class="kv-list">
          <div class="kv-row"><b>Sample ID</b><span>${querySnapshot.sample_id || "-"}</span></div>
          <div class="kv-row"><b>Feature Count</b><span>${querySnapshot.feature_count || "-"}</span></div>
          <div class="kv-row"><b>Evidence Channels</b><span>${(querySnapshot.channels || []).join(", ") || "-"}</span></div>
          <div class="kv-row"><b>Feature Examples</b><span>${(querySnapshot.feature_examples || []).join(", ") || "-"}</span></div>
        </div>
      </section>
    </div>
    <section class="detail-card" style="margin-top:18px;">
      <h3>Historical Neighbor Evidence</h3>
      <div class="table-shell compact-shell">
        <table class="data-table compact-table">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Sample ID</th>
              <th>Type</th>
              <th>Origin</th>
              <th>Distance</th>
              <th>Shared features</th>
              <th>Channels</th>
            </tr>
          </thead>
          <tbody>${neighborRows}</tbody>
        </table>
      </div>
    </section>
    <section class="detail-card" style="margin-top:18px;">
      <h3>Property Evidence Summary</h3>
      <div class="table-shell compact-shell">
        <table class="data-table compact-table">
          <thead>
            <tr>
              <th>Property</th>
              <th>Weighted estimate</th>
              <th>Historical range</th>
              <th>Support count</th>
              <th>Supporting samples</th>
              ${validationHeaders}
            </tr>
          </thead>
          <tbody>${propertyRows}</tbody>
        </table>
      </div>
    </section>
    <section class="detail-card" style="margin-top:18px;">
      <h3>Evidence-constrained Final Inference</h3>
      <div class="kv-list">
        <div class="kv-row"><b>Reasoning Stage</b><span>${constrained.status || "-"}</span></div>
        <div class="kv-row"><b>Neighbor Count</b><span>${diagnostics.neighbor_count || constrained.neighbor_count || "-"}</span></div>
        <div class="kv-row"><b>Distance Range</b><span>${diagnostics.distance_min || "-"} to ${diagnostics.distance_max || "-"}</span></div>
        <div class="kv-row"><b>Shared Feature Range</b><span>${diagnostics.shared_feature_count_min || "-"} to ${diagnostics.shared_feature_count_max || "-"}</span></div>
        <div class="kv-row"><b>Property Support Counts</b><span>${supportCounts || "-"}</span></div>
      </div>
      <p class="evidence-statement">${constrained.final_answer || "-"}</p>
      <h4>LLM analytical reasoning</h4>
      <div class="kv-list">
        <div class="kv-row"><b>LLM Provider</b><span>${expertReasoning.provider || "-"}</span></div>
        <div class="kv-row"><b>LLM Model</b><span>${expertReasoning.model || "-"}</span></div>
      </div>
      <div class="expert-answer">${expertReasoning.answer || expertReasoning.error || "LLM reasoning was not requested or did not return an answer."}</div>
      <h4>Property-level interpretation</h4>
      <ul class="evidence-notes">${propertyClaims || "<li>No property-level claims generated.</li>"}</ul>
      <h4>Prompt constraints used downstream</h4>
      <ul class="evidence-notes">${promptPolicy || "<li>No prompt policy returned.</li>"}</ul>
    </section>
    <section class="detail-card" style="margin-top:18px;">
      <h3>Evidence Statement</h3>
      <p class="evidence-statement">${presentation.evidence_statement || "-"}</p>
      <ul class="evidence-notes">${notes}</ul>
    </section>
  `;
}

renderEvidencePresentation(null);
window.dbStatus();
