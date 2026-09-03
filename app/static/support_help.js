(function () {
  function nearestLabel(helpNode) {
    let previous = helpNode.previousElementSibling;
    while (previous) {
      if (previous.matches && previous.matches("label, h3, h2")) return previous;
      previous = previous.previousElementSibling;
    }
    const parent = helpNode.parentElement;
    if (!parent) return null;
    return parent.querySelector("label") || parent.querySelector("h3") || parent.querySelector("h2");
  }

  function bindHelp() {
    const explicitMode = document.body && document.body.dataset.helpMode === "explicit";
    const selector = explicitMode
      ? "[data-help-control]"
      : ".field-help, .hint, .option-help, .form-block > p, .action-guide";
    document.querySelectorAll(selector).forEach((helpNode) => {
      if (helpNode.dataset.helpBound === "true") return;
      const label = nearestLabel(helpNode);
      if (!label) return;
      helpNode.dataset.helpBound = "true";
      helpNode.classList.add("collapsed");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "help-dot";
      button.title = "Show option details";
      button.textContent = "?";
      button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        helpNode.classList.toggle("collapsed");
        helpNode.classList.toggle("help-open");
      });
      label.appendChild(button);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindHelp);
  } else {
    bindHelp();
  }
})();

