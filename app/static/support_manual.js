export function bindManual(buttonId, modalId) {
  const button = document.getElementById(buttonId);
  const modal = document.getElementById(modalId);
  const close = modal && modal.querySelector("[data-manual-close]");
  if (!button || !modal || !close) return;
  const hide = () => {
    modal.classList.remove("visible");
    modal.setAttribute("aria-hidden", "true");
  };
  button.addEventListener("click", () => {
    modal.classList.add("visible");
    modal.setAttribute("aria-hidden", "false");
  });
  close.addEventListener("click", hide);
  modal.addEventListener("click", event => {
    if (event.target === modal) hide();
  });
}
