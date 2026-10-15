const THEME = {
  joy:      { color: "#f2b33d", ink: "#2b1f00", face: "😄" },
  sadness:  { color: "#4a6fa5", ink: "#ffffff", face: "😢" },
  anger:    { color: "#d64545", ink: "#ffffff", face: "😠" },
  fear:     { color: "#6b4fa0", ink: "#ffffff", face: "😨" },
  love:     { color: "#e07a9f", ink: "#2b0a17", face: "🥰" },
  surprise: { color: "#2ba89b", ink: "#03231f", face: "😲" }
};
const FALLBACK = { color: "#8795a8", ink: "#ffffff", face: "🙂" };
const theme = (emotion) => THEME[emotion.toLowerCase()] || FALLBACK;

const $ = (id) => document.getElementById(id);
const textEl = $("text");
const goBtn = $("go");
const errEl = $("error");

textEl.addEventListener("input", () => {
  $("count").textContent = `${textEl.value.length} / 1000`;
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    textEl.value = chip.textContent;
    textEl.dispatchEvent(new Event("input"));
    detect();
  });
});

goBtn.addEventListener("click", detect);
textEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) detect();
});

async function detect() {
  errEl.textContent = "";
  const text = textEl.value.trim();
  if (!text) {
    errEl.textContent = "Enter a sentence first.";
    return;
  }

  goBtn.disabled = true;
  goBtn.textContent = "Reading…";
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Something went wrong.");
    show(data);
  } catch (error) {
    errEl.textContent = error.message === "Failed to fetch"
      ? "Cannot reach the server. Start the app with run_app.py."
      : error.message;
  } finally {
    goBtn.disabled = false;
    goBtn.textContent = "Detect emotion";
  }
}

function show(data) {
  const selectedTheme = theme(data.emotion);
  const root = document.documentElement.style;
  root.setProperty("--tint", selectedTheme.color);
  root.setProperty("--tint-ink", selectedTheme.ink);
  document.body.classList.add("has-result");

  $("face").textContent = selectedTheme.face;
  $("name").textContent = data.emotion;
  $("conf").textContent = `${data.confidence.toFixed(1)}% confident`;

  const bars = $("bars");
  bars.innerHTML = "";
  data.scores.forEach((score) => {
    const item = document.createElement("li");
    item.innerHTML = '<span class="lbl"></span><div class="track"><div class="fill"></div></div><span class="pct"></span>';
    item.querySelector(".lbl").textContent = score.emotion;
    item.querySelector(".pct").textContent = `${score.score.toFixed(1)}%`;
    const fill = item.querySelector(".fill");
    fill.style.background = theme(score.emotion).color;
    bars.appendChild(item);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      fill.style.width = `${score.score}%`;
    }));
  });

  const result = $("result");
  result.classList.remove("show");
  void result.offsetWidth;
  result.classList.add("show");
}