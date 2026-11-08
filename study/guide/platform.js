const catalogUrl = new URL("../catalog.json", location.href);

function showStatus(message) {
  const status = document.getElementById("status");
  status.hidden = false;
  status.textContent = message;
}

async function readJson(url) {
  const response = await fetch(url, { cache: "no-cache" });
  if (!response.ok) {
    throw new Error(`${response.status} ${url.pathname}`);
  }
  return response.json();
}

function fillPicker(catalog, currentId) {
  const select = document.getElementById("guide-picker");
  select.replaceChildren();
  for (const guide of catalog.guides) {
    const option = document.createElement("option");
    option.value = guide.id;
    option.textContent = guide.title;
    if (guide.id === currentId) option.selected = true;
    select.appendChild(option);
  }
  select.hidden = catalog.guides.length < 2;
  select.closest("label").hidden = select.hidden;
}

async function renderGuide(entry) {
  const manifestUrl = new URL(`../${entry.manifest}`, location.href);
  const manifest = await readJson(manifestUrl);
  document.getElementById("guide-title").textContent = manifest.title || entry.title;
  document.getElementById("guide-summary").textContent = entry.summary || "";
  document.title = `${manifest.title} — Deep Generative Models`;

  const pdf = document.getElementById("pdf-link");
  pdf.href = new URL(`../${entry.pdf}`, location.href).href;

  const toc = document.getElementById("toc");
  toc.replaceChildren();
  const article = document.getElementById("guide");
  article.replaceChildren();

  for (const section of manifest.sections) {
    const link = document.createElement("a");
    link.href = `#${section.id}`;
    link.textContent = section.title;
    toc.appendChild(link);

    const response = await fetch(new URL(section.file, manifestUrl), { cache: "no-cache" });
    if (!response.ok) {
      throw new Error(`${response.status} ${section.file}`);
    }
    const block = document.createElement("div");
    block.innerHTML = await response.text();
    article.appendChild(block.firstElementChild);
  }

  if (window.renderMathInElement) {
    window.renderMathInElement(article, {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
    });
  }
}

async function main() {
  const select = document.getElementById("guide-picker");
  select.addEventListener("change", () => {
    const url = new URL(location.href);
    url.searchParams.set("guide", select.value);
    location.assign(url);
  });

  try {
    const catalog = await readJson(catalogUrl);
    if (!catalog.guides || catalog.guides.length === 0) {
      showStatus("The catalog has no guides yet.");
      return;
    }
    const requested = new URLSearchParams(location.search).get("guide");
    const entry = catalog.guides.find((guide) => guide.id === requested) || catalog.guides[0];
    fillPicker(catalog, entry.id);
    await renderGuide(entry);
  } catch (error) {
    showStatus(
      `The guide sections could not be loaded (${error.message}). Open this page over http or on GitHub Pages. A file:// address cannot fetch the section files.`
    );
  }
}

main();
