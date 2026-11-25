// pptxgenjs 3.12 writes a <a:pPr> for every run, which drops the bullet on items with bold or code runs.
// Patch node_modules/pptxgenjs/dist/pptxgen.cjs.js to emit pPr only for the first run (idx === 0).
// Usage: node build_pptx.js day2_slides.md day2_slides.pptx  (images resolve relative to the .md)
// Build an editable .pptx (native text, tables, speaker notes) from the Marp deck.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const SRC = process.argv[2] || "day2_slides.md";
const SRCDIR = path.dirname(path.resolve(SRC));
function pngSize(p) { const b = fs.readFileSync(p); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
const OUT = process.argv[3] || "day2_slides.pptx";

const COL = {
  ink: "0B2A3A",      // deep navy-teal: titles, dark slides, table headers
  body: "243B47",
  muted: "6B7C86",
  accent: "D55E00",   // vermillion, as in the notebooks' palette
  blue: "0072B2",
  tint: "EEF4F7",
  zebra: "F4F7F9",
  rule: "D3DCE1",
  white: "FFFFFF",
  green: "009E73",
  greenTint: "F2F8F5",
};
let TEXT = null, HEADC = null;  // per-slide body and heading colours (dark slides)
const HEAD = "Calibri";
const BODY = "Calibri";
const MONO = "Courier New";

const W = 13.333, H = 7.5, MX = 0.6, CW = W - 2 * MX;
const TOP = 1.45, BOTTOM = 6.85;

// ---------------------------------------------------------------- parsing
function parseSlides(md) {
  const parts = md.split(/\n---\n/);
  parts.shift(); // front matter body (after first ---)
  const slides = [];
  // front matter: the first "---" is the file start, so parts[0] was the YAML; style block ended at next ---
  for (const raw of parts) {
    let cls = "";
    const m = raw.match(/<!--\s*_class:\s*(\w+)\s*-->/);
    if (m) cls = m[1];
    const notes = [];
    let body = raw.replace(/<!--([\s\S]*?)-->/g, (all, inner) => {
      if (/^\s*_/.test(inner)) return "";
      notes.push(inner.trim());
      return "";
    });
    slides.push({ cls, notes: notes.join("\n\n"), blocks: parseBlocks(body) });
  }
  return slides;
}

function parseBlocks(text) {
  const lines = text.split("\n");
  const blocks = [];
  let i = 0;
  while (i < lines.length) {
    let line = lines[i];
    if (!line.trim()) { i++; continue; }
    if (line.trim() === '<div class="cols">') {
      // two <div> children
      const cols = [];
      i++;
      while (i < lines.length && lines[i].trim() !== "</div>" || (i < lines.length && cols.length < 2)) {
        if (lines[i].trim() === "<div>") {
          const buf = [];
          i++;
          while (lines[i].trim() !== "</div>") { buf.push(lines[i]); i++; }
          cols.push(parseBlocks(buf.join("\n")));
        }
        i++;
        if (cols.length === 2) { while (i < lines.length && lines[i].trim() !== "</div>") i++; i++; break; }
      }
      blocks.push({ type: "cols", cols });
      continue;
    }
    if (line.startsWith("```")) {
      const buf = [];
      i++;
      while (!lines[i].startsWith("```")) { buf.push(lines[i]); i++; }
      i++;
      blocks.push({ type: "code", text: buf.join("\n") });
      continue;
    }
    if (line.startsWith("> ")) {
      const buf = [];
      while (i < lines.length && lines[i].startsWith(">")) { buf.push(lines[i].replace(/^>\s?/, "")); i++; }
      blocks.push({ type: "takeaway", text: buf.join(" ").trim() });
      continue;
    }
    const im = line.trim().match(/^!\[[^\]]*\]\(([^)\s]+)(?:\s+"video:([^"]+)")?\)$/);
    if (im) {
      const file = path.join(SRCDIR, im[1]);
      const { w, h } = pngSize(file);
      // ![..](poster.png "video:clip.mp4") embeds the clip, with the poster as its cover
      blocks.push({ type: "img", file, aspect: w / h, video: im[2] ? path.join(SRCDIR, im[2]) : null });
      i++;
      continue;
    }
    if (line.startsWith("# ")) { blocks.push({ type: "h1", text: line.slice(2).trim() }); i++; continue; }
    if (line.startsWith("## ")) { blocks.push({ type: "h2", text: line.slice(3).trim() }); i++; continue; }
    if (line.startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].startsWith("|")) {
        const cells = lines[i].trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(s => s.trim());
        if (!cells.every(c => /^:?-+:?$/.test(c))) rows.push(cells);
        i++;
      }
      blocks.push({ type: "table", rows });
      continue;
    }
    if (/^\s*[-*] /.test(line) || /^\s*\d+\. /.test(line)) {
      const ordered = /^\s*\d+\. /.test(line);
      const items = [];
      while (i < lines.length && (/^\s*[-*] /.test(lines[i]) || /^\s*\d+\. /.test(lines[i]) || (/^\s{2,}\S/.test(lines[i]) && items.length))) {
        const l = lines[i];
        if (/^\s*[-*] /.test(l) || /^\s*\d+\. /.test(l)) items.push(l.replace(/^\s*([-*]|\d+\.) /, "").trim());
        else items[items.length - 1] += " " + l.trim();
        i++;
      }
      blocks.push({ type: ordered ? "ol" : "ul", items });
      continue;
    }
    // paragraph: join until blank line or another block
    const buf = [];
    while (i < lines.length && lines[i].trim() && !/^(#|\||```|\s*[-*] |\s*\d+\. |<div|!\[|> )/.test(lines[i])) {
      buf.push(lines[i].trim());
      i++;
    }
    blocks.push({ type: "p", lines: buf });
  }
  return blocks;
}

function cleanMath(s) {
  return s.replace(/\$\\tilde\{y\} = S\\,G\\,\\hat\{y\}\$/, "ỹ = S·G·ŷ");
}

// inline markdown -> pptxgenjs runs
function runs(text, base) {
  text = cleanMath(text);
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push({ text: text.slice(last, m.index), options: { ...base } });
    const t = m[0];
    if (t.startsWith("**")) out.push({ text: t.slice(2, -2), options: { ...base, bold: true } });
    else if (t.startsWith("`")) out.push({ text: t.slice(1, -1), options: { ...base, fontFace: MONO, color: COL.blue } });
    else out.push({ text: t.slice(1, -1), options: { ...base, italic: true } });
    last = m.index + t.length;
  }
  if (last < text.length) out.push({ text: text.slice(last), options: { ...base } });
  return out;
}
const plain = s => cleanMath(s).replace(/\*\*|`|\*/g, "");

// ---------------------------------------------------------------- measuring
const lineH = pt => pt * 1.25 / 72;
function nLines(text, widthIn, pt, mono = false) {
  const charW = (mono ? 0.6 : 0.5) * pt / 72;
  const per = Math.max(4, Math.floor((widthIn - 0.2) / charW));
  return text.split("\n").reduce((n, l) => n + Math.max(1, Math.ceil(plain(l).length / per)), 0);
}

function colWidths(rows, width, pt) {
  const n = rows[0].length;
  const lens = Array.from({ length: n }, (_, c) => Math.max(...rows.map(r => plain(r[c] || "").length), 3));
  // cap very long columns so short ones keep room
  const capped = lens.map(l => Math.min(l, 60));
  const tot = capped.reduce((a, b) => a + b, 0);
  let ws = capped.map(l => width * l / tot);
  const minW = 0.9;
  ws = ws.map(w => Math.max(w, minW));
  const s = ws.reduce((a, b) => a + b, 0);
  return ws.map(w => w * width / s);
}

function tableHeight(rows, width, pt) {
  const ws = colWidths(rows, width, pt);
  return rows.reduce((h, r) => h + Math.max(...r.map((c, j) => nLines(c, ws[j], pt))) * lineH(pt) + 0.14, 0);
}

function blockHeight(b, width, pt) {
  switch (b.type) {
    case "h2": return lineH(pt + 2) * nLines(b.text, width, pt + 2) + 0.05;
    case "p": return b.lines.reduce((h, l) => h + nLines(l, width, pt) * lineH(pt), 0) + 0.05;
    case "ul": case "ol": return b.items.reduce((h, it) => h + nLines(it, width - 0.4, pt) * lineH(pt) + pt * 0.35 / 72, 0) + 0.05;
    case "code": return nLines(b.text, width, pt - 3, true) * lineH(pt - 3) + 0.25;
    case "table": return tableHeight(b.rows, width, Math.max(12, pt - 3));
    case "img": return b.fitH || width / b.aspect;
    case "cols": return Math.max(...b.cols.map(c => stackHeight(c, (width - 0.4) / 2, pt)));
  }
  return 0;
}
const GAP = 0.18;
function stackHeight(blocks, width, pt) {
  return blocks.reduce((h, b) => h + blockHeight(b, width, pt), 0) + GAP * Math.max(0, blocks.length - 1);
}

// ---------------------------------------------------------------- drawing
function drawBlocks(slide, blocks, x, y, width, pt) {
  for (const b of blocks) {
    const h = blockHeight(b, width, pt);
    const base = { fontFace: BODY, fontSize: pt, color: TEXT || COL.body };
    if (b.type === "h2") {
      slide.addText(runs(b.text, { ...base, fontSize: pt + 2, bold: true, color: HEADC || COL.ink }), { x, y, w: width, h, margin: 0, valign: "top", isTextBox: true });
    } else if (b.type === "p") {
      const r = [];
      b.lines.forEach((l, k) => {
        const rr = runs(l, base);
        if (k < b.lines.length - 1) rr[rr.length - 1].options.breakLine = true;
        r.push(...rr);
      });
      slide.addText(r, { x, y, w: width, h, margin: 0, valign: "top", isTextBox: true });
    } else if (b.type === "ul" || b.type === "ol") {
      const r = [];
      b.items.forEach((it, k) => {
        const rr = runs(it, base);
        rr[0].options.bullet = b.type === "ol" ? { type: "number" } : { indent: 18 };
        rr[0].options.paraSpaceAfter = pt * 0.35;
        if (k < b.items.length - 1) rr[rr.length - 1].options.breakLine = true;
        r.push(...rr);
      });
      slide.addText(r, { x, y, w: width, h, margin: 0, valign: "top", isTextBox: true });
    } else if (b.type === "code") {
      slide.addText(b.text, { x, y, w: width, h, fontFace: MONO, fontSize: pt - 3, color: COL.ink, fill: { color: COL.zebra }, margin: 0.12, valign: "middle", isTextBox: true });
    } else if (b.type === "table") {
      const tp = Math.max(12, pt - 3);
      const ws = colWidths(b.rows, width, tp);
      const rows = b.rows.map((r, ri) => r.map(c => ({
        text: ri === 0 ? [{ text: plain(c), options: { bold: true, color: COL.white, fontFace: BODY, fontSize: tp } }] : (c ? runs(c, { fontFace: BODY, fontSize: tp, color: COL.body }) : [{ text: " ", options: { fontFace: BODY, fontSize: tp } }]),
        options: { fill: { color: ri === 0 ? COL.ink : (ri % 2 ? COL.white : COL.zebra) }, valign: "middle" },
      })));
      slide.addTable(rows, { x, y, w: width, colW: ws, border: { type: "solid", pt: 0.75, color: COL.rule }, margin: [3, 6, 3, 6] });
    } else if (b.type === "img") {
      const ih = h, iw = Math.min(width, ih * b.aspect), ihh = iw / b.aspect;
      const box = { x: x + (width - iw) / 2, y, w: iw, h: ihh };
      if (b.video) slide.addMedia({ type: "video", path: b.video, cover: "data:image/png;base64," + fs.readFileSync(b.file).toString("base64"), ...box });
      else slide.addImage({ path: b.file, ...box, altText: path.basename(b.file, ".png").replace(/_/g, " ") });
    } else if (b.type === "cols") {
      const cw = (width - 0.4) / 2;
      b.cols.forEach((c, k) => drawBlocks(slide, c, x + k * (cw + 0.4), y, cw, pt));
    }
    y += h + GAP;
  }
}

// ---------------------------------------------------------------- deck
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Packt Nov 2026 · Time Series Bootcamp · Day 2";
pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };

const slides = parseSlides(fs.readFileSync(SRC, "utf8"));
const total = slides.length;

slides.forEach((s, idx) => {
  const slide = pres.addSlide();
  const h1 = s.blocks.find(b => b.type === "h1");
  const rest = s.blocks.filter(b => b !== h1);
  const isDark = idx === 0 || /^Break/.test(h1?.text || "");

  if (isDark) {
    slide.background = { color: COL.ink };
    const subs = rest.filter(b => b.type === "h2");
    const others = rest.filter(b => b.type !== "h2");
    slide.addShape(pres.shapes.OVAL, { x: MX, y: 2.05, w: 0.5, h: 0.5, fill: { color: COL.accent }, line: { type: "none" } });
    slide.addText(plain(h1.text), { x: MX, y: 2.7, w: CW, h: 1.1, fontFace: HEAD, fontSize: 44, bold: true, color: COL.white, margin: 0, isTextBox: true });
    let y = 3.85;
    subs.forEach(b => { slide.addText(plain(b.text), { x: MX, y, w: CW, h: 0.6, fontFace: HEAD, fontSize: 24, color: "9FC3D6", margin: 0, isTextBox: true }); y += 0.7; });
    others.forEach(b => {
      if (b.type === "p") b.lines.forEach(l => { slide.addText(plain(l), { x: MX, y: y + 0.3, w: CW, h: 0.4, fontFace: BODY, fontSize: 16, color: "C9D6DD", margin: 0, isTextBox: true }); y += 0.42; });
    });
    if (s.notes) slide.addNotes(s.notes);
    return;
  }

  const lab = s.cls === "lab";
  if (lab) slide.background = { color: COL.tint };
  if (s.cls === "result") slide.background = { color: COL.greenTint };
  const summary = s.cls === "summary";
  TEXT = summary ? "E8F0F4" : null; HEADC = summary ? "FFFFFF" : null;
  if (summary) slide.background = { color: COL.ink };
  const take = rest.find(b => b.type === "takeaway");
  const body = rest.filter(b => b.type !== "takeaway");
  const bottom = take ? 6.05 : BOTTOM;
  let title = h1 ? h1.text : "";
  let pill = null;
  if (lab) {
    const m = title.match(/^(.*?)\s*·\s*`(\d+)`\s*\((\d+ min)\)/);
    if (m) { title = m[1]; pill = `LAB  ·  ${m[2]}  ·  ${m[3]}`; }
  }
  let pillColor = COL.accent;
  if (s.cls === "intuition") { pill = "INTUITION"; pillColor = COL.blue; }
  if (s.cls === "result") { pill = "WHAT WE FOUND"; pillColor = COL.green; }
  if (summary) { pill = "SUMMARY"; pillColor = COL.accent; }
  const titleW = pill ? CW - 3.0 : CW;
  slide.addText(plain(title), { x: MX, y: 0.45, w: titleW, h: 0.85, fontFace: HEAD, fontSize: 30, bold: true, color: summary ? COL.white : COL.ink, margin: 0, valign: "middle", isTextBox: true });
  if (pill) {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: W - MX - 2.7, y: 0.6, w: 2.7, h: 0.55, rectRadius: 0.27, fill: { color: pillColor }, line: { type: "none" } });
    slide.addText(pill, { x: W - MX - 2.7, y: 0.6, w: 2.7, h: 0.55, fontFace: HEAD, fontSize: 14, bold: true, color: COL.white, align: "center", valign: "middle", margin: 0 });
  }

  // fit body: shrink font until the stack fits
  let pt = s.cls === "dense" ? 17 : 19;
  const img = body.find(b => b.type === "img");
  if (img) {
    // text first, the picture takes what is left (but never less than 2.6in)
    const fit = () => {
      const textH = stackHeight(body.filter(b => b !== img), CW, pt) + GAP;
      img.fitH = Math.min(CW / img.aspect, 3.9, bottom - TOP - textH);
    };
    pt = 18; fit();
    while (pt > 14 && img.fitH < 2.6) { pt--; fit(); }
  }
  while (pt > 14 && stackHeight(body, CW, pt) > bottom - TOP) pt--;
  const hgt = stackHeight(body, CW, pt);
  if (hgt > bottom - TOP) console.warn(`slide ${idx + 1} may overflow: ${hgt.toFixed(2)}in`);
  drawBlocks(slide, body, MX, TOP, CW, pt);
  if (take) {
    const ty = 6.18, th = 0.62;
    slide.addShape(pres.shapes.RECTANGLE, { x: MX, y: ty, w: CW, h: th, fill: { color: summary ? "16384B" : "FFF4EC" }, line: { type: "none" } });
    slide.addShape(pres.shapes.RECTANGLE, { x: MX, y: ty, w: 0.09, h: th, fill: { color: COL.accent }, line: { type: "none" } });
    const tr = runs(take.text, { fontFace: BODY, fontSize: 16, color: summary ? COL.white : COL.ink });
    slide.addText(tr, { x: MX + 0.25, y: ty, w: CW - 0.35, h: th, margin: 0, valign: "middle", isTextBox: true, fit: "shrink" });
  }

  slide.addText(`Packt Nov 2026 · Day 2`, { x: MX, y: 7.0, w: 6, h: 0.3, fontFace: BODY, fontSize: 10, color: summary ? "9FC3D6" : COL.muted, margin: 0, isTextBox: true });
  slide.addText(`${idx + 1} / ${total}`, { x: W - MX - 2, y: 7.0, w: 2, h: 0.3, fontFace: BODY, fontSize: 10, color: summary ? "9FC3D6" : COL.muted, align: "right", margin: 0, isTextBox: true });
  if (s.notes) slide.addNotes(s.notes);
});

pres.writeFile({ fileName: OUT }).then(f => console.log("wrote", f));
