#!/usr/bin/env python3
"""Assemble the self-contained blind faithfulness-annotation website.

Reads human_eval/web_items.json (80 blind items) and writes
human_eval/annotate.html with the data embedded, so it runs from a single file
with no network. Render of all data fields is via textContent (XSS-safe).
"""
import json
from pathlib import Path

OUT = Path("<DATA_ROOT>")
items = json.loads((OUT / "web_items.json").read_text())
DATA = json.dumps(items, ensure_ascii=False)

HTML = r"""<style>
:root{
  --bg:#FAFAFB; --panel:#ffffff; --ink:#181a20; --muted:#5d6470; --line:#e5e7ec;
  --accent:#3b5ba9; --answer:#f4f6fb;
  --good:#2e7d52; --warn:#b7791f; --bad:#c0392b;
  --good-bg:#e8f3ec; --warn-bg:#f7efdd; --bad-bg:#f7e7e4;
  --serif:ui-serif,Georgia,"Iowan Old Style","Times New Roman",serif;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,"SF Mono","Cascadia Code",Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#14161c; --panel:#1c1f27; --ink:#e7e9ef; --muted:#9aa1b0; --line:#2b2f3a;
  --accent:#8aa0e0; --answer:#20242e;
  --good:#6fce9a; --warn:#e2b35a; --bad:#e88b7d;
  --good-bg:#1c2b23; --warn-bg:#2c2718; --bad-bg:#2c1e1c;
}}
:root[data-theme="dark"]{
  --bg:#14161c; --panel:#1c1f27; --ink:#e7e9ef; --muted:#9aa1b0; --line:#2b2f3a;
  --accent:#8aa0e0; --answer:#20242e;
  --good:#6fce9a; --warn:#e2b35a; --bad:#e88b7d;
  --good-bg:#1c2b23; --warn-bg:#2c2718; --bad-bg:#2c1e1c;
}
:root[data-theme="light"]{
  --bg:#FAFAFB; --panel:#ffffff; --ink:#181a20; --muted:#5d6470; --line:#e5e7ec;
  --accent:#3b5ba9; --answer:#f4f6fb;
  --good:#2e7d52; --warn:#b7791f; --bad:#c0392b;
  --good-bg:#e8f3ec; --warn-bg:#f7efdd; --bad-bg:#f7e7e4;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:820px;margin:0 auto;padding:0 20px}
button{font-family:inherit;cursor:pointer;border:1px solid var(--line);
  background:var(--panel);color:var(--ink);border-radius:8px;padding:8px 14px;font-size:14px}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.12em;
  text-transform:uppercase;color:var(--muted)}

/* header */
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line)}
.bar{height:4px;background:var(--line)}
.bar>i{display:block;height:100%;background:var(--accent);width:0;transition:width .2s}
.htop{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 0}
.htop h1{font-family:var(--serif);font-size:17px;margin:0;font-weight:600}
.counts{font-family:var(--mono);font-size:12px;color:var(--muted);
  font-variant-numeric:tabular-nums}
.ghost{background:transparent;border-color:transparent;color:var(--muted);padding:6px 8px}

/* card */
main{padding:22px 0 260px}
.qlabel{margin:0 0 6px}
.query{font-family:var(--serif);font-size:22px;line-height:1.35;margin:0 0 20px;
  text-wrap:balance;font-weight:600}
.answer{background:var(--answer);border:1px solid var(--line);border-left:3px solid var(--accent);
  border-radius:10px;padding:16px 18px;font-family:var(--serif);font-size:16px;
  line-height:1.6;white-space:pre-wrap}
.section-h{display:flex;align-items:baseline;justify-content:space-between;
  margin:26px 0 10px}
#sources{max-height:46vh;overflow-y:auto;border:1px solid var(--line);border-radius:10px;
  background:var(--panel);padding:4px 4px}
.srccard{border-bottom:1px solid var(--line);padding:12px 14px}
.srccard:last-child{border-bottom:none}
.snum{font-family:var(--mono);font-size:11px;letter-spacing:.05em;color:var(--accent);
  font-weight:700;display:block;margin-bottom:5px}
.stext{font-family:var(--serif);font-size:14.5px;line-height:1.62;color:var(--ink);
  white-space:pre-wrap}
.scrollhint{font-size:12px;color:var(--muted)}

/* rating dock */
.dock{position:fixed;left:0;right:0;bottom:0;background:var(--panel);
  border-top:1px solid var(--line);padding:12px 0;z-index:6}
.dockrow{display:flex;gap:10px;align-items:center}
.rate{flex:1;display:flex;flex-direction:column;align-items:center;gap:3px;
  padding:12px 8px;border-radius:10px;font-weight:600;font-size:15px}
.rate small{font-weight:500;color:var(--muted);font-size:11.5px}
.rate .kbd{font-family:var(--mono);font-size:10px;border:1px solid var(--line);
  border-radius:4px;padding:1px 5px;color:var(--muted)}
.rate[data-v="1"].on{background:var(--good-bg);border-color:var(--good);color:var(--good)}
.rate[data-v="0.5"].on{background:var(--warn-bg);border-color:var(--warn);color:var(--warn)}
.rate[data-v="0"].on{background:var(--bad-bg);border-color:var(--bad);color:var(--bad)}
.nav{display:flex;gap:8px;align-items:center;margin-top:10px;justify-content:space-between}
.nav .mid{display:flex;gap:14px;align-items:center;color:var(--muted);font-size:13px}
.extra{display:flex;gap:16px;align-items:center;margin-top:10px;flex-wrap:wrap}
.extra label{font-size:13px;color:var(--muted);display:flex;gap:7px;align-items:center}
.extra input[type=number]{width:56px}
.extra input,.extra select{font-family:var(--mono);background:var(--bg);color:var(--ink);
  border:1px solid var(--line);border-radius:6px;padding:5px 7px}
.notes{flex:1;min-width:180px}
.notes input{width:100%}
.primary{background:var(--accent);color:#fff;border-color:var(--accent);font-weight:600}
.primary:disabled{opacity:.4;cursor:default}

/* start + done overlays */
.overlay{position:fixed;inset:0;background:var(--bg);z-index:20;overflow:auto;
  display:flex;align-items:center;justify-content:center;padding:24px}
.card{max-width:600px;background:var(--panel);border:1px solid var(--line);
  border-radius:14px;padding:30px 32px}
.card h2{font-family:var(--serif);font-size:26px;margin:.1em 0 .4em;text-wrap:balance}
.card p{color:var(--muted);font-size:15px}
.card ol{color:var(--ink);font-size:14.5px;line-height:1.7;padding-left:20px}
.card .field{margin:18px 0 6px}
.card input[type=text]{width:100%;padding:10px 12px;border:1px solid var(--line);
  border-radius:8px;background:var(--bg);color:var(--ink);font-family:var(--mono);font-size:15px}
.legend{display:flex;gap:8px;margin:14px 0}
.chip{flex:1;text-align:center;border-radius:8px;padding:8px;font-size:12.5px;font-weight:600}
.chip.g{background:var(--good-bg);color:var(--good)} .chip.a{background:var(--warn-bg);color:var(--warn)}
.chip.b{background:var(--bad-bg);color:var(--bad)}
.hide{display:none!important}
@media (max-width:560px){.rate small{display:none}.query{font-size:19px}}
</style>

<header>
  <div class="bar"><i id="prog"></i></div>
  <div class="wrap htop">
    <h1>Faithfulness Review</h1>
    <div style="display:flex;gap:10px;align-items:center">
      <span class="counts" id="counts">0 / 0</span>
      <button class="ghost" id="themeBtn" title="Toggle theme">◐</button>
      <button class="ghost" id="exportTop">Export</button>
    </div>
  </div>
</header>

<main class="wrap" id="main">
  <p class="eyebrow qlabel">Question</p>
  <p class="query" id="query"></p>
  <p class="eyebrow">Answer under review</p>
  <div class="answer" id="answer"></div>
  <div class="section-h">
    <p class="eyebrow" style="margin:0">Retrieved evidence <span id="nsrc"></span></p>
    <span class="scrollhint">scroll within the box to read all sources</span>
  </div>
  <div id="sources"></div>
</main>

<div class="dock">
  <div class="wrap">
    <div class="dockrow">
      <button class="rate" data-v="1">Fully grounded <small>every claim supported</small><span class="kbd">1</span></button>
      <button class="rate" data-v="0.5">Partly grounded <small>some claims unsupported</small><span class="kbd">2</span></button>
      <button class="rate" data-v="0">Not grounded <small>key claims unsupported</small><span class="kbd">3</span></button>
    </div>
    <div class="extra">
      <label># unsupported claims
        <input type="number" id="unsup" min="0" step="1" placeholder="0"></label>
      <label class="notes">note
        <input type="text" id="note" placeholder="optional"></label>
    </div>
    <div class="nav">
      <button id="prev">← Prev</button>
      <div class="mid"><span id="pos"></span><label style="display:flex;gap:6px;align-items:center">
        <input type="checkbox" id="auto" checked> auto-advance</label></div>
      <button id="next" class="primary">Next →</button>
    </div>
  </div>
</div>

<div class="overlay" id="start">
  <div class="card">
    <p class="eyebrow">WYDOT RAG · blind evaluation</p>
    <h2>Rate how well each answer is grounded in its evidence</h2>
    <p>You will see <b id="nItems">80</b> answers, one at a time. For each, read the
      <b>Question</b>, the <b>Answer under review</b>, and the <b>Retrieved evidence</b>, then judge
      one thing only: <b>is every claim in the answer supported by the evidence shown?</b></p>
    <div class="legend">
      <div class="chip g">Fully grounded</div><div class="chip a">Partly grounded</div><div class="chip b">Not grounded</div>
    </div>
    <ol>
      <li>Judge <b>grounding only</b> — not whether the answer is correct or well-written.</li>
      <li>An answer can be correct yet <em>not grounded</em> (states facts the sources don't show), and grounded yet wrong.</li>
      <li>Optionally count claims with no support, and add a short note.</li>
      <li>Progress saves in this browser automatically; you can close and resume.</li>
      <li>Keys: <b>1 / 2 / 3</b> to rate, <b>← / →</b> to move.</li>
    </ol>
    <div class="field"><p class="eyebrow" style="margin:0 0 6px">Your initials (labels your export)</p>
      <input type="text" id="who" placeholder="e.g. NS" maxlength="16"></div>
    <div style="margin-top:18px;display:flex;gap:10px">
      <button class="primary" id="begin">Begin review</button>
      <button id="resume" class="hide">Resume where I left off</button>
    </div>
  </div>
</div>

<div class="overlay hide" id="results">
  <div class="card">
    <p class="eyebrow">Your results</p>
    <h2 id="resHead">Copy your ratings</h2>
    <p>Select all of the text below and copy it (or use the button), then paste it back
      in the chat. Your progress stays saved in this browser — copying changes nothing.</p>
    <div style="display:flex;gap:10px;margin:6px 0 12px">
      <button class="primary" id="copyBtn">Copy to clipboard</button>
      <button id="dlBtn">Try file download</button>
      <button id="resClose">Back to review</button>
    </div>
    <textarea id="resText" readonly
      style="width:100%;height:230px;font-family:var(--mono);font-size:12px;
      border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink);
      padding:10px;white-space:pre"></textarea>
    <p id="copyMsg" class="eyebrow" style="margin-top:8px"></p>
  </div>
</div>

<div class="overlay hide" id="done">
  <div class="card">
    <p class="eyebrow">Complete</p>
    <h2>All answers reviewed — export your results</h2>
    <p id="doneStat"></p>
    <p>Download the file and send it back. It contains only your ratings keyed by item id —
      no answer text.</p>
    <div style="display:flex;gap:10px;margin-top:8px">
      <button class="primary" id="dl">Download my results</button>
      <button id="review">Review answers</button>
    </div>
  </div>
</div>

<script>
const ITEMS = __DATA__;
const KEY = "wydot_faith_v1";
let S = load();
function load(){ try{return JSON.parse(localStorage.getItem(KEY))||{}}catch(e){return {}} }
function save(){ localStorage.setItem(KEY, JSON.stringify(S)); }
let i = 0;
const $ = s => document.querySelector(s);

// theme
$("#themeBtn").onclick = ()=>{
  const cur = document.documentElement.getAttribute("data-theme");
  const next = cur==="dark"?"light":(cur==="light"?"dark":
    (matchMedia("(prefers-color-scheme:dark)").matches?"light":"dark"));
  document.documentElement.setAttribute("data-theme", next);
};

function render(){
  const it = ITEMS[i];
  $("#query").textContent = it.query;
  $("#answer").textContent = it.answer;
  $("#nsrc").textContent = "(" + it.sources.length + ")";
  const box = $("#sources"); box.textContent = ""; box.scrollTop = 0;
  it.sources.forEach((s,n)=>{
    const card = document.createElement("div"); card.className="srccard";
    const num = document.createElement("span"); num.className="snum"; num.textContent="Source "+(n+1);
    const tx = document.createElement("div"); tx.className="stext"; tx.textContent=s;
    card.appendChild(num); card.appendChild(tx); box.appendChild(card);
  });
  const r = S[it.uid] || {};
  document.querySelectorAll(".rate").forEach(b=>b.classList.toggle("on", r.faithful===b.dataset.v));
  $("#unsup").value = r.unsup ?? "";
  $("#note").value  = r.note ?? "";
  $("#pos").textContent = "Item "+(i+1)+" of "+ITEMS.length;
  const done = Object.keys(S).filter(k=>S[k]&&S[k].faithful!=null).length;
  $("#counts").textContent = done+" / "+ITEMS.length;
  $("#prog").style.width = (done/ITEMS.length*100)+"%";
  $("#prev").disabled = i===0;
  window.scrollTo(0,0);
}
function setRating(v){
  const uid = ITEMS[i].uid;
  S[uid] = Object.assign({}, S[uid], {faithful:v});
  save(); render();
  if($("#auto").checked) setTimeout(next, 180);
}
function stashExtras(){
  const uid = ITEMS[i].uid, r = S[uid];
  if(!r) return;
  r.unsup = $("#unsup").value; r.note = $("#note").value; save();
}
function next(){ stashExtras(); if(i<ITEMS.length-1){i++;render();} else finish(); }
function prev(){ stashExtras(); if(i>0){i--;render();} }
document.querySelectorAll(".rate").forEach(b=> b.onclick=()=>setRating(b.dataset.v));
$("#next").onclick=next; $("#prev").onclick=prev;
$("#unsup").onchange=stashExtras; $("#note").onchange=stashExtras;

document.addEventListener("keydown",e=>{
  if(/input|textarea/i.test(e.target.tagName)) return;
  if(e.key==="1")setRating("1"); else if(e.key==="2")setRating("0.5");
  else if(e.key==="3"||e.key==="0")setRating("0");
  else if(e.key==="ArrowRight")next(); else if(e.key==="ArrowLeft")prev();
});

function buildExport(){
  const out = {annotator:S.__who||"anon", study:"paradox_faithfulness_v1",
    n_items:ITEMS.length, ratings:{}};
  ITEMS.forEach(it=>{ const r=S[it.uid]; if(r&&r.faithful!=null)
    out.ratings[it.uid]={faithful:parseFloat(r.faithful),
      unsupported:r.unsup?parseInt(r.unsup):null, note:r.note||""}; });
  return out;
}
function exportData(){
  stashExtras();
  const out=buildExport(), n=Object.keys(out.ratings).length;
  $("#resText").value=JSON.stringify(out,null,2);
  $("#resHead").textContent="Copy your ratings ("+n+" of "+ITEMS.length+" done)";
  $("#copyMsg").textContent="";
  $("#results").classList.remove("hide");
  $("#resText").focus(); $("#resText").select();
}
$("#exportTop").onclick=exportData; $("#dl").onclick=exportData;
$("#resClose").onclick=()=>$("#results").classList.add("hide");
$("#copyBtn").onclick=async()=>{
  const t=$("#resText"); t.select();
  try{ await navigator.clipboard.writeText(t.value);
       $("#copyMsg").textContent="Copied — now paste it back in the chat."; }
  catch(e){ try{ document.execCommand("copy");
       $("#copyMsg").textContent="Copied — now paste it back in the chat."; }
    catch(e2){ $("#copyMsg").textContent="Press Ctrl/Cmd+C to copy the selected text."; } }
};
$("#dlBtn").onclick=()=>{
  const who=(S.__who||"anon").replace(/[^a-z0-9]/gi,"_");
  try{ const blob=new Blob([$("#resText").value],{type:"application/json"});
    const a=document.createElement("a"); a.href=URL.createObjectURL(blob);
    a.download="paradox_"+who+".json"; document.body.appendChild(a); a.click(); a.remove();
    $("#copyMsg").textContent="If nothing downloaded, use Copy to clipboard instead."; }
  catch(e){ $("#copyMsg").textContent="Download blocked here — use Copy to clipboard instead."; }
};
function finish(){ const done=Object.keys(S).filter(k=>S[k]&&S[k].faithful!=null).length;
  $("#doneStat").textContent="You rated "+done+" of "+ITEMS.length+" answers.";
  $("#done").classList.remove("hide"); }
$("#review").onclick=()=>{ $("#done").classList.add("hide"); i=0; render(); };

// start screen
$("#nItems").textContent=ITEMS.length; $("#nItems2")&&($("#nItems2").textContent=ITEMS.length);
const hasProgress = Object.keys(S).some(k=>k[0]!=="_"&&S[k]&&S[k].faithful!=null);
if(hasProgress){ $("#resume").classList.remove("hide"); if(S.__who)$("#who").value=S.__who; }
function beginWith(resume){
  const who=$("#who").value.trim(); if(!who){ $("#who").focus(); return; }
  S.__who=who; save(); $("#start").classList.add("hide");
  if(resume){ const firstUn=ITEMS.findIndex(it=>!(S[it.uid]&&S[it.uid].faithful!=null));
    i=firstUn<0?0:firstUn; } else i=0;
  render();
}
$("#begin").onclick=()=>beginWith(false);
$("#resume").onclick=()=>beginWith(true);
$("#who").addEventListener("keydown",e=>{if(e.key==="Enter")beginWith(false)});
</script>
"""

html = HTML.replace("__DATA__", DATA)
(OUT / "annotate.html").write_text(html)
print(f"wrote {OUT/'annotate.html'} ({len(html)//1024} KB, {len(items)} items)")
