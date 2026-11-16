"""Extras for the defense: animated GIFs and a self-contained interactive viewer.

    figures/demo_D_sys-v1.gif, figures/demo_D_sys-v2.gif
    viewer.html  (open it in any browser, works offline)
"""
from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from controllers import METHODS
from plots import MODE_COLORS, arr, col, draw_arena
from scenarios import SCENARIOS

TEST = ("D", "E", "F")
SYSTEMS = ("sys-v1", "sys-v2")
STRIDE = 4            # keep one sample out of 4 (0.2 s) in the viewer
VIEW_SEEDS = 5        # nominal + seeds 0..4 in the viewer


def run(ctx):
    print("\nExtras: GIF animations + interactive viewer")
    nominal, noisy = ctx.campaign
    for sysv in SYSTEMS:
        gif(ctx, nominal, "D", sysv)
    viewer(ctx, nominal, noisy)


def gif(ctx, nominal, sid, sysv, frame_dt=0.5):
    p = ctx.params
    sc = SCENARIOS[sid]
    runs = {m: nominal[(f"{m}|{sysv}", sid)] for m in METHODS}
    t_end = max(arr(r, "t")[-1] for r in runs.values())
    times = np.arange(0, t_end + frame_dt, frame_dt)
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    draw_arena(ax, sc, p)
    trails, bodies, heads = {}, {}, {}
    for m in METHODS:
        trails[m], = ax.plot([], [], color=col(m), lw=2, label=m)
        bodies[m] = plt.Circle((0, 0), p.robot_radius, fc=col(m), ec="k", alpha=0.85, zorder=6)
        ax.add_patch(bodies[m])
        heads[m], = ax.plot([], [], color="w", lw=2, zorder=7)
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=8, family="monospace",
                  bbox=dict(fc="w", ec="0.7", alpha=0.9))
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"Scenario {sid} - {sysv} - nominal run")

    def update(t):
        lines = [f"t = {t:5.1f} s"]
        for m, r in runs.items():
            k = min(int(round(t / p.dt)), len(r.rows) - 1)
            row = r.rows[k]
            trails[m].set_data(arr(r, "x")[:k + 1], arr(r, "y")[:k + 1])
            bodies[m].center = (row["x"], row["y"])
            heads[m].set_data([row["x"], row["x"] + p.robot_radius * math.cos(row["theta"])],
                              [row["y"], row["y"] + p.robot_radius * math.sin(row["theta"])])
            state = row["mode"] if row["status"] == "running" else row["status"]
            lines.append(f"{m:8s} {state:9s} v={row['v']:.2f}")
        txt.set_text("\n".join(lines))
        return []

    anim = FuncAnimation(fig, update, frames=times, blit=False)
    path = ctx.figs / f"demo_{sid}_{sysv}.gif"
    anim.save(path, writer=PillowWriter(fps=20), dpi=80)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ctx.root)}")


def _pack(r):
    rows = r.rows[::STRIDE]
    if rows[-1] is not r.rows[-1]:
        rows = rows + [r.rows[-1]]
    code = {"goal": "g", "avoid": "a", "stop": "s", "arrived": "r"}
    return {
        "status": r.status,
        "T": round(r.rows[-1]["t"], 2),
        "t": [round(q["t"], 2) for q in rows],
        "x": [round(q["x"], 3) for q in rows],
        "y": [round(q["y"], 3) for q in rows],
        "th": [round(q["theta"], 3) for q in rows],
        "v": [round(q["v"], 3) for q in rows],
        "w": [round(q["omega"], 3) for q in rows],
        "rho": [round(q["rho"], 3) for q in rows],
        "mode": "".join(code.get(q["mode"], "g") for q in rows),
        "guard": None if math.isinf(r.min_guard) else round(r.min_guard, 3),
    }


def viewer(ctx, nominal, noisy):
    p = ctx.params
    eps = {}
    for sysv in SYSTEMS:
        for m in METHODS:
            for sid in TEST:
                key = f"{m}|{sysv}"
                eps[f"{sid}|{sysv}|{m}|nominal"] = _pack(nominal[(key, sid)])
                for s, r in enumerate(noisy[(key, sid)][:VIEW_SEEDS]):
                    eps[f"{sid}|{sysv}|{m}|seed{s}"] = _pack(r)
    agg = [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in a.items()}
           for a in getattr(ctx, "campaign_agg", [])]
    data = {
        "params": {"R": p.robot_radius, "margin": p.margin, "tol": p.goal_tol, "arena": p.arena,
                   "version": p.version, "n_seeds": len(ctx.seeds)},
        "scenarios": {sid: {"start": SCENARIOS[sid].start, "goal": SCENARIOS[sid].goal,
                            "obstacles": SCENARIOS[sid].obstacles,
                            "desc": SCENARIOS[sid].description} for sid in TEST},
        "methods": list(METHODS), "systems": list(SYSTEMS),
        "colors": {m: col(m) for m in METHODS}, "modeColors": MODE_COLORS,
        "runs": ["nominal"] + [f"seed{s}" for s in range(VIEW_SEEDS)],
        "episodes": eps, "agg": agg,
    }
    html = TEMPLATE.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    path = ctx.root / "viewer.html"
    path.write_text(html, encoding="utf-8")
    print(f"  wrote viewer.html ({path.stat().st_size / 1e6:.1f} MB) - open it in a browser")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fuzzy navigation - replay viewer</title>
<style>
:root{--bg:#f6f5f1;--panel:#fff;--ink:#1d1d1b;--mute:#6b6b66;--line:#e2e0d8;--acc:#1d1d1b}
*{box-sizing:border-box}
body{margin:0;font:14px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--ink)}
header{padding:20px 28px 8px}
h1{font:600 20px/1.2 ui-serif,Georgia,serif;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--mute);font-size:13px}
main{display:grid;grid-template-columns:minmax(360px,640px) minmax(320px,1fr);gap:18px;padding:12px 28px 28px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.controls{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;margin-bottom:10px}
label{font-size:12px;color:var(--mute);display:flex;gap:6px;align-items:center}
select,button{font:inherit;font-size:13px;padding:5px 9px;border:1px solid var(--line);border-radius:6px;background:#fff;color:var(--ink);cursor:pointer}
button.primary{background:var(--acc);color:#fff;border-color:var(--acc);min-width:76px}
canvas{display:block;width:100%;border-radius:6px}
#arena{aspect-ratio:1/1;background:#fbfaf7;border:1px solid var(--line)}
input[type=range]{width:100%;accent-color:var(--acc)}
.time{display:flex;gap:10px;align-items:center;margin-top:8px;font-variant-numeric:tabular-nums}
.time span{min-width:76px;font-size:13px}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--mute);font-weight:500}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}
.chip{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11.5px;color:#fff}
h2{font:600 13px/1.2 ui-sans-serif,system-ui;margin:2px 0 10px;color:var(--mute);text-transform:uppercase;letter-spacing:.06em}
.grid2{display:grid;gap:14px}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:var(--mute);margin-top:8px}
@media (max-width:900px){main{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Soft computing - mobile robot navigation: replay viewer</h1>
<div class="sub" id="sub"></div></header>
<main>
<section class="card">
  <div class="controls">
    <label>Scenario <select id="sc"></select></label>
    <label>System <select id="sys"></select></label>
    <label>Run <select id="run"></select></label>
    <label>Speed <select id="speed"><option>1</option><option>2</option><option selected>5</option><option>10</option><option>20</option></select>x</label>
    <span id="checks"></span>
  </div>
  <canvas id="arena" width="1200" height="1200"></canvas>
  <div class="time"><button class="primary" id="play">Play</button>
    <input type="range" id="scrub" min="0" max="1000" value="0"><span id="tlab">t = 0.0 s</span></div>
  <div class="legend" id="modeLegend"></div>
</section>
<section class="grid2">
  <div class="card"><h2>Live state</h2><table id="live"></table></div>
  <div class="card"><h2>Distance to goal rho(t) and angular speed omega(t)</h2>
    <canvas id="chart" width="1100" height="420"></canvas></div>
  <div class="card"><h2>Campaign summary (noisy runs)</h2><table id="summary"></table>
    <div class="sub" style="margin-top:6px">Time and length over successful runs only (n). Generated by run_experiments.py.</div></div>
</section>
</main>
<script>
const D=/*__DATA__*/null;
const $=id=>document.getElementById(id);
const P=D.params;
let t=0,playing=false,last=null,enabled={};
D.methods.forEach(m=>enabled[m]=true);
$("sub").textContent=`${P.version} - robot radius ${P.R} m, desired margin ${P.margin} m - nominal run + ${D.runs.length-1} noisy seeds per method (full campaign: ${P.n_seeds} seeds)`;
for(const s of Object.keys(D.scenarios)) $("sc").add(new Option(`${s} - ${D.scenarios[s].desc}`,s));
D.systems.forEach(s=>$("sys").add(new Option(s,s)));
D.runs.forEach(r=>$("run").add(new Option(r,r)));
$("checks").innerHTML=D.methods.map(m=>`<label><input type="checkbox" data-m="${m}" checked><span class="dot" style="background:${D.colors[m]}"></span>${m}</label>`).join("");
document.querySelectorAll("#checks input").forEach(c=>c.onchange=()=>{enabled[c.dataset.m]=c.checked;draw()});
$("modeLegend").innerHTML="Trail colour = supervision mode: "+Object.entries(D.modeColors).map(([k,c])=>`<span><span class="dot" style="background:${c}"></span>${k}</span>`).join("");
const eps=()=>D.methods.map(m=>[m,D.episodes[`${$("sc").value}|${$("sys").value}|${m}|${$("run").value}`]]);
const tEnd=()=>Math.max(...eps().map(([m,e])=>e.T));
function idx(e,tt){let lo=0,hi=e.t.length-1;if(tt>=e.t[hi])return hi;while(hi-lo>1){const mid=(lo+hi)>>1;if(e.t[mid]<=tt)lo=mid;else hi=mid}return lo}
const MC={g:D.modeColors.goal,a:D.modeColors.avoid,s:D.modeColors.stop,r:D.modeColors.arrived};
const MN={g:"goal",a:"avoid",s:"stop",r:"arrived"};
function draw(){
  const cv=$("arena"),c=cv.getContext("2d"),W=cv.width,pad=40,S=(W-2*pad)/P.arena;
  const X=x=>pad+x*S,Y=y=>W-pad-y*S;
  c.clearRect(0,0,W,W);
  c.strokeStyle="#e9e7df";c.lineWidth=1;
  for(let i=0;i<=P.arena;i++){c.beginPath();c.moveTo(X(i),Y(0));c.lineTo(X(i),Y(P.arena));c.moveTo(X(0),Y(i));c.lineTo(X(P.arena),Y(i));c.stroke()}
  c.fillStyle="#8a877c";c.font="22px system-ui";
  for(let i=0;i<=P.arena;i+=2){c.fillText(i,X(i)-6,W-8);c.fillText(i,6,Y(i)+7)}
  c.strokeStyle="#1d1d1b";c.lineWidth=3;c.strokeRect(X(0),Y(P.arena),P.arena*S,P.arena*S);
  const sc=D.scenarios[$("sc").value];
  for(const [cx,cy,r] of sc.obstacles){
    c.setLineDash([10,8]);c.strokeStyle="#d64545";c.lineWidth=2;c.beginPath();c.arc(X(cx),Y(cy),(r+P.R+P.margin)*S,0,7);c.stroke();c.setLineDash([]);
    c.fillStyle="#9b988e";c.strokeStyle="#1d1d1b";c.beginPath();c.arc(X(cx),Y(cy),r*S,0,7);c.fill();c.stroke()}
  c.strokeStyle="#2ca02c";c.lineWidth=3;c.beginPath();c.arc(X(sc.goal[0]),Y(sc.goal[1]),P.tol*S,0,7);c.stroke();
  c.fillStyle="#2ca02c";c.beginPath();c.arc(X(sc.goal[0]),Y(sc.goal[1]),7,0,7);c.fill();
  c.fillStyle="#1d1d1b";c.fillRect(X(sc.start[0])-8,Y(sc.start[1])-8,16,16);
  const rows=[];
  for(const [m,e] of eps()){
    if(!enabled[m])continue;
    const k=idx(e,t);
    c.lineWidth=4;c.lineCap="round";
    for(let i=1;i<=k;i++){c.strokeStyle=MC[e.mode[i-1]];c.beginPath();c.moveTo(X(e.x[i-1]),Y(e.y[i-1]));c.lineTo(X(e.x[i]),Y(e.y[i]));c.stroke()}
    c.lineWidth=1.5;c.strokeStyle=D.colors[m];c.beginPath();for(let i=0;i<=k;i++){i?c.lineTo(X(e.x[i]),Y(e.y[i])):c.moveTo(X(e.x[i]),Y(e.y[i]))}c.stroke();
    const x=e.x[k],y=e.y[k],th=e.th[k];
    c.fillStyle=D.colors[m];c.globalAlpha=.9;c.beginPath();c.arc(X(x),Y(y),P.R*S,0,7);c.fill();c.globalAlpha=1;
    c.strokeStyle="#fff";c.lineWidth=4;c.beginPath();c.moveTo(X(x),Y(y));c.lineTo(X(x+P.R*Math.cos(th)),Y(y+P.R*Math.sin(th)));c.stroke();
    if(k===e.t.length-1&&e.status==="collision"){c.strokeStyle="#000";c.lineWidth=5;const q=14;c.beginPath();c.moveTo(X(x)-q,Y(y)-q);c.lineTo(X(x)+q,Y(y)+q);c.moveTo(X(x)+q,Y(y)-q);c.lineTo(X(x)-q,Y(y)+q);c.stroke()}
    const done=t>=e.T, md=done?e.status:MN[e.mode[k]];
    rows.push(`<tr><td><span class="dot" style="background:${D.colors[m]}"></span>${m}</td><td><span class="chip" style="background:${done?(e.status==="arrived"?MC.r:"#333"):MC[e.mode[k]]}">${md}</span></td><td>${e.rho[k].toFixed(2)} m</td><td>${e.v[k].toFixed(2)} m/s</td><td>${e.w[k].toFixed(2)} rad/s</td><td>${e.status==="arrived"?e.T.toFixed(2)+" s":e.status}</td><td>${e.guard===null?"-":e.guard.toFixed(2)+" m"}</td></tr>`)}
  $("live").innerHTML=`<tr><th>method</th><th>mode</th><th>rho</th><th>v</th><th>omega</th><th>final</th><th>min guard</th></tr>`+rows.join("");
  $("tlab").textContent=`t = ${t.toFixed(1)} s`;
  $("scrub").value=Math.round(1000*t/tEnd());
  chart();
}
function chart(){
  const cv=$("chart"),c=cv.getContext("2d"),W=cv.width,H=cv.height,l=60,r=12,h=(H-50)/2,T=tEnd();
  c.clearRect(0,0,W,H);c.font="20px system-ui";
  const panels=[["rho",0,16,"rho [m]"],["w",-1.6,1.6,"omega [rad/s]"]];
  panels.forEach(([key,lo,hi,lab],p)=>{
    const top=10+p*(h+20),X=tt=>l+(W-l-r)*tt/T,Y=v=>top+h-(v-lo)/(hi-lo)*h;
    c.strokeStyle="#e2e0d8";c.strokeRect(l,top,W-l-r,h);c.fillStyle="#6b6b66";c.fillText(lab,l+8,top+22);
    c.fillText(hi,8,top+16);c.fillText(lo,8,top+h);
    for(const [m,e] of eps()){if(!enabled[m])continue;c.strokeStyle=D.colors[m];c.lineWidth=2;c.beginPath();
      e.t.forEach((tt,i)=>{const v=key==="rho"?e.rho[i]:e.w[i];i?c.lineTo(X(tt),Y(v)):c.moveTo(X(tt),Y(v))});c.stroke()}
    c.strokeStyle="#1d1d1b";c.setLineDash([6,6]);c.beginPath();c.moveTo(X(t),top);c.lineTo(X(t),top+h);c.stroke();c.setLineDash([])});
}
function summary(){
  const s=$("sc").value,sy=$("sys").value;
  const f=(a,b,n)=>a===null?"-":`${a.toFixed(1)} ± ${b.toFixed(1)} (${n})`;
  const rows=D.agg.filter(a=>a.scenario===s&&a.system===sy&&a.condition==="noisy").map(a=>`<tr><td><span class="dot" style="background:${D.colors[a.controller]}"></span>${a.controller}</td><td>${a.success}/${a.N}</td><td>${a.collision}/${a.N}</td><td>${f(a.T_mean,a.T_std,a.n_used)}</td><td>${f(a.L_mean,a.L_std,a.n_used)}</td><td>${a.min_guard_mean.toFixed(2)} (${a.min_guard_worst.toFixed(2)})</td><td>${a.mode_switches_mean.toFixed(0)}</td></tr>`);
  $("summary").innerHTML=`<tr><th>method</th><th>success</th><th>collisions</th><th>time [s]</th><th>length [m]</th><th>guard mean (worst) [m]</th><th>switches</th></tr>`+rows.join("");
}
function tick(ts){if(!playing){last=null;return}if(last!==null)t+=(ts-last)/1000*(+$("speed").value);last=ts;
  if(t>=tEnd()){t=tEnd();playing=false;$("play").textContent="Replay"}draw();requestAnimationFrame(tick)}
$("play").onclick=()=>{if(!playing&&t>=tEnd())t=0;playing=!playing;$("play").textContent=playing?"Pause":"Play";if(playing)requestAnimationFrame(tick)};
$("scrub").oninput=e=>{t=tEnd()*e.target.value/1000;draw()};
["sc","sys","run"].forEach(id=>$(id).onchange=()=>{t=0;playing=false;$("play").textContent="Play";summary();draw()});
summary();draw();
</script></body></html>
"""
