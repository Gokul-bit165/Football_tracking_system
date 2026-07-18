"""
Generate a fully self-contained dashboard HTML with:
- Chart.js embedded inline (no CDN needed)
- output_stats.json embedded inline (no fetch/CORS needed)
- Heatmap images embedded as base64 (no file:// issues)

Run:
  python generate_dashboard.py
Opens dashboard_standalone.html directly in the browser with no server needed.
"""

import json
import os
import base64
import webbrowser
import sys

STATS_FILE      = "output_stats.json"
CHARTJS_FILE    = "chart.min.js"
HM_A_FILE       = "output_heatmap_A.png"
HM_B_FILE       = "output_heatmap_B.png"
OUTPUT_HTML     = "dashboard_standalone.html"


def load_file(path: str, mode="r", encoding="utf-8"):
    if not os.path.exists(path):
        print(f"  WARN: {path} not found — skipping")
        return None
    with open(path, mode=mode, encoding=encoding if mode == "r" else None) as f:
        return f.read()


def img_to_b64(path: str) -> str:
    data = load_file(path, mode="rb")
    if data is None:
        return ""
    return "data:image/png;base64," + base64.b64encode(data).decode()


def main():
    print("Generating standalone dashboard...")

    # 1. Load stats JSON
    stats_raw = load_file(STATS_FILE)
    if stats_raw is None:
        print("ERROR: output_stats.json not found. Run run_pipeline.py first.")
        sys.exit(1)
    stats_data = json.loads(stats_raw)
    print(f"  Loaded stats: {len(stats_data['per_player'])} player records")

    # 2. Load Chart.js
    chartjs = load_file(CHARTJS_FILE)
    if chartjs is None:
        print("ERROR: chart.min.js not found. Run: python -c \"import urllib.request; urllib.request.urlretrieve('https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js', 'chart.min.js')\"")
        sys.exit(1)
    print(f"  Loaded Chart.js: {len(chartjs)//1024} KB")

    # 3. Load heatmap images as base64
    hm_a = img_to_b64(HM_A_FILE)
    hm_b = img_to_b64(HM_B_FILE)
    print(f"  Heatmap A: {'embedded' if hm_a else 'not found'}")
    print(f"  Heatmap B: {'embedded' if hm_b else 'not found'}")

    # 4. Build HTML
    html = build_html(chartjs, json.dumps(stats_data), hm_a, hm_b)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  Saved: {OUTPUT_HTML}")

    # 5. Open in browser
    abs_path = os.path.abspath(OUTPUT_HTML)
    print(f"  Opening in browser: {abs_path}")
    webbrowser.open(f"file:///{abs_path.replace(os.sep, '/')}")
    print("\nDone! Dashboard is ready.")


def build_html(chartjs: str, stats_json: str, hm_a_b64: str, hm_b_b64: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Football Analytics Dashboard</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    :root {{
      --bg: #080c14; --surface: #0e1520; --card: #141c2b; --border: #1f2d42;
      --accent-a: #e8e8e8; --accent-b: #e03050; --accent-gold: #f5a623;
      --accent-cyan: #00d4ff; --text: #e2e8f0; --muted: #7a8fa8;
      --green: #22c55e; --radius: 14px;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: var(--bg); color: var(--text); font-family: 'Inter', system-ui, sans-serif; min-height: 100vh; }}

    header {{
      background: linear-gradient(135deg, #0e1520 0%, #0a1628 100%);
      border-bottom: 1px solid var(--border);
      padding: 20px 40px; display: flex; align-items: center; gap: 16px;
    }}
    .logo-badge {{
      width: 44px; height: 44px;
      background: linear-gradient(135deg, var(--accent-cyan), #0077ff);
      border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 22px;
    }}
    header h1 {{ font-size: 1.4rem; font-weight: 700; letter-spacing: -0.02em; }}
    header p  {{ font-size: 0.78rem; color: var(--muted); margin-top: 2px; }}
    .badge {{
      margin-left: auto; background: rgba(0,212,255,0.1);
      border: 1px solid rgba(0,212,255,0.3); color: var(--accent-cyan);
      padding: 4px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;
    }}

    main {{ padding: 32px 40px; display: flex; flex-direction: column; gap: 28px; }}
    .row {{ display: grid; gap: 20px; }}
    .row-3 {{ grid-template-columns: repeat(3, 1fr); }}
    .row-2 {{ grid-template-columns: 1fr 1fr; }}
    .row-4 {{ grid-template-columns: repeat(4, 1fr); }}

    .card {{
      background: var(--card); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 22px;
      transition: border-color 0.2s;
    }}
    .card:hover {{ border-color: #2a3d58; }}
    .card-title {{
      font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.1em; color: var(--muted); margin-bottom: 14px;
    }}

    .kpi {{ text-align: center; }}
    .kpi .value {{ font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em; }}
    .kpi .label {{ font-size: 0.75rem; color: var(--muted); margin-top: 4px; }}
    .kpi.team-a .value {{ color: var(--accent-a); }}
    .kpi.team-b .value {{ color: var(--accent-b); }}
    .kpi.gold .value {{ color: var(--accent-gold); }}
    .kpi.cyan .value {{ color: var(--accent-cyan); }}

    .poss-section {{ display: flex; flex-direction: column; gap: 12px; }}
    .poss-labels {{ display: flex; justify-content: space-between; font-size: 0.82rem; }}
    .poss-labels .team-a {{ color: var(--accent-a); font-weight: 700; }}
    .poss-labels .team-b {{ color: var(--accent-b); font-weight: 700; }}
    .poss-bar-wrap {{ height: 24px; border-radius: 12px; background: var(--border); overflow: hidden; display: flex; }}
    .poss-bar-a {{
      background: linear-gradient(90deg, #a0a0a0, #e8e8e8);
      display: flex; align-items: center; justify-content: center;
      font-size: 0.72rem; font-weight: 700; color: #111; transition: width 1s ease;
    }}
    .poss-bar-b {{
      background: linear-gradient(90deg, #c01030, #e03050);
      display: flex; align-items: center; justify-content: center;
      font-size: 0.72rem; font-weight: 700; color: #fff; flex: 1;
    }}

    .stat-row {{ display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid var(--border); }}
    .stat-row:last-child {{ border-bottom: none; }}
    .stat-row .stat-label {{ font-size: 0.82rem; color: var(--muted); }}
    .stat-values {{ display: flex; gap: 20px; }}
    .stat-val-a {{ font-weight: 700; color: var(--accent-a); min-width: 65px; text-align: right; }}
    .stat-val-b {{ font-weight: 700; color: var(--accent-b); min-width: 65px; text-align: right; }}

    .chart-wrap {{ position: relative; height: 240px; }}
    .chart-wrap.tall {{ height: 320px; }}

    .heatmap-img {{ width: 100%; border-radius: 8px; border: 1px solid var(--border); }}
    .no-hm {{ color: var(--muted); font-size: 0.8rem; text-align: center; padding: 60px 0; }}

    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
    thead th {{
      text-align: left; padding: 8px 12px;
      font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.08em; color: var(--muted); border-bottom: 1px solid var(--border);
    }}
    tbody tr {{ border-bottom: 1px solid rgba(255,255,255,0.04); transition: background 0.15s; }}
    tbody tr:hover {{ background: rgba(255,255,255,0.03); }}
    tbody td {{ padding: 9px 12px; }}

    .team-chip {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.68rem; font-weight: 600; }}
    .chip-a {{ background: rgba(232,232,232,0.12); color: #e8e8e8; }}
    .chip-b {{ background: rgba(224,48,80,0.15); color: #e03050; }}
    .chip-gk {{ background: rgba(0,212,255,0.12); color: #00d4ff; }}
    .chip-ref {{ background: rgba(34,197,94,0.12); color: #22c55e; }}

    .section-label {{
      font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.12em; color: var(--muted); margin-bottom: -8px;
    }}
    .filter-btn {{
      background: var(--border); border: none; color: var(--muted);
      padding: 5px 14px; border-radius: 6px; cursor: pointer;
      font-size: 0.75rem; font-weight: 600; transition: all 0.15s; font-family: inherit;
    }}
    .filter-btn:hover {{ background: #2a3d58; color: var(--text); }}
    .filter-btn.active {{ background: var(--accent-cyan); color: #000; }}

    @media (max-width: 900px) {{
      .row-3, .row-4 {{ grid-template-columns: 1fr 1fr; }}
      .row-2 {{ grid-template-columns: 1fr; }}
      main {{ padding: 20px; }}
    }}
    @media (max-width: 600px) {{
      .row-3, .row-4, .row-2 {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>

<header>
  <div class="logo-badge">⚽</div>
  <div>
    <h1>Football Analytics Dashboard</h1>
    <p>YOLO11m · ByteTrack · Kalman Filter · Homography Projection</p>
  </div>
  <span class="badge" id="frame-badge">Loading...</span>
</header>

<main>

  <!-- KPI Row -->
  <p class="section-label">Match Overview</p>
  <div class="row row-4">
    <div class="card kpi team-a"><div class="value" id="kpi-poss-a">—</div><div class="label">Team A Possession</div></div>
    <div class="card kpi team-b"><div class="value" id="kpi-poss-b">—</div><div class="label">Team B Possession</div></div>
    <div class="card kpi gold"><div class="value" id="kpi-events">—</div><div class="label">Total Events</div></div>
    <div class="card kpi cyan"><div class="value" id="kpi-players">—</div><div class="label">Players Tracked</div></div>
  </div>

  <!-- Possession bar + Team stats -->
  <div class="row row-2">
    <div class="card">
      <div class="card-title">Ball Possession</div>
      <div class="poss-section">
        <div class="poss-labels"><span class="team-a">Team A (White)</span><span class="team-b">Team B (Red)</span></div>
        <div class="poss-bar-wrap">
          <div class="poss-bar-a" id="bar-a">—</div>
          <div class="poss-bar-b" id="bar-b">—</div>
        </div>
      </div>
      <div style="margin-top:20px">
        <div class="stat-row"><span class="stat-label">Total Distance</span><div class="stat-values"><span class="stat-val-a" id="s-dist-a">—</span><span class="stat-val-b" id="s-dist-b">—</span></div></div>
        <div class="stat-row"><span class="stat-label">Passes Made</span><div class="stat-values"><span class="stat-val-a" id="s-pass-a">—</span><span class="stat-val-b" id="s-pass-b">—</span></div></div>
        <div class="stat-row"><span class="stat-label">Interceptions</span><div class="stat-values"><span class="stat-val-a" id="s-int-a">—</span><span class="stat-val-b" id="s-int-b">—</span></div></div>
        <div class="stat-row"><span class="stat-label">Players Detected</span><div class="stat-values"><span class="stat-val-a" id="s-players-a">—</span><span class="stat-val-b" id="s-players-b">—</span></div></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Possession Split</div>
      <div class="chart-wrap"><canvas id="chart-poss"></canvas></div>
    </div>
  </div>

  <!-- Charts row -->
  <div class="row row-2">
    <div class="card">
      <div class="card-title">Top 10 Players — Distance Covered (m)</div>
      <div class="chart-wrap tall"><canvas id="chart-dist"></canvas></div>
    </div>
    <div class="card">
      <div class="card-title">Top 10 Players — Top Speed (m/s)</div>
      <div class="chart-wrap tall"><canvas id="chart-speed"></canvas></div>
    </div>
  </div>

  <!-- Heatmaps -->
  <p class="section-label">Positional Heatmaps</p>
  <div class="row row-2">
    <div class="card">
      <div class="card-title">Team A — Positional Heatmap</div>
      {'<img src="' + hm_a_b64 + '" class="heatmap-img" alt="Team A Heatmap" />' if hm_a_b64 else '<div class="no-hm">Run pipeline to generate heatmaps</div>'}
    </div>
    <div class="card">
      <div class="card-title">Team B — Positional Heatmap</div>
      {'<img src="' + hm_b_b64 + '" class="heatmap-img" alt="Team B Heatmap" />' if hm_b_b64 else '<div class="no-hm">Run pipeline to generate heatmaps</div>'}
    </div>
  </div>

  <!-- Player table -->
  <p class="section-label">Player Statistics</p>
  <div class="card">
    <div class="card-title">All Tracked Players</div>
    <div style="margin-bottom:12px; display:flex; gap:10px; flex-wrap:wrap">
      <button class="filter-btn active" onclick="filterTable('all')">All</button>
      <button class="filter-btn" onclick="filterTable('Team A')">Team A</button>
      <button class="filter-btn" onclick="filterTable('Team B')">Team B</button>
      <button class="filter-btn" onclick="filterTable('Goalkeeper')">Goalkeeper</button>
      <button class="filter-btn" onclick="filterTable('Referee')">Referee</button>
    </div>
    <div class="table-wrap">
      <table id="player-table">
        <thead><tr>
          <th>ID</th><th>Team</th><th>Distance (m)</th>
          <th>Top Speed (m/s)</th><th>Avg Speed (m/s)</th>
          <th>Possession (s)</th><th>Passes</th><th>Received</th><th>Intercepts</th>
        </tr></thead>
        <tbody id="player-tbody"></tbody>
      </table>
    </div>
  </div>

</main>

<!-- Embedded Chart.js -->
<script>
{chartjs}
</script>

<!-- Embedded Stats Data -->
<script>
const STATS = {stats_json};
</script>

<script>
(function() {{
  const data    = STATS;
  const players = data.per_player;
  const teams   = data.per_team;
  const poss    = data.possession_seconds;
  const total   = data.total_events;

  const totalPoss = (poss['Team A'] || 0) + (poss['Team B'] || 0);
  const pctA = totalPoss > 0 ? (poss['Team A'] / totalPoss * 100).toFixed(1) : '50.0';
  const pctB = totalPoss > 0 ? (poss['Team B'] / totalPoss * 100).toFixed(1) : '50.0';

  document.getElementById('kpi-poss-a').textContent = pctA + '%';
  document.getElementById('kpi-poss-b').textContent = pctB + '%';
  document.getElementById('kpi-events').textContent = total;

  const playerIds = Object.keys(players);
  const activePlayers = playerIds.filter(id => !['Referee','Goalkeeper'].includes(players[id].team));
  document.getElementById('kpi-players').textContent = activePlayers.length;
  document.getElementById('frame-badge').textContent = playerIds.length + ' Tracks Detected';

  const barA = document.getElementById('bar-a');
  barA.style.width = pctA + '%';
  barA.textContent = pctA + '%';
  document.getElementById('bar-b').textContent = pctB + '%';

  const ta = teams['Team A'] || {{}};
  const tb = teams['Team B'] || {{}};
  document.getElementById('s-dist-a').textContent = (ta.total_distance_m || 0).toFixed(0) + ' m';
  document.getElementById('s-dist-b').textContent = (tb.total_distance_m || 0).toFixed(0) + ' m';
  document.getElementById('s-pass-a').textContent  = ta.passes_made || 0;
  document.getElementById('s-pass-b').textContent  = tb.passes_made || 0;
  document.getElementById('s-int-a').textContent   = ta.interceptions || 0;
  document.getElementById('s-int-b').textContent   = tb.interceptions || 0;
  document.getElementById('s-players-a').textContent = ta.players || 0;
  document.getElementById('s-players-b').textContent = tb.players || 0;

  // Possession doughnut
  new Chart(document.getElementById('chart-poss'), {{
    type: 'doughnut',
    data: {{
      labels: ['Team A (White)', 'Team B (Red)'],
      datasets: [{{ data: [parseFloat(pctA), parseFloat(pctB)],
        backgroundColor: ['rgba(232,232,232,0.85)', 'rgba(224,48,80,0.85)'],
        borderColor: ['#141c2b','#141c2b'], borderWidth: 3, hoverOffset: 6 }}]
    }},
    options: {{ responsive: true, maintainAspectRatio: false, cutout: '68%',
      plugins: {{
        legend: {{ position: 'bottom', labels: {{ color: '#7a8fa8', padding: 16, font: {{ size: 12 }} }} }},
        tooltip: {{ callbacks: {{ label: ctx => ` ${{ctx.label}}: ${{ctx.parsed}}%` }} }}
      }}
    }}
  }});

  function teamColor(t) {{
    return t === 'Team A' ? 'rgba(232,232,232,0.85)'
         : t === 'Team B' ? 'rgba(224,48,80,0.85)'
         : t === 'Goalkeeper' ? 'rgba(0,212,255,0.85)'
         : 'rgba(34,197,94,0.85)';
  }}

  // Distance chart
  const sorted = playerIds.filter(id => players[id].distance_m > 5)
    .sort((a,b) => players[b].distance_m - players[a].distance_m).slice(0,10);
  new Chart(document.getElementById('chart-dist'), {{
    type: 'bar',
    data: {{ labels: sorted.map(id => 'P' + id),
      datasets: [{{ label: 'Distance (m)', data: sorted.map(id => players[id].distance_m),
        backgroundColor: sorted.map(id => teamColor(players[id].team)),
        borderRadius: 6, borderSkipped: false }}] }},
    options: {{ responsive: true, maintainAspectRatio: false, indexAxis: 'y',
      plugins: {{ legend: {{ display: false }},
        tooltip: {{ callbacks: {{ label: ctx => ` ${{ctx.parsed.x.toFixed(1)}} m` }} }} }},
      scales: {{
        x: {{ grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#7a8fa8' }} }},
        y: {{ grid: {{ display: false }}, ticks: {{ color: '#a0b0c0', font: {{ size: 11 }} }} }}
      }}
    }}
  }});

  // Speed chart
  const speedSorted = playerIds.filter(id => players[id].top_speed_ms > 1)
    .sort((a,b) => players[b].top_speed_ms - players[a].top_speed_ms).slice(0,10);
  new Chart(document.getElementById('chart-speed'), {{
    type: 'bar',
    data: {{ labels: speedSorted.map(id => 'P' + id),
      datasets: [{{ label: 'Top Speed (m/s)', data: speedSorted.map(id => players[id].top_speed_ms),
        backgroundColor: speedSorted.map(id => teamColor(players[id].team)),
        borderRadius: 6, borderSkipped: false }}] }},
    options: {{ responsive: true, maintainAspectRatio: false, indexAxis: 'y',
      plugins: {{ legend: {{ display: false }},
        tooltip: {{ callbacks: {{ label: ctx => ` ${{ctx.parsed.x.toFixed(2)}} m/s` }} }} }},
      scales: {{
        x: {{ grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#7a8fa8' }} }},
        y: {{ grid: {{ display: false }}, ticks: {{ color: '#a0b0c0', font: {{ size: 11 }} }} }}
      }}
    }}
  }});

  // Player table
  window._playerData = players;
  filterTable('all');
}})();

function teamChip(t) {{
  if (t === 'Team A') return '<span class="team-chip chip-a">Team A</span>';
  if (t === 'Team B') return '<span class="team-chip chip-b">Team B</span>';
  if (t === 'Goalkeeper') return '<span class="team-chip chip-gk">GK</span>';
  return '<span class="team-chip chip-ref">Ref</span>';
}}

function filterTable(team) {{
  const players = window._playerData;
  const ids = Object.keys(players)
    .filter(id => team === 'all' || players[id].team === team)
    .sort((a,b) => players[b].distance_m - players[a].distance_m);
  document.getElementById('player-tbody').innerHTML = ids.map(id => {{
    const p = players[id];
    return `<tr>
      <td><b>#${{id}}</b></td><td>${{teamChip(p.team)}}</td>
      <td>${{p.distance_m.toFixed(1)}}</td><td>${{p.top_speed_ms.toFixed(2)}}</td>
      <td>${{p.avg_speed_ms.toFixed(2)}}</td><td>${{p.possession_s.toFixed(1)}}</td>
      <td>${{p.passes_made}}</td><td>${{p.passes_received}}</td><td>${{p.interceptions}}</td>
    </tr>`;
  }}).join('');
  document.querySelectorAll('.filter-btn').forEach(b => {{
    b.classList.toggle('active', b.textContent === (team === 'all' ? 'All' : team));
  }});
}}
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
