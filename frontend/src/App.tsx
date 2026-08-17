import React, { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";

type AnyObj = Record<string, any>;

const API_BASE = "http://localhost:8000";

function numericKeys(obj?: AnyObj): string[] {
  if (!obj || typeof obj !== "object") return [];
  return Object.keys(obj).sort((a, b) => Number(a) - Number(b));
}

function clamp01(v: number) {
  return Math.max(0, Math.min(1, v));
}

function toArray(v: any): any[] {
  return Array.isArray(v) ? v : [];
}

function meanAbs1D(arr: number[]): number {
  if (!arr.length) return 0;
  let s = 0;
  for (const v of arr) s += Math.abs(v);
  return s / arr.length;
}

function flattenLastDimMeanAbs(matrix: number[][]): number[] {
  return matrix.map((row) => meanAbs1D(row));
}

function latestEpochFromState(state: AnyObj | null): string {
  if (!state) return "1";
  const buckets = [
    state.activations,
    state.diffusion,
    state.gradients,
    state.attention,
    state.sequence,
    state.vision,
  ];
  for (const b of buckets) {
    const ks = numericKeys(b);
    if (ks.length) return ks[ks.length - 1];
  }
  const current = state.meta?.current_epoch;
  return current ? String(current) : "1";
}

function getEpochOptions(state: AnyObj | null): string[] {
  if (!state) return ["1"];
  const merged = new Set<string>();
  [
    state.activations,
    state.diffusion,
    state.gradients,
    state.attention,
    state.sequence,
    state.vision,
  ].forEach((bucket) => numericKeys(bucket).forEach((k) => merged.add(k)));
  const out = Array.from(merged).sort((a, b) => Number(a) - Number(b));
  if (!out.length && state.meta?.current_epoch) out.push(String(state.meta.current_epoch));
  return out.length ? out : ["1"];
}

function getMetrics(state: AnyObj | null): AnyObj {
  return state?.history?.metrics || {};
}

function getActivationEpoch(state: AnyObj | null, epoch: string): AnyObj {
  return state?.activations?.[epoch] || {};
}

function getAttentionEpoch(state: AnyObj | null, epoch: string): AnyObj {
  return state?.attention?.[epoch] || {};
}

function getDiffusionEpoch(state: AnyObj | null, epoch: string): AnyObj {
  return state?.diffusion?.[epoch] || {};
}

function getGradientEpoch(state: AnyObj | null, epoch: string): AnyObj {
  return state?.gradients?.[epoch] || {};
}

function getCandidateLayers(state: AnyObj | null, epoch: string): string[] {
  const arch = state?.meta?.arch_type || "generic";

  if (arch === "transformer") {
    return Object.keys(getAttentionEpoch(state, epoch));
  }

  if (arch === "cnn") {
    const epochActs = getActivationEpoch(state, epoch);
    return Object.keys(epochActs).filter((lname) => {
      const vals = epochActs?.[lname]?.values;
      return lname.toLowerCase().includes("conv") && Array.isArray(vals) && Array.isArray(vals[0]) && Array.isArray(vals[0][0]);
    });
  }

  if (arch === "rnn" || arch === "hybrid" || arch === "sequence_cnn") {
    const epochActs = getActivationEpoch(state, epoch);
    return Object.keys(epochActs).filter((lname) => {
      const vals = epochActs?.[lname]?.values;
      return Array.isArray(vals) && Array.isArray(vals[0]) && Array.isArray(vals[0][0]);
    });
  }

  return Object.keys(getActivationEpoch(state, epoch));
}

function pointsFromSeries(values: number[], width: number, height: number, pad = 24): string {
  if (!values.length) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = Math.max(max - min, 1e-8);

  return values
    .map((v, i) => {
      const x = pad + (i / Math.max(values.length - 1, 1)) * (width - pad * 2);
      const y = height - pad - ((v - min) / range) * (height - pad * 2);
      return `${x},${y}`;
    })
    .join(" ");
}

function multiSeriesBounds(series: { values: number[] }[]) {
  let min = Infinity;
  let max = -Infinity;
  series.forEach((s) => {
    s.values.forEach((v) => {
      if (v < min) min = v;
      if (v > max) max = v;
    });
  });
  if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: 0, max: 1 };
  if (Math.abs(max - min) < 1e-8) return { min, max: min + 1 };
  return { min, max };
}

function project3D(rows: number[][]): { x: number; y: number; z: number }[] {
  if (!rows.length) return [];
  const dims = rows[0]?.length || 0;

  return rows.map((r, i) => ({
    x: Number(r[0] ?? i),
    y: Number(r[Math.min(1, Math.max(0, dims - 1))] ?? 0),
    z: Number(r[Math.min(2, Math.max(0, dims - 1))] ?? 0),
  }));
}

function LatentGalaxy3D(props: { rows: number[][]; labels?: string[] }) {
  const rows = props.rows || [];
  if (!rows.length) return <div style={styles.emptyNotice}>No latent points</div>;

  const pts = project3D(rows);

  const x = pts.map((p) => Number(p.x));
  const y = pts.map((p) => Number(p.y));
  const z = pts.map((p) => Number(p.z));
  const labels = props.labels || pts.map((_, i) => `S${i}`);

  return (
    <div style={{ width: "100%", height: 420 }}>
      <Plot
        data={[
          {
            type: "scatter3d",
            mode: "markers+text",
            x,
            y,
            z,
            text: labels,
            textposition: "top center",
            hovertemplate:
              "<b>%{text}</b><br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            marker: {
              size: 6,
              color: z,
              colorscale: [
                [0, "#22d3ee"],
                [0.5, "#818cf8"],
                [1, "#d946ef"],
              ],
              opacity: 0.92,
              line: {
                color: "rgba(255,255,255,0.35)",
                width: 0.6,
              },
            },
          } as any,
        ]}
        layout={{
                  autosize: true,
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  margin: { l: 0, r: 0, t: 0, b: 0 },
                  scene: {
                    bgcolor: "#050816",
                    xaxis: {
                      title: { text: "x" },
                      showbackground: true,
                      backgroundcolor: "rgba(8,12,24,1)",
                      gridcolor: "rgba(255,255,255,0.16)",
                      zerolinecolor: "rgba(255,255,255,0.22)",
                      linecolor: "rgba(255,255,255,0.28)",
                      color: "#a1a1aa",
                      showspikes: false,
                    },
                    yaxis: {
                      title: { text: "y" },
                      showbackground: true,
                      backgroundcolor: "rgba(8,12,24,1)",
                      gridcolor: "rgba(255,255,255,0.16)",
                      zerolinecolor: "rgba(255,255,255,0.22)",
                      linecolor: "rgba(255,255,255,0.28)",
                      color: "#a1a1aa",
                      showspikes: false,
                    },
                    zaxis: {
                      title: { text: "z" },
                      showbackground: true,
                      backgroundcolor: "rgba(8,12,24,1)",
                      gridcolor: "rgba(255,255,255,0.16)",
                      zerolinecolor: "rgba(255,255,255,0.22)",
                      linecolor: "rgba(255,255,255,0.28)",
                      color: "#a1a1aa",
                      showspikes: false,
                    },
                    camera: {
                      eye: { x: 1.45, y: 1.35, z: 1.15 },
                    },
                  },
                  showlegend: false,
                }}
        config={{
          responsive: true,
          displayModeBar: true,
          scrollZoom: true,
        }}
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}

function frameToDataUrl(frame: number[][]): string {
  const h = frame.length;
  const w = frame[0]?.length || 0;
  if (!h || !w) return "";

  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";

  let min = Infinity;
  let max = -Infinity;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = Number(frame[y][x] || 0);
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  const range = Math.max(max - min, 1e-8);
  const image = ctx.createImageData(w, h);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;
      const norm = Math.round(clamp01((Number(frame[y][x] || 0) - min) / range) * 255);
      image.data[idx] = norm;
      image.data[idx + 1] = norm;
      image.data[idx + 2] = norm;
      image.data[idx + 3] = 255;
    }
  }

  ctx.putImageData(image, 0, 0);
  return canvas.toDataURL();
}

function Card(props: { title: string; subtitle?: string; right?: React.ReactNode; children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div style={{ ...styles.card, ...props.style }}>
      <div style={styles.cardHeader}>
        <div>
          <div style={styles.cardTitle}>{props.title}</div>
          {props.subtitle ? <div style={styles.cardSubtitle}>{props.subtitle}</div> : null}
        </div>
        {props.right}
      </div>
      <div>{props.children}</div>
    </div>
  );
}

function LiveBadge({ training }: { training: boolean }) {
  return (
    <div style={{ ...styles.liveBadge, color: training ? "#4ade80" : "#f59e0b" }}>
      <span
        style={{
          ...styles.liveDot,
          background: training ? "#22c55e" : "#f59e0b",
          boxShadow: training ? "0 0 18px rgba(34,197,94,0.8)" : "0 0 18px rgba(245,158,11,0.6)",
        }}
      />
      {training ? "LIVE TRAINING" : "IDLE"}
    </div>
  );
}

function SelectControl(props: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  return (
    <label style={styles.controlWrap}>
      <span style={styles.controlLabel}>{props.label}</span>
      <select style={styles.select} value={props.value} onChange={(e) => props.onChange(e.target.value)}>
        {props.options.map((o) => (
          <option key={o} value={o}>
            {props.label === "Epoch" ? `Epoch ${o}` : o}
          </option>
        ))}
      </select>
    </label>
  );
}

function MetricTile(props: { label: string; value: string; delta?: string }) {
  return (
    <div style={styles.metricTile}>
      <div style={styles.metricLabel}>{props.label}</div>
      <div style={styles.metricValue}>{props.value}</div>
      {props.delta ? <div style={styles.metricDelta}>{props.delta}</div> : null}
    </div>
  );
}

function MultiLineChart(props: { series: { name: string; values: number[]; color: string }[]; height?: number }) {
  const width = 760;
  const height = props.height || 320;
  const bounds = multiSeriesBounds(props.series);
  const yTicks = 4;
  const xLen = Math.max(...props.series.map((s) => s.values.length), 1);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height }}>
      <defs>
        <linearGradient id="gridGlow" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="rgba(34,211,238,0.14)" />
          <stop offset="100%" stopColor="rgba(217,70,239,0.08)" />
        </linearGradient>
      </defs>

      <rect x="0" y="0" width={width} height={height} rx="18" fill="transparent" />

      {Array.from({ length: yTicks + 1 }).map((_, i) => {
        const y = 24 + (i / yTicks) * (height - 48);
        const val = bounds.max - (i / yTicks) * (bounds.max - bounds.min);
        return (
          <g key={i}>
            <line x1="24" y1={y} x2={width - 18} y2={y} stroke="rgba(255,255,255,0.09)" strokeWidth="1" />
            <text x="0" y={y + 4} fill="#9ca3af" fontSize="12">
              {val.toFixed(2)}
            </text>
          </g>
        );
      })}

      {Array.from({ length: xLen }).map((_, i) => {
        const x = 24 + (i / Math.max(xLen - 1, 1)) * (width - 48);
        return (
          <text key={i} x={x - 5} y={height - 6} fill="#9ca3af" fontSize="12">
            {i + 1}
          </text>
        );
      })}

      {props.series.map((s) => {
        const pts = s.values.map((v, i) => {
          const x = 24 + (i / Math.max(s.values.length - 1, 1)) * (width - 48);
          const y = height - 24 - ((v - bounds.min) / (bounds.max - bounds.min)) * (height - 48);
          return { x, y };
        });

        const poly = pts.map((p) => `${p.x},${p.y}`).join(" ");
        return (
          <g key={s.name}>
            <polyline fill="none" stroke={s.color} strokeWidth="3" points={poly} />
            {pts.map((p, i) => (
              <circle key={i} cx={p.x} cy={p.y} r="3" fill={s.color} />
            ))}
          </g>
        );
      })}

      <g transform={`translate(${width - 150},20)`}>
        {props.series.map((s, i) => (
          <g key={s.name} transform={`translate(0, ${i * 18})`}>
            <line x1="0" y1="7" x2="18" y2="7" stroke={s.color} strokeWidth="3" />
            <text x="24" y="11" fill="#d4d4d8" fontSize="12">
              {s.name}
            </text>
          </g>
        ))}
      </g>
    </svg>
  );
}

function Heatmap(props: { matrix: number[][]; height?: number; color?: "viridis" | "gray" | "cyan" }) {
  const mat = props.matrix;
  const rows = mat.length;
  const cols = mat[0]?.length || 0;
  const height = props.height || 320;
  if (!rows || !cols) return <div style={styles.emptyNotice}>No values</div>;

  let min = Infinity;
  let max = -Infinity;
  for (const row of mat) for (const v of row) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = Math.max(max - min, 1e-8);

  function colorFor(v: number) {
    const t = clamp01((v - min) / range);
    if (props.color === "gray") {
      const c = Math.round(t * 255);
      return `rgb(${c},${c},${c})`;
    }
    if (props.color === "cyan") {
      const a = Math.round(40 + t * 180);
      return `rgba(34,211,238,${a / 255})`;
    }
    const r = Math.round(68 + t * 120);
    const g = Math.round(20 + t * 170);
    const b = Math.round(110 + (1 - t) * 120);
    return `rgb(${r},${g},${b})`;
  }

  return (
    <div
      style={{
        ...styles.heatmapGrid,
        gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
        height,
      }}
    >
      {mat.flatMap((row, ri) =>
        row.map((v, ci) => (
          <div
            key={`${ri}-${ci}`}
            title={`${ri}, ${ci}: ${Number(v).toFixed(4)}`}
            style={{
              background: colorFor(Number(v)),
              aspectRatio: "1 / 1",
            }}
          />
        ))
      )}
    </div>
  );
}

function Scatter2D(props: { points: { x: number; y: number }[]; labels?: string[]; color?: string; height?: number; connect?: boolean }) {
  const width = 500;
  const height = props.height || 320;
  if (!props.points.length) return <div style={styles.emptyNotice}>No points</div>;

  const xs = props.points.map((p) => p.x);
  const ys = props.points.map((p) => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const rangeX = Math.max(maxX - minX, 1e-8);
  const rangeY = Math.max(maxY - minY, 1e-8);

  const pts = props.points.map((p) => ({
    x: 24 + ((p.x - minX) / rangeX) * (width - 48),
    y: height - 24 - ((p.y - minY) / rangeY) * (height - 48),
  }));

  return (
    <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height }}>
      <rect x="0" y="0" width={width} height={height} rx="16" fill="transparent" />
      <line x1="24" y1={height - 24} x2={width - 12} y2={height - 24} stroke="rgba(255,255,255,0.12)" />
      <line x1="24" y1="12" x2="24" y2={height - 24} stroke="rgba(255,255,255,0.12)" />
      {props.connect && <polyline fill="none" stroke="rgba(34,211,238,0.8)" strokeWidth="2.5" points={pts.map((p) => `${p.x},${p.y}`).join(" ")} />}
      {pts.map((p, i) => (
        <g key={i}>
          <circle cx={p.x} cy={p.y} r="5" fill={props.color || "#22d3ee"} />
          {props.labels?.[i] ? (
            <text x={p.x + 8} y={p.y - 8} fill="#d4d4d8" fontSize="11">
              {props.labels[i]}
            </text>
          ) : null}
        </g>
      ))}
    </svg>
  );
}

function BarChart(props: { items: { label: string; value: number; color?: string }[]; height?: number }) {
  const width = 500;
  const height = props.height || 320;
  if (!props.items.length) return <div style={styles.emptyNotice}>No bars</div>;
  const max = Math.max(...props.items.map((i) => i.value), 1e-8);
  const barW = Math.max(18, (width - 48) / props.items.length - 10);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} style={{ width: "100%", height }}>
      <line x1="24" y1={height - 24} x2={width - 12} y2={height - 24} stroke="rgba(255,255,255,0.12)" />
      {props.items.map((it, i) => {
        const x = 36 + i * ((width - 60) / props.items.length);
        const h = ((height - 60) * it.value) / max;
        const y = height - 24 - h;
        return (
          <g key={it.label}>
            <rect x={x} y={y} width={barW} height={h} rx="8" fill={it.color || "#22d3ee"} />
            <text x={x + barW / 2} y={height - 6} fill="#a1a1aa" fontSize="11" textAnchor="middle">
              {it.label}
            </text>
            <text x={x + barW / 2} y={y - 6} fill="#e4e4e7" fontSize="10" textAnchor="middle">
              {it.value.toFixed(3)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function LogsTerminal({ logs }: { logs: { time?: string; msg?: string }[] }) {
  return (
    <div style={styles.terminal}>
      {logs.length ? logs.map((l, i) => (
        <div key={i} style={styles.logLine}>
          <span style={styles.logTime}>[{(l.time || "").slice(11, 19)}]</span> {l.msg}
        </div>
      )) : <div style={styles.emptyNotice}>No logs yet</div>}
    </div>
  );
}

export default function App() {
  const [state, setState] = useState<AnyObj | null>(null);
  const [logs, setLogs] = useState<any[]>([]);
  const [epoch, setEpoch] = useState("1");
  const [layer, setLayer] = useState("");
  const [trainEpochs, setTrainEpochs] = useState("3");
  const [sampleIdx, setSampleIdx] = useState("0");
  const [headIdx, setHeadIdx] = useState("0");
  const [queryIdx, setQueryIdx] = useState("0");
  const [seqView, setSeqView] = useState("Temporal Importance");
  const [diffusionView, setDiffusionView] = useState("Film Strip");

  useEffect(() => {
    let mounted = true;

    async function poll() {
      try {
        const [stateRes, logsRes] = await Promise.all([
          fetch(`${API_BASE}/api/state`),
          fetch(`${API_BASE}/api/logs`),
        ]);
        const nextState = await stateRes.json();
        const nextLogs = await logsRes.json();

        if (!mounted) return;

        setState(nextState || {});
        setLogs(Array.isArray(nextLogs) ? nextLogs : []);

        const latestEpoch = latestEpochFromState(nextState || {});
        setEpoch((prev) => {
          const opts = getEpochOptions(nextState || {});
          if (nextState?.status?.training) {
            return latestEpoch;
          }
          return opts.includes(prev) ? prev : latestEpoch;
        });
      } catch (e) {
        console.error(e);
      }
    }

    poll();
    const id = window.setInterval(poll, 800);
    return () => {
      mounted = false;
      window.clearInterval(id);
    };
  }, []);

  const epochOptions = useMemo(() => getEpochOptions(state), [state]);

  useEffect(() => {
    if (!state) return;
    const layers = getCandidateLayers(state, epoch);
    setLayer((prev) => (layers.includes(prev) ? prev : (layers[0] || "")));
  }, [state, epoch]);

  const training = !!state?.status?.training;
  const arch = state?.meta?.arch_type || "generic";
  const metrics = getMetrics(state);
  const loss = toArray(metrics.loss);
  const accuracy = toArray(metrics.accuracy);
  const valLoss = toArray(metrics.val_loss);
  const valAccuracy = toArray(metrics.val_accuracy);
  const currentEpoch = String(state?.meta?.current_epoch || epoch);

  const currentActivationEpoch = getActivationEpoch(state, epoch);
  const currentAttentionEpoch = getAttentionEpoch(state, epoch);
  const currentDiffusionEpoch = getDiffusionEpoch(state, epoch);
  const currentGradientEpoch = getGradientEpoch(state, epoch);

  const summaryTiles = useMemo(() => {
    const lossNow = loss.length ? Number(loss[loss.length - 1]).toFixed(4) : "--";
    const accNow = accuracy.length ? `${(Number(accuracy[accuracy.length - 1]) * 100).toFixed(1)}%` : "--";
    const lossDelta = state?.history?.loss_delta?.length
      ? `${Number(state.history.loss_delta[state.history.loss_delta.length - 1]).toFixed(4)} vs prev`
      : undefined;
    const energies = toArray(currentDiffusionEpoch?.energies || []);
    const denoise = energies.length ? `${(((energies[0] - energies[energies.length - 1]) / Math.max(energies[0], 1e-8)) * 100).toFixed(1)}%` : "--";
    return [
      { label: "Architecture", value: arch },
      { label: "Current Epoch", value: `${currentEpoch}/${state?.meta?.total_epochs || "--"}` },
      { label: "Loss", value: lossNow, delta: lossDelta },
      { label: "Accuracy", value: accNow, delta: denoise !== "--" ? `Denoise Gain ${denoise}` : undefined },
    ];
  }, [arch, currentEpoch, state, loss, accuracy, currentDiffusionEpoch]);

  const q1Series = useMemo(() => {
    const out: { name: string; values: number[]; color: string }[] = [];
    if (loss.length) out.push({ name: "loss", values: loss.map(Number), color: "#22d3ee" });
    if (accuracy.length) out.push({ name: "accuracy", values: accuracy.map(Number), color: "#a78bfa" });
    if (valLoss.length) out.push({ name: "val_loss", values: valLoss.map(Number), color: "#f97316" });
    if (valAccuracy.length) out.push({ name: "val_accuracy", values: valAccuracy.map(Number), color: "#34d399" });
    return out;
  }, [loss, accuracy, valLoss, valAccuracy]);

  const diffusionFrames = useMemo(() => {
    const frames = toArray(currentDiffusionEpoch?.frames || []);
    return frames.slice(0, 8).map((f) => {
      try {
        return frameToDataUrl(f as number[][]);
      } catch {
        return "";
      }
    });
  }, [currentDiffusionEpoch]);

  const selectedAttention = layer ? currentAttentionEpoch?.[layer] : null;
  const attentionScores = Array.isArray(selectedAttention?.scores) ? selectedAttention.scores : [];
  const attentionHeads = attentionScores?.[0]?.length || 0;
  const attentionBatch = attentionScores?.length || 0;
  const attentionSeq = attentionScores?.[0]?.[0]?.length || 0;
  const attnSample = Math.min(Number(sampleIdx || "0"), Math.max(attentionBatch - 1, 0));
  const attnHead = Math.min(Number(headIdx || "0"), Math.max(attentionHeads - 1, 0));
  const attnQuery = Math.min(Number(queryIdx || "0"), Math.max(attentionSeq - 1, 0));
  const transformerMatrix = attentionScores?.[attnSample]?.[attnHead] || [];

  const cnnSample = useMemo(() => {
    const vals = currentActivationEpoch?.[layer]?.values;
    if (!Array.isArray(vals) || !Array.isArray(vals[0])) return null;
    const idx = Math.min(Number(sampleIdx || "0"), Math.max(vals.length - 1, 0));
    return vals[idx];
  }, [currentActivationEpoch, layer, sampleIdx]);

  const cnnFeatureMaps = useMemo(() => {
    if (!Array.isArray(cnnSample) || !Array.isArray(cnnSample[0]) || !Array.isArray(cnnSample[0][0])) return [];
    const H = cnnSample.length;
    const W = cnnSample[0].length;
    const C = cnnSample[0][0].length;
    const maps: number[][][] = [];
    for (let c = 0; c < Math.min(C, 16); c++) {
      const map: number[][] = [];
      for (let i = 0; i < H; i++) {
        const row: number[] = [];
        for (let j = 0; j < W; j++) row.push(Number(cnnSample[i][j][c] || 0));
        map.push(row);
      }
      maps.push(map);
    }
    return maps;
  }, [cnnSample]);

  const seqSample = useMemo(() => {
    const vals = currentActivationEpoch?.[layer]?.values;
    if (!Array.isArray(vals) || !Array.isArray(vals[0]) || !Array.isArray(vals[0][0])) return [];
    const idx = Math.min(Number(sampleIdx || "0"), Math.max(vals.length - 1, 0));
    return vals[idx] as number[][];
  }, [currentActivationEpoch, layer, sampleIdx]);

  const seqImportance = useMemo(() => flattenLastDimMeanAbs(seqSample), [seqSample]);

  const vitalityBars = useMemo(() => {
    if (arch === "diffusion") {
      const energies = toArray(currentDiffusionEpoch?.energies || []);
      return energies.map((v, i) => ({ label: `t${i}`, value: Number(v), color: "#22d3ee" }));
    }
    return Object.entries(currentGradientEpoch || {}).map(([lname, vars]) => {
      const vals = Object.values(vars as AnyObj).map((s: any) => Number(s?.l2 || 0));
      const avg = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
      return {
        label: lname,
        value: avg,
        color: avg > 0.1 ? "#ef4444" : avg < 1e-4 ? "#f59e0b" : "#22d3ee",
      };
    });
  }, [arch, currentGradientEpoch, currentDiffusionEpoch]);

  const layerOptions = getCandidateLayers(state, epoch);
  const sampleOptions = useMemo(() => {
    if (arch === "transformer") {
      return Array.from({ length: Math.max(attentionBatch, 1) }, (_, i) => String(i));
    }
    const vals = currentActivationEpoch?.[layer]?.values;
    if (Array.isArray(vals)) return Array.from({ length: Math.max(vals.length, 1) }, (_, i) => String(i));
    return ["0"];
  }, [arch, attentionBatch, currentActivationEpoch, layer]);

  return (
    <div style={styles.page}>
      <div style={styles.glowA} />
      <div style={styles.glowB} />
      <div style={styles.container}>
        <div style={styles.headerRow}>
          <div>
            <div style={styles.brand}>OmniVision</div>
            <div style={styles.brandSub}>Advanced deep learning observability</div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <LiveBadge training={training} />

          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <LiveBadge training={training} />

          <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 12, color: "#a1a1aa", fontWeight: 700 }}>Train Epochs</span>
            <select
              value={trainEpochs}
              onChange={(e) => setTrainEpochs(e.target.value)}
              style={{
                background: "rgba(10,14,28,0.92)",
                color: "#f4f4f5",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 12,
                padding: "10px 12px",
                outline: "none",
              }}
            >
              {["1", "2", "3", "5", "10", "20", "50"].map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>

          <button
            onClick={() =>
              fetch("http://localhost:8000/train", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ epochs: Number(trainEpochs) }),
              })
            }
            style={{
              background: "linear-gradient(90deg, #22c55e, #4ade80)",
              border: "none",
              borderRadius: 12,
              padding: "10px 16px",
              fontWeight: 700,
              cursor: "pointer",
              color: "#022c22",
              boxShadow: "0 0 20px rgba(34,197,94,0.4)",
            }}
          >
            🚀 Start
          </button>
        </div>
        </div>
        </div>

        <div style={styles.controlsRow}>
          <SelectControl label="Epoch" value={epoch} options={epochOptions} onChange={setEpoch} />
          {layerOptions.length ? (
            <SelectControl label="Layer" value={layer} options={layerOptions} onChange={setLayer} />
          ) : null}
          {sampleOptions.length ? (
            <SelectControl label="Sample" value={sampleIdx} options={sampleOptions} onChange={setSampleIdx} />
          ) : null}
          {arch === "transformer" && attentionHeads > 0 ? (
            <SelectControl
              label="Head"
              value={headIdx}
              options={Array.from({ length: attentionHeads }, (_, i) => String(i))}
              onChange={setHeadIdx}
            />
          ) : null}
          {arch === "transformer" && attentionSeq > 0 ? (
            <SelectControl
              label="Query"
              value={queryIdx}
              options={Array.from({ length: attentionSeq }, (_, i) => String(i))}
              onChange={setQueryIdx}
            />
          ) : null}
          {(arch === "rnn" || arch === "hybrid" || arch === "sequence_cnn") ? (
            <SelectControl
              label="Sequence View"
              value={seqView}
              options={["Temporal Importance", "Heatmap", "Trajectory", "Top Neurons"]}
              onChange={setSeqView}
            />
          ) : null}
          {arch === "diffusion" ? (
            <SelectControl
              label="Diffusion View"
              value={diffusionView}
              options={["Film Strip", "Noise Curve", "Latent Trajectory"]}
              onChange={setDiffusionView}
            />
          ) : null}
        </div>

        <div style={styles.metricGrid}>
          {summaryTiles.map((t) => (
            <MetricTile key={t.label} label={t.label} value={t.value} delta={t.delta} />
          ))}
        </div>

        <div style={styles.grid}>
          <Card
            title="Performance Rhythm"
            subtitle="Readable curves for training, validation, and replay."
            style={{ gridColumn: "span 7" }}
          >
            {q1Series.length ? <MultiLineChart series={q1Series} /> : <div style={styles.emptyNotice}>No metrics yet</div>}
          </Card>

          <Card
            title={arch === "diffusion" ? "Diffusion Filmstrip" : "Latent Galaxy"}
            subtitle={
              arch === "diffusion"
                ? "Glanceable denoising frames for the latest epoch."
                : "Compact latent projection for the selected epoch."
            }
            style={{ gridColumn: "span 5" }}
          >
            {arch === "diffusion" ? (
              diffusionView === "Film Strip" ? (
                <>
                  <div style={styles.filmGrid}>
                    {diffusionFrames.map((src, i) => (
                      <div key={i} style={styles.frameWrap}>
                        {src ? <img src={src} alt={`frame-${i}`} style={styles.frameImg} /> : <div style={styles.framePlaceholder} />}
                        <div style={styles.frameTag}>T{i}</div>
                      </div>
                    ))}
                  </div>
                  <div style={{ marginTop: 16 }}>
                    <MultiLineChart
                      height={180}
                      series={[
                        {
                          name: "noise_energy",
                          values: toArray(currentDiffusionEpoch?.energies || []).map(Number),
                          color: "#22d3ee",
                        },
                      ]}
                    />
                  </div>
                </>
              ) : diffusionView === "Noise Curve" ? (
                <MultiLineChart
                  series={[
                    {
                      name: "noise_energy",
                      values: toArray(currentDiffusionEpoch?.energies || []).map(Number),
                      color: "#22d3ee",
                    },
                  ]}
                />
              ) : (
                <Scatter2D
                  connect
                  labels={toArray(currentDiffusionEpoch?.embeddings || []).map((_: any, i: number) => `t${i}`)}
                  points={project3D(toArray(currentDiffusionEpoch?.embeddings || []).map((r: any) => toArray(r).map(Number)))}
                  color="#22d3ee"
                />
              )
            ) : (
              (() => {
                const epochActs = getActivationEpoch(state, epoch);
                const keys = Object.keys(epochActs || {});
                const candidate = keys.find((k) => {
                                  const v = epochActs[k]?.values;
                                  return epochActs[k]?.sampled === false &&
                                    Array.isArray(v) &&
                                    Array.isArray(v[0]) &&
                                    !Array.isArray(v[0]?.[0]);
                                }) ||
                                keys.find((k) => epochActs[k]?.sampled === false) ||
                                keys[0];

                const vals = toArray(epochActs?.[candidate]?.values || []);
                let rows: number[][] = [];

                if (Array.isArray(vals) && vals.length > 0) {
                  if (Array.isArray(vals[0]) && Array.isArray((vals[0] as any[])[0])) {
                    // shape like [samples, ..., features]
                    rows = (vals as any[]).map((sample) => {
                      const flat = (sample as any[]).flat(Infinity).map(Number);
                      return flat.slice(0, 32);
                    });
                  } else if (Array.isArray(vals[0])) {
                    // shape like [samples, features]
                    rows = (vals as any[]).map((sample) => (sample as any[]).map(Number).slice(0, 32));
                  } else {
                    // shape like [features] -> single sample
                    rows = [(vals as any[]).map(Number).slice(0, 32)];
                  }
                }
                return rows.length ? (
                  <LatentGalaxy3D rows={rows} labels={rows.map((_, i) => `S${i}`)} />
                ) : (
                  <div style={styles.emptyNotice}>No activations yet</div>
                );
              })()
            )}
          </Card>

          <Card
            title={arch === "transformer" ? "Attention Grid" : arch === "cnn" ? "Feature Grid" : arch === "diffusion" ? "Diffusion Observatory" : "Sequence Grid"}
            subtitle={
              arch === "transformer"
                ? "Head-wise token relationships for the selected layer."
                : arch === "cnn"
                ? "Layer-selectable feature maps."
                : arch === "diffusion"
                ? "Primary diffusion view with premium styling."
                : "Interpretable temporal behavior for sequence models."
            }
            style={{ gridColumn: "span 7" }}
          >
            {arch === "transformer" ? (
              transformerMatrix.length ? (
                <div>
                  <Heatmap matrix={transformerMatrix} color="viridis" />
                  <div style={styles.smallMeta}>Layer: {layer} • Head: {headIdx} • Query token: T{queryIdx}</div>
                  <div style={{ marginTop: 14 }}>
                    <BarChart
                      items={toArray(transformerMatrix[attnQuery] || [])
                        .map((v: any, i: number) => ({ label: `T${i}`, value: Number(v), color: i === attnQuery ? "#f59e0b" : "#22d3ee" }))
                        .sort((a, b) => b.value - a.value)
                        .slice(0, Math.min(8, attentionSeq))}
                      height={220}
                    />
                  </div>
                </div>
              ) : (
                <div style={styles.emptyNotice}>No attention payloads available</div>
              )
            ) : arch === "cnn" ? (
              cnnFeatureMaps.length ? (
                <div style={styles.featureMapGrid}>
                  {cnnFeatureMaps.map((map, i) => (
                    <div key={i} style={styles.featureMapCard}>
                      <div style={styles.featureMapTitle}>F{i}</div>
                      <Heatmap matrix={map} height={140} color="gray" />
                    </div>
                  ))}
                </div>
              ) : (
                <div style={styles.emptyNotice}>No full conv activations available</div>
              )
            ) : arch === "diffusion" ? (
              diffusionView === "Film Strip" ? (
                <div style={styles.filmGrid}>
                  {diffusionFrames.map((src, i) => (
                    <div key={i} style={styles.frameWrap}>
                      {src ? <img src={src} alt={`frame-${i}`} style={styles.frameImg} /> : <div style={styles.framePlaceholder} />}
                      <div style={styles.frameTag}>T{i}</div>
                    </div>
                  ))}
                </div>
              ) : diffusionView === "Noise Curve" ? (
                <MultiLineChart
                  series={[
                    {
                      name: "residual_noise",
                      values: toArray(currentDiffusionEpoch?.energies || []).map(Number),
                      color: "#22d3ee",
                    },
                  ]}
                />
              ) : (
                <Scatter2D
                  connect
                  labels={toArray(currentDiffusionEpoch?.embeddings || []).map((_: any, i: number) => `t${i}`)}
                  points={project2D(toArray(currentDiffusionEpoch?.embeddings || []).map((r: any) => toArray(r).map(Number)))}
                  color="#22d3ee"
                />
              )
            ) : seqSample.length ? (
              seqView === "Heatmap" ? (
                <Heatmap matrix={seqSample.map((r) => r.slice(0, 64).map(Number))} color="viridis" />
              ) : seqView === "Temporal Importance" ? (
                <MultiLineChart series={[{ name: "importance", values: seqImportance.map(Number), color: "#34d399" }]} />
              ) : seqView === "Trajectory" ? (
                <Scatter2D connect points={project2D(seqSample.map((r) => r.slice(0, 8).map(Number)))} labels={seqSample.map((_, i) => `T${i}`)} color="#34d399" />
              ) : (
                <MultiLineChart
                  series={seqSample[0]
                    ? seqSample[0]
                        .map((_: number, i: number) => i)
                        .map((i) => ({
                          idx: i,
                          score: meanAbs1D(seqSample.map((t) => [Number(t[i] || 0)])),
                        }))
                        .sort((a, b) => b.score - a.score)
                        .slice(0, 5)
                        .map((n, k) => ({
                          name: `N${n.idx}`,
                          values: seqSample.map((t) => Number(t[n.idx] || 0)),
                          color: ["#22d3ee", "#a78bfa", "#34d399", "#f59e0b", "#f43f5e"][k % 5],
                        }))
                    : []}
                />
              )
            ) : (
              <div style={styles.emptyNotice}>No sequence activations found</div>
            )}
          </Card>

          <Card
            title={arch === "diffusion" ? "Diffusion Vitals" : "Structural Vitality"}
            subtitle={
              arch === "diffusion"
                ? "Convergence and denoising health at a glance."
                : "Cleaner gradient diagnostics from the latest epoch."
            }
            style={{ gridColumn: "span 5" }}
          >
            {vitalityBars.length ? (
              <>
                <BarChart items={vitalityBars.slice(0, 12)} />
                <div style={{ ...styles.summaryBox, marginTop: 14 }}>
                  {arch === "diffusion" ? (
                    <>

                      <div style={styles.summaryLine}>Avg denoise step: {(() => {
                        const e = toArray(currentDiffusionEpoch?.energies || []).map(Number);
                        if (e.length < 2) return "--";
                        let s = 0;
                        for (let i = 1; i < e.length; i++) s += Math.abs(e[i - 1] - e[i]);
                        return (s / (e.length - 1)).toFixed(4);
                      })()}</div>
                      <div style={styles.summaryLine}>Steps captured: {toArray(currentDiffusionEpoch?.energies || []).length || "--"}</div>
                      <div style={styles.summaryLine}>Frames available: {toArray(currentDiffusionEpoch?.frames || []).length || "--"}</div>
                    </>
                  ) : (
                    <>
                      <div style={styles.summaryLine}>Tracked layers: {Object.keys(currentGradientEpoch || {}).length || "--"}</div>
                      <div style={styles.summaryLine}>Dominant layer: {vitalityBars.length ? vitalityBars.slice().sort((a, b) => b.value - a.value)[0].label : "--"}</div>
                      <div style={styles.summaryLine}>Mean vitality: {vitalityBars.length ? (vitalityBars.reduce((acc, item) => acc + item.value, 0) / vitalityBars.length).toFixed(4) : "--"}</div>
                    </>
                  )}
                </div>
              </>
            ) : (
              <div style={styles.emptyNotice}>{arch === "diffusion" ? "No diffusion vitals found" : "No gradient summaries found"}</div>
            )}
          </Card>

          <Card
            title="Runtime Console"
            subtitle="Live backend logs and training heartbeat."
            style={{ gridColumn: "span 7" }}
            right={<div style={styles.softPill}>{training ? "Polling 800ms" : "Waiting"}</div>}
          >
            <LogsTerminal logs={logs} />
          </Card>

          <Card
            title="Run Snapshot"
            subtitle="High-signal metadata for the current session."
            style={{ gridColumn: "span 5" }}
          >
            <div style={styles.summaryBox}>
              <div style={styles.summaryLine}>Architecture: {arch}</div>
              <div style={styles.summaryLine}>Selected epoch: {epoch}</div>
              <div style={styles.summaryLine}>Selected layer: {layer || "--"}</div>
              <div style={styles.summaryLine}>Samples visible: {sampleOptions.length}</div>
              <div style={styles.summaryLine}>Current epoch from backend: {currentEpoch}</div>
              <div style={styles.summaryLine}>Training state: {training ? "Running" : "Idle"}</div>
              {arch === "transformer" ? (
                <>
                  <div style={styles.summaryLine}>Attention heads: {attentionHeads || "--"}</div>
                  <div style={styles.summaryLine}>Sequence length: {attentionSeq || "--"}</div>
                </>
              ) : null}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  page: {
    minHeight: "100vh",
    background:
      "radial-gradient(circle at top left, rgba(34,211,238,0.12), transparent 28%), radial-gradient(circle at top right, rgba(168,85,247,0.14), transparent 30%), linear-gradient(180deg, #050816 0%, #0b1020 48%, #050816 100%)",
    color: "#f4f4f5",
    position: "relative",
    overflow: "hidden",
    fontFamily: "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
  },
  glowA: {
    position: "absolute",
    width: 420,
    height: 420,
    borderRadius: "50%",
    background: "rgba(34,211,238,0.10)",
    filter: "blur(80px)",
    top: -120,
    left: -120,
    pointerEvents: "none",
  },
  glowB: {
    position: "absolute",
    width: 480,
    height: 480,
    borderRadius: "50%",
    background: "rgba(168,85,247,0.12)",
    filter: "blur(90px)",
    top: 120,
    right: -140,
    pointerEvents: "none",
  },
  container: {
    position: "relative",
    zIndex: 1,
    maxWidth: 1500,
    margin: "0 auto",
    padding: "32px 24px 48px",
  },
  headerRow: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    gap: 16,
    marginBottom: 24,
    flexWrap: "wrap",
  },
  brand: {
    fontSize: 34,
    fontWeight: 800,
    letterSpacing: "0.04em",
    background: "linear-gradient(90deg, #ffffff 0%, #9ae6ff 40%, #d8b4fe 100%)",
    WebkitBackgroundClip: "text",
    WebkitTextFillColor: "transparent",
  },
  brandSub: {
    marginTop: 6,
    color: "#a1a1aa",
    fontSize: 14,
  },
  liveBadge: {
    display: "inline-flex",
    alignItems: "center",
    gap: 10,
    fontWeight: 700,
    fontSize: 12,
    letterSpacing: "0.18em",
    padding: "12px 16px",
    borderRadius: 999,
    background: "rgba(255,255,255,0.04)",
    border: "1px solid rgba(255,255,255,0.08)",
    backdropFilter: "blur(14px)",
  },
  liveDot: {
    width: 10,
    height: 10,
    borderRadius: "50%",
    display: "inline-block",
  },
  controlsRow: {
    display: "flex",
    flexWrap: "wrap",
    gap: 12,
    marginBottom: 20,
  },
  controlWrap: {
    display: "flex",
    flexDirection: "column",
    gap: 8,
    minWidth: 150,
  },
  controlLabel: {
    fontSize: 12,
    color: "#a1a1aa",
    fontWeight: 600,
    letterSpacing: "0.04em",
  },
  select: {
    background: "rgba(10,14,28,0.92)",
    color: "#f4f4f5",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 14,
    padding: "12px 14px",
    outline: "none",
    boxShadow: "0 10px 30px rgba(0,0,0,0.20)",
  },
  metricGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
    gap: 14,
    marginBottom: 18,
  },
  metricTile: {
    background: "linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03))",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 20,
    padding: 18,
    boxShadow: "0 16px 50px rgba(0,0,0,0.24)",
    backdropFilter: "blur(18px)",
  },
  metricLabel: {
    color: "#a1a1aa",
    fontSize: 12,
    fontWeight: 600,
    letterSpacing: "0.04em",
    marginBottom: 8,
  },
  metricValue: {
    fontSize: 26,
    fontWeight: 800,
    lineHeight: 1.1,
  },
  metricDelta: {
    marginTop: 8,
    color: "#67e8f9",
    fontSize: 12,
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(12, minmax(0, 1fr))",
    gap: 16,
  },
  card: {
    background: "linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03))",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 24,
    padding: 18,
    boxShadow: "0 18px 60px rgba(0,0,0,0.28)",
    backdropFilter: "blur(18px)",
    minWidth: 0,
  },
  cardHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    gap: 12,
    marginBottom: 14,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 800,
    marginBottom: 4,
  },
  cardSubtitle: {
    color: "#9ca3af",
    fontSize: 13,
    lineHeight: 1.45,
  },
  smallMeta: {
    marginTop: 12,
    fontSize: 12,
    color: "#9ca3af",
  },
  heatmapGrid: {
    display: "grid",
    gap: 2,
    width: "100%",
    overflow: "hidden",
    borderRadius: 16,
    background: "rgba(255,255,255,0.03)",
    padding: 8,
  },
  emptyNotice: {
    padding: "28px 16px",
    textAlign: "center",
    color: "#9ca3af",
    border: "1px dashed rgba(255,255,255,0.12)",
    borderRadius: 18,
    background: "rgba(255,255,255,0.02)",
  },
  filmGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
    gap: 12,
  },
  frameWrap: {
    position: "relative",
    overflow: "hidden",
    borderRadius: 18,
    border: "1px solid rgba(255,255,255,0.08)",
    background: "rgba(255,255,255,0.03)",
    minHeight: 120,
  },
  frameImg: {
    width: "100%",
    display: "block",
    imageRendering: "pixelated",
  },
  framePlaceholder: {
    minHeight: 120,
    background: "linear-gradient(135deg, rgba(34,211,238,0.14), rgba(168,85,247,0.12))",
  },
  frameTag: {
    position: "absolute",
    left: 10,
    bottom: 10,
    padding: "6px 10px",
    borderRadius: 999,
    fontSize: 11,
    fontWeight: 700,
    background: "rgba(5,8,22,0.76)",
    border: "1px solid rgba(255,255,255,0.08)",
  },
  featureMapGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
    gap: 12,
  },
  featureMapCard: {
    borderRadius: 18,
    padding: 10,
    background: "rgba(255,255,255,0.03)",
    border: "1px solid rgba(255,255,255,0.08)",
  },
  featureMapTitle: {
    fontSize: 12,
    color: "#d4d4d8",
    marginBottom: 8,
    fontWeight: 700,
  },
  terminal: {
    background: "rgba(3,7,18,0.92)",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 18,
    padding: 14,
    minHeight: 260,
    maxHeight: 360,
    overflow: "auto",
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
    fontSize: 12,
  },
  logLine: {
    color: "#d4d4d8",
    padding: "4px 0",
    borderBottom: "1px solid rgba(255,255,255,0.04)",
    whiteSpace: "pre-wrap",
    wordBreak: "break-word",
  },
  logTime: {
    color: "#67e8f9",
  },
  summaryBox: {
    display: "flex",
    flexDirection: "column",
    gap: 10,
    padding: 14,
    borderRadius: 18,
    background: "rgba(255,255,255,0.03)",
    border: "1px solid rgba(255,255,255,0.08)",
  },
  summaryLine: {
    fontSize: 13,
    color: "#e4e4e7",
    lineHeight: 1.5,
  },
  softPill: {
    fontSize: 12,
    color: "#d4d4d8",
    background: "rgba(255,255,255,0.04)",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 999,
    padding: "8px 12px",
  },
};
