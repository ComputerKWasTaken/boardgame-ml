// Hand-rolled SVG line chart (no chart library): training episode
// returns as a moving average, eval means as distinct points.

interface Props {
  returns: { timesteps: number; episode_return: number }[];
  evals: { timesteps: number; mean_return: number }[];
  height?: number;
}

function movingAverage(
  data: { timesteps: number; episode_return: number }[],
  window = 25,
): { x: number; y: number }[] {
  const out: { x: number; y: number }[] = [];
  let sum = 0;
  for (let i = 0; i < data.length; i++) {
    sum += data[i].episode_return;
    if (i >= window) sum -= data[i - window].episode_return;
    const n = Math.min(i + 1, window);
    out.push({ x: data[i].timesteps, y: sum / n });
  }
  return out;
}

export default function Chart({ returns, evals, height = 180 }: Props) {
  const width = 720;
  const pad = { l: 44, r: 10, t: 10, b: 22 };
  const ma = movingAverage(returns);

  const xs = [
    ...ma.map((p) => p.x),
    ...evals.map((e) => e.timesteps),
  ];
  const ys = [
    ...ma.map((p) => p.y),
    ...evals.map((e) => e.mean_return),
    0,
  ];
  if (xs.length === 0) {
    return (
      <div className="chart-empty" style={{ height }}>
        No episodes yet — returns will appear as training runs.
      </div>
    );
  }
  const xMin = 0;
  const xMax = Math.max(...xs, 1);
  let yMin = Math.min(...ys);
  let yMax = Math.max(...ys);
  if (yMax - yMin < 0.5) {
    yMin -= 0.25;
    yMax += 0.25;
  }
  const sx = (x: number) =>
    pad.l + ((x - xMin) / (xMax - xMin)) * (width - pad.l - pad.r);
  const sy = (y: number) =>
    pad.t + (1 - (y - yMin) / (yMax - yMin)) * (height - pad.t - pad.b);

  const line = ma
    .map((p, i) => `${i === 0 ? "M" : "L"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`)
    .join(" ");
  const ticks = [yMin, (yMin + yMax) / 2, yMax];

  return (
    <svg
      className="chart"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Training returns chart"
    >
      {ticks.map((t, i) => (
        <g key={i}>
          <line
            x1={pad.l}
            x2={width - pad.r}
            y1={sy(t)}
            y2={sy(t)}
            className="chart-grid"
          />
          <text x={pad.l - 6} y={sy(t) + 4} className="chart-label" textAnchor="end">
            {t.toFixed(1)}
          </text>
        </g>
      ))}
      <line
        x1={pad.l}
        x2={width - pad.r}
        y1={sy(0)}
        y2={sy(0)}
        className="chart-zero"
      />
      {ma.length > 1 && <path d={line} className="chart-line" fill="none" />}
      {evals.map((e, i) => (
        <circle
          key={i}
          cx={sx(e.timesteps)}
          cy={sy(e.mean_return)}
          r={3.5}
          className="chart-eval"
        >
          <title>
            eval @ {e.timesteps}: {e.mean_return.toFixed(3)}
          </title>
        </circle>
      ))}
      <text x={width - pad.r} y={height - 6} className="chart-label" textAnchor="end">
        timesteps
      </text>
    </svg>
  );
}
