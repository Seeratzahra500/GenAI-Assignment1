import { useEffect, useRef, useState } from "react";
import { getSamples, sampleUrl } from "../api.js";

export const MAX_UPLOAD_MB = 8;

export function useImageInput() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  useEffect(() => {
    if (!file) {
      setPreview(null);
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  return { file, preview, setFile };
}

export function Button({ children, variant = "primary", className = "", ...props }) {
  const styles = {
    primary: "bg-teal text-white hover:bg-teal-dark disabled:bg-line disabled:text-muted",
    quiet: "border border-line bg-paper text-ink hover:border-ink disabled:text-muted",
  };
  return (
    <button className={`rounded px-4 py-2.5 text-sm font-medium transition-colors ${styles[variant]} ${className}`} {...props}>
      {children}
    </button>
  );
}

export function ErrorNote({ children }) {
  if (!children) return null;
  return (
    <p role="alert" className="border-l-4 border-danger bg-paper px-3 py-2 text-sm text-danger">
      {children}
    </p>
  );
}

export function Segmented({ label, value, options, onChange, disabled = false }) {
  return (
    <div className={disabled ? "opacity-50" : ""}>
      <p className="mb-1.5 text-sm font-medium">{label}</p>
      <div role="radiogroup" aria-label={label} className="flex border border-line bg-paper">
        {options.map((o) => {
          const on = o.value === value;
          return (
            <button
              key={o.value}
              type="button"
              role="radio"
              aria-checked={on}
              disabled={disabled}
              onClick={() => onChange(o.value)}
              className={`flex-1 px-2 py-2 text-sm transition-colors ${on ? "bg-ink text-white" : "text-ink hover:bg-canvas"}`}
            >
              {o.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function Dropzone({ onFile, preview, hint }) {
  const input = useRef(null);
  const [over, setOver] = useState(false);
  const [problem, setProblem] = useState("");

  const accept = (f) => {
    if (!f) return;
    if (!/^image\/(jpeg|png|webp)$/.test(f.type)) return setProblem("Use a JPEG, PNG or WebP image.");
    if (f.size > MAX_UPLOAD_MB * 1024 * 1024) return setProblem(`The file is larger than ${MAX_UPLOAD_MB} MB.`);
    setProblem("");
    onFile(f);
  };

  return (
    <div>
      <div
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); accept(e.dataTransfer.files[0]); }}
        className={`flex items-center gap-4 border border-dashed p-3 ${over ? "border-teal bg-teal-soft" : "border-steel bg-paper"}`}
      >
        {preview ? (
          <img src={preview} alt="Selected input" className="h-20 w-20 shrink-0 object-cover" />
        ) : (
          <div className="grid h-20 w-20 shrink-0 place-items-center bg-canvas text-xs text-muted">No image</div>
        )}
        <div className="min-w-0 text-sm">
          <p className="text-muted">{hint}</p>
          <button type="button" onClick={() => input.current.click()} className="mt-1 font-medium text-teal underline underline-offset-2 hover:text-teal-dark">
            Choose a file
          </button>
          <input ref={input} type="file" accept="image/jpeg,image/png,image/webp" className="sr-only" onChange={(e) => { accept(e.target.files[0]); e.target.value = ""; }} />
        </div>
      </div>
      <ErrorNote>{problem}</ErrorNote>
    </div>
  );
}

export function SampleStrip({ kind, selected, onPick }) {
  const [names, setNames] = useState(null);
  useEffect(() => {
    getSamples().then((s) => setNames(s[kind] || [])).catch(() => setNames([]));
  }, [kind]);
  if (names === null) return <p className="text-sm text-muted">Loading samples</p>;
  if (names.length === 0) return <p className="text-sm text-muted">No sample images are installed. Upload your own image.</p>;
  return (
    <div>
      <p className="mb-1.5 text-sm font-medium">Clean samples</p>
      <div className="flex gap-2 overflow-x-auto pb-1">
        {names.map((n) => (
          <button key={n} type="button" onClick={() => onPick(n)} aria-label={`Use sample ${n}`} aria-pressed={selected === n}
            className={`h-14 w-14 shrink-0 overflow-hidden border-2 ${selected === n ? "border-teal" : "border-transparent hover:border-steel"}`}>
            <img src={sampleUrl(kind, n)} alt="" className="h-full w-full object-cover" />
          </button>
        ))}
      </div>
    </div>
  );
}

/** Image with corner ticks, like crop marks on a contact sheet. */
export function Frame({ title, src, empty = "Nothing to show yet", action }) {
  const tick = "absolute h-3 w-3 border-ink";
  return (
    <figure>
      <figcaption className="mb-2 flex items-baseline justify-between gap-2 text-sm">
        <span className="font-medium">{title}</span>
        {action}
      </figcaption>
      <div className="relative bg-paper p-2">
        <span className={`${tick} -left-1 -top-1 border-l-2 border-t-2`} />
        <span className={`${tick} -right-1 -top-1 border-r-2 border-t-2`} />
        <span className={`${tick} -bottom-1 -left-1 border-b-2 border-l-2`} />
        <span className={`${tick} -bottom-1 -right-1 border-b-2 border-r-2`} />
        <div className="grid aspect-square place-items-center bg-canvas">
          {src ? <img src={src} alt={title} className="h-full w-full object-contain" /> : <span className="px-4 text-center text-sm text-muted">{empty}</span>}
        </div>
      </div>
    </figure>
  );
}

export function DownloadLink({ href, name }) {
  return (
    <a href={href} download={name} className="font-medium text-teal underline underline-offset-2 hover:text-teal-dark">
      Download
    </a>
  );
}

const BRANCH_COLOR = { identity: "#6B7A90", clean: "#6B7A90", salt: "#0B7F86", blur: "#3D5A9B", occlusion: "#D9822B" };

/** A stacked strip plus one row per branch. Used for classifier probabilities and gate weights. */
export function Distribution({ title, items, highlight = [], note }) {
  return (
    <section aria-label={title}>
      <h3 className="mb-2 text-sm font-medium">{title}</h3>
      <div className="flex h-5 w-full overflow-hidden border border-line bg-canvas" role="img"
        aria-label={items.map((i) => `${i.label} ${(i.value * 100).toFixed(1)} percent`).join(", ")}>
        {items.map((i) => (
          <div key={i.key} style={{ width: `${i.value * 100}%`, background: BRANCH_COLOR[i.key] }} />
        ))}
      </div>
      <ul className="mt-3 space-y-1.5">
        {items.map((i) => {
          const on = highlight.includes(i.key);
          return (
            <li key={i.key} className={`grid grid-cols-[8.5rem_1fr_3.6rem] items-center gap-3 text-sm ${on ? "font-semibold" : "text-muted"}`}>
              <span className="flex items-center gap-2"><i className="h-2.5 w-2.5" style={{ background: BRANCH_COLOR[i.key] }} />{i.label}</span>
              <span className="h-1.5 bg-canvas"><span className="block h-full" style={{ width: `${i.value * 100}%`, background: BRANCH_COLOR[i.key] }} /></span>
              <span className="text-right">{(i.value * 100).toFixed(1)}%</span>
            </li>
          );
        })}
      </ul>
      {note && <p className="mt-2 text-xs text-muted">{note}</p>}
    </section>
  );
}

export function Readout({ title, rows }) {
  return (
    <section aria-label={title}>
      <h3 className="mb-2 text-sm font-medium">{title}</h3>
      <dl className="divide-y divide-line border-y border-line text-sm">
        {rows.filter(Boolean).map(([k, v]) => (
          <div key={k} className="grid grid-cols-[10rem_1fr] gap-3 py-1.5">
            <dt className="text-muted">{k}</dt>
            <dd className="min-w-0 break-words">{v}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

export function settingsRows(s) {
  if (!s || s.type === "none") return [["Corruption", "None (image used as uploaded)"]];
  const rows = [["Corruption", { salt: "Salt-and-pepper noise", blur: "Gaussian blur", occlusion: "Rectangular occlusion" }[s.type]], ["Severity", s.severity]];
  if (s.type === "salt") rows.push(["Pixels replaced", `${(s.p * 100).toFixed(0)}%`]);
  if (s.type === "blur") rows.push(["Kernel / sigma", `${s.kernel} x ${s.kernel} / ${s.sigma}`]);
  if (s.type === "occlusion") {
    rows.push(["Rectangles", s.rects.length]);
    rows.push(["Area covered", `${(s.measured_area * 100).toFixed(1)}% (target ${(s.target_area * 100).toFixed(0)}%)`]);
  }
  rows.push(["Seed", s.seed]);
  return rows;
}

export function metricRows(m) {
  if (!m) return [["Quality", "Not available (no clean reference for an uploaded corrupted image)"]];
  return [
    ["PSNR vs clean", `${m.psnr} dB (input ${m.input_psnr} dB)`],
    ["SSIM vs clean", `${m.ssim} (input ${m.input_ssim})`],
  ];
}
