import { useState } from "react";
import { fetchSampleFile, postForm } from "../api.js";
import {
  Button, Distribution, DownloadLink, Dropzone, ErrorNote, Frame, Readout, SampleStrip, Segmented,
  metricRows, settingsRows, useImageInput,
} from "../components/ui.jsx";

const CORRUPTIONS = [
  { value: "none", label: "As uploaded" },
  { value: "salt", label: "Salt & pepper" },
  { value: "blur", label: "Blur" },
  { value: "occlusion", label: "Occlusion" },
];
const SEVERITIES = [{ value: "low", label: "Low" }, { value: "medium", label: "Medium" }, { value: "high", label: "High" }];
const LABELS = { clean: "Clean", identity: "Identity (clean)", salt: "Salt-and-pepper", blur: "Blur", occlusion: "Occlusion" };

const MODES = {
  universal: { path: "/universal", run: "Restore image" },
  hard: { path: "/hard", run: "Classify and restore" },
  soft: { path: "/soft", run: "Mix experts and restore" },
};

export default function Restoration({ mode }) {
  const cfg = MODES[mode];
  const { file, preview, setFile } = useImageInput();
  const [sample, setSample] = useState(null);
  const [corruption, setCorruption] = useState("salt");
  const [severity, setSeverity] = useState("medium");
  const [seed, setSeed] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [res, setRes] = useState(null);

  const pickSample = async (name) => {
    try {
      setError("");
      setFile(await fetchSampleFile("pets", name));
      setSample(name);
    } catch (e) {
      setError(e.message);
    }
  };

  const run = async () => {
    setBusy(true);
    setError("");
    const form = new FormData();
    form.append("file", file);
    form.append("corruption", corruption);
    form.append("severity", severity);
    if (seed.trim() !== "") form.append("seed", seed.trim());
    try {
      setRes(await postForm(cfg.path, form));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const rows = [];
  if (res) {
    if (mode === "hard") {
      rows.push(["Predicted corruption", LABELS[res.predicted]]);
      rows.push(["Selected expert", res.selected_expert]);
      if (res.true_label) rows.push(["Routing", res.routing_correct ? "Correct" : `Wrong (true: ${LABELS[res.true_label]})`]);
      rows.push(["Classifier time", `${res.classifier_ms} ms`]);
      rows.push(["Expert time", `${res.expert_ms} ms`]);
    }
    if (mode === "soft") {
      rows.push(["Dominant branch", LABELS[res.dominant]]);
      rows.push(["Contributing (5% or more)", res.contributors.map((c) => LABELS[c]).join(", ")]);
    }
    rows.push(["Inference time", `${res.inference_ms} ms`]);
  }

  return (
    <div className="grid gap-8 lg:grid-cols-[19rem_1fr]">
      <div className="space-y-5">
        <Dropzone onFile={(f) => { setSample(null); setFile(f); }} preview={preview} hint="Upload a clean or an already corrupted image." />
        <SampleStrip kind="pets" selected={sample} onPick={pickSample} />
        <Segmented label="Corruption to apply" value={corruption} options={CORRUPTIONS} onChange={setCorruption} />
        <Segmented label="Severity" value={severity} options={SEVERITIES} onChange={setSeverity} disabled={corruption === "none"} />
        <label className="block text-sm">
          <span className="mb-1.5 block font-medium">Seed (optional)</span>
          <input value={seed} onChange={(e) => setSeed(e.target.value.replace(/\D/g, ""))} inputMode="numeric" placeholder="Random"
            disabled={corruption === "none"} className="w-full border border-line bg-paper px-3 py-2 disabled:opacity-50" />
        </label>
        <Button onClick={run} disabled={!file || busy} className="w-full">{busy ? "Working" : cfg.run}</Button>
        <ErrorNote>{error}</ErrorNote>
      </div>

      <div className="min-w-0 space-y-8">
        <div className={`grid gap-6 ${res?.clean_image ? "sm:grid-cols-3" : "sm:grid-cols-2"}`}>
          {res?.clean_image && <Frame title="Clean target" src={res.clean_image} />}
          <Frame title="Input to the model" src={res?.input_image} empty="Choose an image and run" />
          <Frame title="Restored" src={res?.output_image} empty="The result appears here"
            action={res && <DownloadLink href={res.output_image} name={`restored-${mode}.png`} />} />
        </div>

        {res && mode === "hard" && (
          <Distribution title="Classifier probabilities" highlight={[res.predicted]}
            items={Object.entries(res.probabilities).map(([key, value]) => ({ key, value, label: LABELS[key] }))} />
        )}
        {res && mode === "soft" && (
          <Distribution title="Routing weights" highlight={res.contributors}
            note="Weights sum to 1. Bold rows each contribute at least 5% of the output."
            items={Object.entries(res.weights).map(([key, value]) => ({ key, value, label: LABELS[key] }))} />
        )}

        {res && (
          <div className="grid gap-8 md:grid-cols-2">
            <Readout title="Corruption settings" rows={settingsRows(res.settings)} />
            <Readout title="Run details" rows={[...rows, ...metricRows(res.metrics)]} />
          </div>
        )}
      </div>
    </div>
  );
}
