import { useCallback, useEffect, useRef, useState } from "react";
import { fetchSampleFile, postForm } from "../api.js";
import { Button, DownloadLink, Dropzone, ErrorNote, Frame, Readout, SampleStrip, Segmented, useImageInput } from "../components/ui.jsx";

const STYLES = [1, 2, 3].map((n) => ({ value: n, label: `Style ${n}` }));

function Webcam({ onCapture }) {
  const video = useRef(null);
  const stream = useRef(null);
  const [on, setOn] = useState(false);
  const [problem, setProblem] = useState("");

  const stop = useCallback(() => {
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    setOn(false);
  }, []);
  useEffect(() => stop, [stop]);

  const start = async () => {
    setProblem("");
    if (!navigator.mediaDevices?.getUserMedia) return setProblem("The camera needs https or localhost. Upload a photo instead.");
    try {
      stream.current = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
      video.current.srcObject = stream.current;
      await video.current.play();
      setOn(true);
    } catch (e) {
      setProblem(`Camera unavailable: ${e.message}`);
    }
  };

  const capture = () => {
    const v = video.current;
    const canvas = document.createElement("canvas");
    canvas.width = v.videoWidth;
    canvas.height = v.videoHeight;
    canvas.getContext("2d").drawImage(v, 0, 0);
    canvas.toBlob((b) => { onCapture(new File([b], "webcam.jpg", { type: "image/jpeg" })); stop(); }, "image/jpeg", 0.92);
  };

  return (
    <div className="space-y-3">
      <video ref={video} playsInline muted className={`aspect-video w-full bg-ink object-cover [transform:scaleX(-1)] ${on ? "" : "hidden"}`} />
      {on ? (
        <div className="flex gap-2">
          <Button onClick={capture} className="flex-1">Capture photo</Button>
          <Button variant="quiet" onClick={stop}>Cancel</Button>
        </div>
      ) : (
        <Button variant="quiet" onClick={start} className="w-full">Start camera</Button>
      )}
      <ErrorNote>{problem}</ErrorNote>
    </div>
  );
}

export default function Sketch() {
  const { file, preview, setFile } = useImageInput();
  const [source, setSource] = useState("upload");
  const [sample, setSample] = useState(null);
  const [style, setStyle] = useState(1);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [res, setRes] = useState(null);

  const pickSample = async (name) => {
    try {
      setError("");
      setFile(await fetchSampleFile("faces", name));
      setSample(name);
    } catch (e) {
      setError(e.message);
    }
  };

  const generate = async () => {
    setBusy(true);
    setError("");
    const form = new FormData();
    form.append("file", file);
    form.append("style", String(style));
    try {
      setRes(await postForm("/sketch", form));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid gap-8 lg:grid-cols-[19rem_1fr]">
      <div className="space-y-5">
        <Segmented label="Photo source" value={source} onChange={setSource}
          options={[{ value: "upload", label: "Upload" }, { value: "webcam", label: "Webcam" }]} />
        {source === "upload" ? (
          <>
            <Dropzone onFile={(f) => { setSample(null); setFile(f); }} preview={preview} hint="Upload a face photo. It is cropped to a centred square." />
            <SampleStrip kind="faces" selected={sample} onPick={pickSample} />
          </>
        ) : (
          <>
            <Webcam onCapture={(f) => { setSample(null); setFile(f); }} />
            {preview && <img src={preview} alt="Captured photo" className="h-20 w-20 object-cover" />}
          </>
        )}
        <Segmented label="Sketch style" value={style} options={STYLES} onChange={setStyle} />
        <Button onClick={generate} disabled={!file || busy} className="w-full">{busy ? "Working" : "Generate sketch"}</Button>
        <ErrorNote>{error}</ErrorNote>
      </div>

      <div className="min-w-0 space-y-8">
        <div className="grid gap-6 sm:grid-cols-2">
          <Frame title="Photo" src={res?.photo_image} empty="Choose a photo and generate" />
          <Frame title={res ? `Sketch, style ${res.style}` : "Sketch"} src={res?.sketch_image} empty="The sketch appears here"
            action={res && <DownloadLink href={res.sketch_image} name={`sketch-style-${res.style}.png`} />} />
        </div>
        {res && (
          <Readout title="Run details" rows={[
            ["Style", `Style ${res.style}`],
            ["Inference time", `${res.inference_ms} ms`],
            ["Model resolution", `${res.model_resolution} x ${res.model_resolution} px`],
            ["Shown and saved at", `${res.model_resolution * res.display_upscale} x ${res.model_resolution * res.display_upscale} px (${res.display_upscale}x enlargement)`],
          ]} />
        )}
      </div>
    </div>
  );
}
