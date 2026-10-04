const BASE = import.meta.env.VITE_API_BASE ?? "/api";

async function parse(res) {
  if (!res.ok) {
    let message = `Request failed (${res.status}).`;
    try {
      const body = await res.json();
      if (body.detail) {
        message = typeof body.detail === "string" ? body.detail : body.detail.map((d) => d.msg).join("; ");
      }
    } catch {
      /* body was not JSON */
    }
    throw new Error(message);
  }
  return res.json();
}

export const getHealth = () => fetch(`${BASE}/health`).then(parse);
export const getSamples = () => fetch(`${BASE}/samples`).then(parse);
export const postForm = (path, form) => fetch(`${BASE}${path}`, { method: "POST", body: form }).then(parse);
export const sampleUrl = (kind, name) => `${BASE}/samples/${kind}/${encodeURIComponent(name)}`;

export async function fetchSampleFile(kind, name) {
  const res = await fetch(sampleUrl(kind, name));
  if (!res.ok) throw new Error("Could not load the sample image.");
  const blob = await res.blob();
  return new File([blob], name, { type: blob.type || "image/jpeg" });
}
