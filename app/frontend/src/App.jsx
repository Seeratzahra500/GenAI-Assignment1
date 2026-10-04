import { useEffect, useState } from "react";
import { getHealth } from "./api.js";
import Restoration from "./workspaces/Restoration.jsx";
import Sketch from "./workspaces/Sketch.jsx";

const WORKSPACES = [
  { id: "universal", name: "Universal Restoration", blurb: "One autoencoder restores every corruption.", node: <Restoration mode="universal" /> },
  { id: "hard", name: "Hard-Routed Restoration", blurb: "A classifier sends the image to one specialist.", node: <Restoration mode="hard" /> },
  { id: "soft", name: "Soft Mixture-of-Experts Restoration", blurb: "A gate blends the specialists with learned weights.", node: <Restoration mode="soft" /> },
  { id: "sketch", name: "Face-to-Sketch Generator", blurb: "Turn a face photo into a sketch in one of three styles.", node: <Sketch /> },
];

function BackendStatus() {
  const [h, setH] = useState(null);
  useEffect(() => {
    const poll = () => getHealth().then(setH).catch(() => setH({ offline: true }));
    poll();
    const t = setInterval(poll, 20000);
    return () => clearInterval(t);
  }, []);
  let text = "Checking backend";
  let color = "bg-steel";
  if (h?.offline) { text = "Backend unreachable"; color = "bg-danger"; }
  else if (h) { text = `${h.loaded} of ${h.total} models loaded`; color = h.loaded === h.total ? "bg-teal" : "bg-amber"; }
  return (
    <p className="flex items-center gap-2 text-sm text-muted" role="status">
      <i className={`h-2.5 w-2.5 rounded-full ${color}`} />
      {text}
    </p>
  );
}

export default function App() {
  const [active, setActive] = useState("universal");
  const current = WORKSPACES.find((w) => w.id === active);

  return (
    <div className="min-h-screen md:grid md:grid-cols-[17rem_1fr]">
      <nav aria-label="Workspaces" className="bg-ink text-white md:sticky md:top-0 md:h-screen">
        <div className="px-5 py-5 md:py-8">
          <p className="text-xl font-semibold">RestoreLab</p>
          <p className="mt-1 text-sm text-white/60">Image restoration and face-to-sketch generation</p>
        </div>
        <ul className="flex overflow-x-auto md:block">
          {WORKSPACES.map((w) => (
            <li key={w.id} className="shrink-0">
              <button
                onClick={() => setActive(w.id)}
                aria-current={active === w.id ? "page" : undefined}
                className={`block w-full border-l-4 px-5 py-3 text-left text-sm transition-colors ${
                  active === w.id ? "border-teal bg-white/10 font-semibold" : "border-transparent text-white/75 hover:bg-white/5"
                }`}
              >
                {w.name}
              </button>
            </li>
          ))}
        </ul>
      </nav>

      <main className="min-w-0 px-5 py-6 md:px-10 md:py-8">
        <header className="mb-8 flex flex-wrap items-start justify-between gap-3">
          <div>
            <h1 className="text-2xl font-semibold">{current.name}</h1>
            <p className="mt-1 max-w-prose text-muted">{current.blurb}</p>
          </div>
          <BackendStatus />
        </header>
        {WORKSPACES.map((w) => (
          <section key={w.id} hidden={w.id !== active} aria-label={w.name}>{w.node}</section>
        ))}
      </main>
    </div>
  );
}
