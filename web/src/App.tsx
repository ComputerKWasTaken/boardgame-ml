import { useEffect, useState } from "react";
import GamesRuns from "./views/GamesRuns";
import TrainView from "./views/TrainView";
import WatchView from "./views/WatchView";
import { store } from "./storage";
import type { WatchTarget } from "./types";

type View = "games" | "train" | "watch";

export default function App() {
  const [view, setView] = useState<View>(() => {
    const last = store.prefs.get().lastView;
    return last === "train" || last === "watch" ? last : "games";
  });
  const [watchTarget, setWatchTarget] = useState<WatchTarget | null>(null);
  const [runsRefresh, setRunsRefresh] = useState(0);

  useEffect(() => {
    store.prefs.set({ lastView: view });
  }, [view]);

  const goWatch = (t: WatchTarget) => {
    setWatchTarget(t);
    setView("watch");
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark">♞</span>
          <div>
            <h1>boardgame-ml</h1>
            <p>local cockpit — train agents, watch them play</p>
          </div>
        </div>
        <nav className="tabs">
          <button
            className={view === "games" ? "tab active" : "tab"}
            onClick={() => setView("games")}
          >
            Games &amp; Runs
          </button>
          <button
            className={view === "train" ? "tab active" : "tab"}
            onClick={() => setView("train")}
          >
            Train
          </button>
          <button
            className={view === "watch" ? "tab active" : "tab"}
            onClick={() => setView("watch")}
          >
            Watch
          </button>
        </nav>
      </header>
      <main>
        {view === "games" && (
          <GamesRuns onWatch={goWatch} refreshKey={runsRefresh} />
        )}
        {view === "train" && (
          <TrainView
            onWatch={goWatch}
            onRunFinished={() => setRunsRefresh((k) => k + 1)}
          />
        )}
        {view === "watch" && (
          <WatchView target={watchTarget} onWatch={goWatch} />
        )}
      </main>
      <footer className="app-footer">
        Runs and checkpoints live on disk in <code>runs/</code>; this
        browser stores only preferences, starred runs, replays, and
        history.
      </footer>
    </div>
  );
}
