import { useEffect, useState } from "react";
import "./App.css";

import {
  getHealth,
  getLiveMatches,
  predictMatch,
  sendChat,
} from "./api";

import type {
  HealthResponse,
  LiveMatch,
  MatchFormat,
  PredictionMode,
  PredictionRequest,
  PredictionResponse,
} from "./types";

type Page = "home" | "predict" | "live" | "chat";

function percentage(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function PredictionCard({
  result,
}: {
  result: PredictionResponse;
}) {
  return (
    <section className="prediction-result">
      <div className="result-heading">
        <div>
          <span className="eyebrow">CRICMASTER PREDICTION</span>
          <h2>{result.predicted_team} has the edge</h2>
        </div>

        <span
          className={`confidence ${result.confidence.toLowerCase()}`}
        >
          {result.confidence} CONFIDENCE
        </span>
      </div>

      <div className="probability-grid">
        <div className="team-probability">
          <span className="team-name">{result.team1}</span>
          <strong>
            {percentage(result.team1_probability)}
          </strong>

          <div className="probability-track">
            <div
              className="probability-fill"
              style={{
                width: percentage(result.team1_probability),
              }}
            />
          </div>
        </div>

        <div className="versus">VS</div>

        <div className="team-probability right">
          <span className="team-name">{result.team2}</span>
          <strong>
            {percentage(result.team2_probability)}
          </strong>

          <div className="probability-track">
            <div
              className="probability-fill secondary"
              style={{
                width: percentage(result.team2_probability),
              }}
            />
          </div>
        </div>
      </div>

      <div className="result-meta">
        <span>
          Edge <strong>{result.edge}</strong>
        </span>

        <span>
          Mode <strong>{result.prediction_mode}</strong>
        </span>

        <span>
          Model <strong>{result.model_family}</strong>
        </span>
      </div>

      {result.drivers && result.drivers.length > 0 && (
        <div className="drivers">
          <h3>Main statistical drivers</h3>

          {result.drivers.slice(0, 4).map((driver) => (
            <div className="driver" key={driver.feature}>
              <span>{driver.label}</span>
              <strong>{driver.supports ?? "Neutral"}</strong>
            </div>
          ))}
        </div>
      )}

      {result.warnings && result.warnings.length > 0 && (
        <div className="warnings">
          {result.warnings.map((warning) => (
            <p key={warning}>⚠ {warning}</p>
          ))}
        </div>
      )}
    </section>
  );
}

function PredictPage() {
  const [team1, setTeam1] = useState("India");
  const [team2, setTeam2] = useState("Australia");
  const [format, setFormat] =
    useState<MatchFormat>("T20I");
  const [mode, setMode] =
    useState<PredictionMode>("PRE_TOSS");

  const [date, setDate] = useState(
    new Date().toISOString().slice(0, 10),
  );

  const [competition, setCompetition] = useState("");
  const [venue, setVenue] = useState("");

  const [tossWinner, setTossWinner] = useState("");
  const [tossDecision, setTossDecision] =
    useState<"bat" | "field">("field");

  const [team1Xi, setTeam1Xi] = useState("");
  const [team2Xi, setTeam2Xi] = useState("");

  const [battingTeam, setBattingTeam] = useState("");
  const [innings, setInnings] = useState(1);
  const [runs, setRuns] = useState(0);
  const [wickets, setWickets] = useState(0);
  const [overs, setOvers] = useState("0.0");
  const [target, setTarget] = useState(0);

  const [result, setResult] =
    useState<PredictionResponse | null>(null);

  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submitPrediction(
    event: React.FormEvent,
  ) {
    event.preventDefault();

    setLoading(true);
    setError("");
    setResult(null);

    const payload: PredictionRequest = {
      team1: team1.trim(),
      team2: team2.trim(),
      format,
      mode,
      date,
    };

    if (competition.trim()) {
      payload.competition = competition.trim();
    }

    if (venue.trim()) {
      payload.venue = venue.trim();
    }

    if (mode === "POST_TOSS") {
      payload.toss_winner = tossWinner || team1;
      payload.toss_decision = tossDecision;

      const firstXi = team1Xi
        .split(",")
        .map((player) => player.trim())
        .filter(Boolean);

      const secondXi = team2Xi
        .split(",")
        .map((player) => player.trim())
        .filter(Boolean);

      if (firstXi.length) payload.team1_xi = firstXi;
      if (secondXi.length) payload.team2_xi = secondXi;
    }

    if (mode === "LIVE") {
      payload.batting_team = battingTeam || team1;
      payload.innings = innings;
      payload.runs = runs;
      payload.wickets = wickets;
      payload.overs = overs;

      if (innings === 2) {
        payload.target = target;
      }
    }

    try {
      const response = await predictMatch(payload);
      setResult(response);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Prediction failed.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page-container">
      <div className="page-heading">
        <span className="eyebrow">MATCH INTELLIGENCE</span>
        <h1>Predict a cricket match</h1>
        <p>
          Historical form, roster strength and live match
          context combined into one probabilistic estimate.
        </p>
      </div>

      <div className="predict-layout">
        <form
          className="prediction-form card"
          onSubmit={submitPrediction}
        >
          <div className="mode-selector">
            {(
              [
                "PRE_TOSS",
                "POST_TOSS",
                "LIVE",
              ] as PredictionMode[]
            ).map((item) => (
              <button
                type="button"
                key={item}
                className={mode === item ? "active" : ""}
                onClick={() => setMode(item)}
              >
                {item.replace("_", " ")}
              </button>
            ))}
          </div>

          <div className="form-grid">
            <label>
              Team 1
              <input
                value={team1}
                onChange={(e) => setTeam1(e.target.value)}
                required
              />
            </label>

            <label>
              Team 2
              <input
                value={team2}
                onChange={(e) => setTeam2(e.target.value)}
                required
              />
            </label>

            <label>
              Format
              <select
                value={format}
                onChange={(e) =>
                  setFormat(e.target.value as MatchFormat)
                }
              >
                <option value="T20I">T20I</option>
                <option value="T20">T20</option>
              </select>
            </label>

            <label>
              Match date
              <input
                type="date"
                value={date}
                onChange={(e) => setDate(e.target.value)}
                required
              />
            </label>

            <label>
              Competition
              <input
                placeholder="IPL, BBL, PSL..."
                value={competition}
                onChange={(e) =>
                  setCompetition(e.target.value)
                }
              />
            </label>

            <label>
              Venue
              <input
                placeholder="Optional"
                value={venue}
                onChange={(e) => setVenue(e.target.value)}
              />
            </label>
          </div>

          {mode === "POST_TOSS" && (
            <div className="conditional-section">
              <h3>Toss & playing XI</h3>

              <div className="form-grid">
                <label>
                  Toss winner
                  <select
                    value={tossWinner}
                    onChange={(e) =>
                      setTossWinner(e.target.value)
                    }
                  >
                    <option value={team1}>{team1}</option>
                    <option value={team2}>{team2}</option>
                  </select>
                </label>

                <label>
                  Toss decision
                  <select
                    value={tossDecision}
                    onChange={(e) =>
                      setTossDecision(
                        e.target.value as "bat" | "field",
                      )
                    }
                  >
                    <option value="bat">Bat</option>
                    <option value="field">Field</option>
                  </select>
                </label>
              </div>

              <label>
                {team1} XI
                <textarea
                  placeholder="Player 1, Player 2, Player 3..."
                  value={team1Xi}
                  onChange={(e) => setTeam1Xi(e.target.value)}
                />
              </label>

              <label>
                {team2} XI
                <textarea
                  placeholder="Player 1, Player 2, Player 3..."
                  value={team2Xi}
                  onChange={(e) => setTeam2Xi(e.target.value)}
                />
              </label>
            </div>
          )}

          {mode === "LIVE" && (
            <div className="conditional-section">
              <h3>Live match state</h3>

              <div className="form-grid">
                <label>
                  Batting team
                  <select
                    value={battingTeam}
                    onChange={(e) =>
                      setBattingTeam(e.target.value)
                    }
                  >
                    <option value={team1}>{team1}</option>
                    <option value={team2}>{team2}</option>
                  </select>
                </label>

                <label>
                  Innings
                  <select
                    value={innings}
                    onChange={(e) =>
                      setInnings(Number(e.target.value))
                    }
                  >
                    <option value={1}>1st innings</option>
                    <option value={2}>2nd innings</option>
                  </select>
                </label>

                <label>
                  Runs
                  <input
                    type="number"
                    min="0"
                    value={runs}
                    onChange={(e) =>
                      setRuns(Number(e.target.value))
                    }
                  />
                </label>

                <label>
                  Wickets
                  <input
                    type="number"
                    min="0"
                    max="10"
                    value={wickets}
                    onChange={(e) =>
                      setWickets(Number(e.target.value))
                    }
                  />
                </label>

                <label>
                  Overs
                  <input
                    placeholder="15.3"
                    value={overs}
                    onChange={(e) => setOvers(e.target.value)}
                  />
                </label>

                {innings === 2 && (
                  <label>
                    Target
                    <input
                      type="number"
                      min="1"
                      value={target}
                      onChange={(e) =>
                        setTarget(Number(e.target.value))
                      }
                    />
                  </label>
                )}
              </div>
            </div>
          )}

          <button
            className="primary-button"
            disabled={loading}
          >
            {loading
              ? "Analysing match..."
              : "Run Cricmaster Prediction"}
          </button>

          {error && <div className="error-box">{error}</div>}
        </form>

        <div>
          {result ? (
            <PredictionCard result={result} />
          ) : (
            <div className="empty-result card">
              <div className="cricket-ball">🏏</div>
              <h2>Your prediction appears here</h2>
              <p>
                Choose the match state and Cricmaster will
                return probabilities, confidence and the
                strongest available statistical drivers.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function LivePage() {
  const [matches, setMatches] = useState<LiveMatch[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function loadMatches() {
    setLoading(true);
    setError("");

    try {
      const result = await getLiveMatches();
      setMatches(Array.isArray(result) ? result : []);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Could not load live matches.",
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadMatches();
  }, []);

  return (
    <div className="page-container">
      <div className="page-heading split-heading">
        <div>
          <span className="eyebrow">LIVE CRICKET</span>
          <h1>Matches happening now</h1>
          <p>
            Live score state from Cricmaster's configured
            cricket data provider.
          </p>
        </div>

        <button
          className="secondary-button"
          onClick={loadMatches}
        >
          Refresh
        </button>
      </div>

      {loading && <div className="card status-card">Loading…</div>}

      {error && <div className="error-box">{error}</div>}

      {!loading && !error && matches.length === 0 && (
        <div className="card status-card">
          <h2>No live matches available</h2>
          <p>
            Either no supported match is live or the provider
            returned no current fixtures.
          </p>
        </div>
      )}

      <div className="live-grid">
        {matches.map((match) => (
          <article className="live-card" key={match.match_id}>
            <div className="live-status">
              <span className="live-dot" />
              {match.status || "LIVE"}
            </div>

            <h2>{match.name || match.teams?.join(" vs ")}</h2>

            <div className="live-meta">
              <span>{match.format || "Cricket"}</span>
              <span>{match.match_id}</span>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}

function ChatPage() {
  const [intent, setIntent] = useState<
    "help" | "model_capabilities" | "predict_match"
  >("help");

  const [team1, setTeam1] = useState("India");
  const [team2, setTeam2] = useState("Australia");
  const [format, setFormat] =
    useState<MatchFormat>("T20I");

  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submitChat(event: React.FormEvent) {
    event.preventDefault();

    setLoading(true);
    setError("");

    const payload: Record<string, unknown> = {
      intent,
    };

    if (intent === "predict_match") {
      payload.team1 = team1;
      payload.team2 = team2;
      payload.format = format;
      payload.mode = "PRE_TOSS";
      payload.date = new Date().toISOString().slice(0, 10);
    }

    try {
      const response = await sendChat(payload);
      setMessage(response.message);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Chat request failed.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page-container">
      <div className="page-heading">
        <span className="eyebrow">CRICMASTER AI</span>
        <h1>Ask Cricmaster</h1>
        <p>
          Structured cricket intelligence backed by the same
          production prediction engine.
        </p>
      </div>

      <div className="chat-shell card">
        <form onSubmit={submitChat}>
          <label>
            What do you want to do?
            <select
              value={intent}
              onChange={(e) =>
                setIntent(
                  e.target.value as
                    | "help"
                    | "model_capabilities"
                    | "predict_match",
                )
              }
            >
              <option value="help">Help</option>
              <option value="model_capabilities">
                Model capabilities
              </option>
              <option value="predict_match">
                Predict match
              </option>
            </select>
          </label>

          {intent === "predict_match" && (
            <div className="form-grid chat-fields">
              <label>
                Team 1
                <input
                  value={team1}
                  onChange={(e) => setTeam1(e.target.value)}
                />
              </label>

              <label>
                Team 2
                <input
                  value={team2}
                  onChange={(e) => setTeam2(e.target.value)}
                />
              </label>

              <label>
                Format
                <select
                  value={format}
                  onChange={(e) =>
                    setFormat(
                      e.target.value as MatchFormat,
                    )
                  }
                >
                  <option value="T20I">T20I</option>
                  <option value="T20">T20</option>
                </select>
              </label>
            </div>
          )}

          <button className="primary-button">
            {loading ? "Thinking..." : "Ask Cricmaster"}
          </button>
        </form>

        {message && (
          <div className="chat-message">
            <span>CRICMASTER</span>
            <p>{message}</p>
          </div>
        )}

        {error && <div className="error-box">{error}</div>}
      </div>
    </div>
  );
}

function HomePage({
  goToPredict,
}: {
  goToPredict: () => void;
}) {
  return (
    <div className="home">
      <section className="hero">
        <div className="hero-copy">
          <span className="eyebrow">
            CRICKET INTELLIGENCE ENGINE
          </span>

          <h1>
            Understand the match.
            <span>Not just the score.</span>
          </h1>

          <p>
            Cricmaster combines historical performance,
            playing-XI strength and live match context to
            produce transparent cricket win probabilities.
          </p>

          <div className="hero-actions">
            <button
              className="primary-button"
              onClick={goToPredict}
            >
              Predict a Match
            </button>

            <span className="supported">
              T20I • IPL • BBL • PSL • CPL • Live
            </span>
          </div>
        </div>

        <div className="hero-visual">
          <div className="prediction-preview">
            <span className="live-status">
              CRICMASTER MODEL
            </span>

            <div className="preview-match">
              <div>
                <small>INDIA</small>
                <strong>61%</strong>
              </div>

              <span>VS</span>

              <div>
                <small>AUSTRALIA</small>
                <strong>39%</strong>
              </div>
            </div>

            <div className="preview-confidence">
              <span>Estimated edge</span>
              <strong>MODERATE</strong>
            </div>

            <p>
              Probabilities are estimates, not guarantees.
            </p>
          </div>
        </div>
      </section>

      <section className="feature-grid">
        <article className="feature-card">
          <span>01</span>
          <h3>Pre-Toss Intelligence</h3>
          <p>
            Historical form, Elo, venue history and
            roster-aware franchise modelling.
          </p>
        </article>

        <article className="feature-card">
          <span>02</span>
          <h3>Post-Toss Analysis</h3>
          <p>
            Add toss information and playing XI context when
            the match gets closer.
          </p>
        </article>

        <article className="feature-card">
          <span>03</span>
          <h3>Live Win Probability</h3>
          <p>
            Analyse match state as runs, wickets, overs and
            chase pressure change.
          </p>
        </article>
      </section>
    </div>
  );
}

function App() {
  const [page, setPage] = useState<Page>("home");
  const [health, setHealth] =
    useState<HealthResponse | null>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  return (
    <div className="app">
      <header className="navbar">
        <button
          className="brand"
          onClick={() => setPage("home")}
        >
          <span className="brand-mark">C</span>
          <span>CRICMASTER</span>
        </button>

        <nav>
          <button onClick={() => setPage("predict")}>
            Predict
          </button>

          <button onClick={() => setPage("live")}>
            Live
          </button>

          <button onClick={() => setPage("chat")}>
            Cricmaster AI
          </button>
        </nav>

        <div className="api-status">
          <span
            className={
              health?.status === "ok"
                ? "status-dot online"
                : "status-dot"
            }
          />
          {health?.status === "ok"
            ? "Engine online"
            : "Engine offline"}
        </div>
      </header>

      <main>
        {page === "home" && (
          <HomePage
            goToPredict={() => setPage("predict")}
          />
        )}

        {page === "predict" && <PredictPage />}
        {page === "live" && <LivePage />}
        {page === "chat" && <ChatPage />}
      </main>

      <footer>
        <span>CRICMASTER</span>
        <p>
          Cricket probabilities are statistical estimates,
          not guaranteed outcomes.
        </p>
      </footer>
    </div>
  );
}

export default App;