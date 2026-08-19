export type MatchFormat = "T20I" | "T20";
export type PredictionMode = "PRE_TOSS" | "POST_TOSS" | "LIVE";

export interface Driver {
  feature: string;
  label: string;
  raw_difference?: number | null;
  contribution?: number;
  supports?: string;
}

export interface PredictionRequest {
  team1: string;
  team2: string;
  format: MatchFormat;
  mode: PredictionMode;
  date: string;

  competition?: string;
  venue?: string;

  toss_winner?: string;
  toss_decision?: "bat" | "field";
  team1_xi?: string[];
  team2_xi?: string[];

  batting_team?: string;
  innings?: number;
  runs?: number;
  wickets?: number;
  overs?: string;
  target?: number;
}

export interface PredictionResponse {
  team1: string;
  team2: string;
  team1_probability: number;
  team2_probability: number;
  predicted_team: string;
  edge: string;
  confidence: "LOW" | "MEDIUM" | "HIGH";
  prediction_mode: string;
  format: string;
  model_name: string;
  model_family: string;

  warnings?: string[];
  drivers?: Driver[];

  matches_applied?: number;
  team1_history_matches?: number;
  team2_history_matches?: number;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
}

export interface LiveMatch {
  match_id: string;
  name?: string;
  format?: string;
  status?: string;
  teams?: string[];
}

export interface ChatResponse {
  intent: string;
  message: string;
  prediction?: PredictionResponse | null;
  suggestions?: string[];
}