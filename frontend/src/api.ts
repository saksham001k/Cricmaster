import type {
  ChatResponse,
  HealthResponse,
  LiveMatch,
  PredictionRequest,
  PredictionResponse,
} from "./types";

const API_BASE =
  import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers ?? {}),
    },
  });

  let body: unknown = null;

  try {
    body = await response.json();
  } catch {
    body = null;
  }

  if (!response.ok) {
    const data =
      body && typeof body === "object"
        ? (body as Record<string, unknown>)
        : {};

    const message =
      typeof data.message === "string"
        ? data.message
        : typeof data.detail === "string"
          ? data.detail
          : `Request failed with status ${response.status}`;

    throw new Error(message);
  }

  return body as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export function predictMatch(
  payload: PredictionRequest,
): Promise<PredictionResponse> {
  return request<PredictionResponse>("/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getLiveMatches(): Promise<LiveMatch[]> {
  return request<LiveMatch[]>("/live/matches");
}

export function sendChat(
  payload: Record<string, unknown>,
): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}