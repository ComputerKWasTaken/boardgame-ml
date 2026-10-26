import type { WatchEvent } from "./types";

/** Pong's end/reset notifications share a physics tick; don't add delay. */
export function nextReplayPosition(
  events: Pick<WatchEvent, "type" | "game">[],
  position: number,
): number {
  let next = Math.min(events.length, position + 1);
  if (events[0]?.game === "pong") {
    while (next < events.length && events[next].type !== "step") next += 1;
  }
  return next;
}
