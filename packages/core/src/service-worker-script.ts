/**
 * Service worker script entry point
 * This file should be bundled and served as the service worker
 */
import { ServiceWorkerController } from "./service-worker.js";
import type { ServiceWorkerMessage, ServiceWorkerResponse } from "./types.js";

// Create controller instance
const controller = new ServiceWorkerController();

// Handle messages from main thread
self.addEventListener("message", async (event: MessageEvent) => {
  const message = event.data as ServiceWorkerMessage;
  const port = event.ports[0];

  if (!port) {
    return;
  }

  try {
    const response = await controller.handleMessage(message);
    port.postMessage(response);
  } catch (error) {
    port.postMessage({
      type: "ERROR",
      error: error instanceof Error ? error.message : String(error),
    } as ServiceWorkerResponse);
  }
});

// Handle service worker install
self.addEventListener("install", (event: Event) => {
  const e = event as ExtendableEvent;
  const swSelf = self as unknown as ServiceWorkerGlobalScope;
  e.waitUntil(swSelf.skipWaiting());
});

// Handle service worker activate
self.addEventListener("activate", (event: Event) => {
  const e = event as ExtendableEvent;
  const swSelf = self as unknown as ServiceWorkerGlobalScope;
  e.waitUntil(swSelf.clients.claim());
});
