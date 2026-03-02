const APP_CACHE = "wasiw-app-v1";
const DATA_CACHE = "wasiw-data-v1";
const RUNTIME_CACHE = "wasiw-runtime-v1";

const APP_SHELL = [
  "./",
  "./index.html",
  "./manifest.webmanifest",
  "./favicon.svg",
  "./icons/icon-192.png",
  "./icons/icon-512.png",
  "./icons/icon-512-maskable.png",
  "./icons/apple-touch-icon.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(APP_CACHE)
      .then((cache) => cache.addAll(APP_SHELL))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => ![APP_CACHE, DATA_CACHE, RUNTIME_CACHE].includes(key))
            .map((key) => caches.delete(key)),
        ),
      )
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") {
    return;
  }

  const requestUrl = new URL(event.request.url);
  const sameOrigin = requestUrl.origin === self.location.origin;

  if (event.request.mode === "navigate") {
    event.respondWith(networkFirst(event.request, APP_CACHE));
    return;
  }

  if (sameOrigin && requestUrl.pathname.includes("/data/")) {
    event.respondWith(cacheFirst(event.request, DATA_CACHE));
    return;
  }

  if (
    sameOrigin &&
    (requestUrl.pathname.endsWith(".js") ||
      requestUrl.pathname.endsWith(".css") ||
      requestUrl.pathname.endsWith(".svg") ||
      requestUrl.pathname.endsWith(".png") ||
      requestUrl.pathname.endsWith(".webmanifest"))
  ) {
    event.respondWith(staleWhileRevalidate(event.request, APP_CACHE));
    return;
  }

  if (!sameOrigin) {
    event.respondWith(staleWhileRevalidate(event.request, RUNTIME_CACHE));
  }
});

async function networkFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  try {
    const response = await fetch(request);
    if (response && response.ok) {
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    const cached = await cache.match(request);
    if (cached) {
      return cached;
    }
    const fallback = await cache.match("./index.html");
    if (fallback) {
      return fallback;
    }
    return new Response("Offline. Cached content is unavailable.", {
      status: 503,
      statusText: "Offline",
      headers: {
        "Content-Type": "text/plain; charset=utf-8",
      },
    });
  }
}

async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  if (cached) {
    return cached;
  }
  const response = await fetch(request);
  if (response && response.ok) {
    cache.put(request, response.clone());
  }
  return response;
}

async function staleWhileRevalidate(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  const fetchPromise = fetch(request)
    .then((response) => {
      if (response && response.ok) {
        cache.put(request, response.clone());
      }
      return response;
    })
    .catch(() => null);

  if (cached) {
    return cached;
  }

  const response = await fetchPromise;
  if (response) {
    return response;
  }
  throw new Error("Network request failed and no cached response exists.");
}
