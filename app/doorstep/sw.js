// ReturnShield AI Doorstep QC Service Worker
const CACHE_NAME = 'returnshield-doorstep-v1';
const STATIC_ASSETS = [
  '/doorstep/',
  '/doorstep/index.html',
  '/doorstep/styles.css',
  '/doorstep/app.js',
  '/doorstep/manifest.json',
  '/doorstep/icons/icon-192.png',
  '/doorstep/icons/icon-512.png'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(STATIC_ASSETS);
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
      );
    })
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  // Only cache GET requests for static UI assets; API POST calls go to network
  if (event.request.method !== 'GET' || event.request.url.includes('/returns/')) {
    return;
  }
  event.respondWith(
    caches.match(event.request).then((cached) => {
      return cached || fetch(event.request).catch(() => {
        return caches.match('/doorstep/index.html');
      });
    })
  );
});

