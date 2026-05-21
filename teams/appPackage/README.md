# Teams tab package (static HTML)

This folder holds a minimal Teams app manifest for sideloading the co-pilot UI (`app/frontend/index.html`).

## Before packaging

1. Host the frontend and API over HTTPS (Teams requires a reachable `contentUrl`; localhost works only in Teams **debug** sideload scenarios).
2. Replace `YOUR_HOSTED_URL` in `manifest.json` with your deployed origin.
3. Add required icons next to this file:
   - `color.png` — 192×192
   - `outline.png` — 32×32 transparent outline

Zip `manifest.json`, `color.png`, and `outline.png`, then upload via **Teams → Apps → Manage your apps → Upload a custom app**.

## Auth without Azure

Configure the same `API_SECRET_KEY` on the server and in the tab’s **API key** field in the Connection panel.
