# Demo: How to use it

This walkthrough shows a complete SIMI/SIMICE demo with the built-in central and two remotes.

Prereq: run the container (see DEPLOYMENT.md) and open http://localhost:8080.

Note: Throughout this guide, localhost URLs are examples for a local deployment. If you deploy to a server or cloud, replace `http://localhost:8080` with your actual host or domain (e.g., `https://your-domain.example`). Likewise, use `wss://` for WebSockets when your site is served over HTTPS.

## 1) Login as admin

- Visit http://localhost:8080/ (or your deployed host URL)
- You'll be redirected to /gui/login. Default password is `admin123` (override with ADMIN_PASSWORD).

## 2) Register and approve remote sites

- Go to "Sites" and create user accounts for your remote nodes: provide site name, institution, email, password.
- As admin, approve sites on "Admin > Users". This issues each site a unique `site_id` and JWT token.
- Approval generates a site_id and a JWT token. The approval page also sends an email if Gmail creds are configured.

Tip: You can also fetch site info via API under `/api/remote/info`.

## 3) Configure remotes (Remote 1 and 2)

- Open Remote 1: http://localhost:8080/remote1/ (demo container) or your remote's host URL
- Click Settings. Paste the central HTTP URL and token from the approval step:
  - HTTP URL: http://localhost:8080 (replace with your deployed host)
  - The page will derive the WebSocket URL (ws://localhost:8080). If using HTTPS, this becomes wss://your-domain.example.
  - Site ID: the one shown on approval
  - Token: the JWT string
- Save; repeat for Remote 2 (http://localhost:8080/remote2/ in the demo). Use the second site's credentials. In non-demo deployments, use the actual remote host URL.

Note: These `/remote1` and `/remote2` paths are provided by the demo container only. In a non-demo setup, each remote can run on its own host/port (e.g., http://remote-a:9001) and simply point to the central HTTP/WS URLs in Settings. Replace `localhost` with your deployed host name accordingly.
- The settings are persisted to `data/site_config.json`.

## 4) Create a job

- Central: "Jobs > Create".
- Choose algorithm:
  - SIMI: single target column
  - SIMICE: multiple target columns
- Fill parameters according to the JSON schema loaded from `config/SIMI.json` or `config/SIMICE.json`.
- Select participant site IDs that will join the job (e.g., the two remotes you approved).

## 5) Prepare data

- Download sample CSVs from Central "Demo" page: http://localhost:8080/demo (or your deployed host URL)
  - central CSV: `central/app/static/demo_data/*central.csv`
  - remote CSVs: use each remote UI to upload its dataset when starting a job
- Datasets contain numeric columns; SIMI/SIMICE require numeric data. The central Start page will report any non-numeric conversion issues.

## 6) Start and monitor

- Central: "Jobs > Start".
- Pick your created job, review parameters, and upload the central CSV.
- Press Start. Central will:
  - Wait until the selected remotes connect to `/ws/{site_id}`
  - Stream algorithm logs in real time to the page
  - Orchestrate SIMI/SIMICE calls via `MIDN_R_PY` modules

Remote side:
- Each remote lists assigned jobs on its Home/Jobs pages (pulled via `/api/remote/jobs`).
- Start the job by selecting it, filling any required fields shown, and uploading the remote CSV when prompted.
- The page shows live status; you can stop the job from the remote UI.

## 7) Results

- On success, central writes results to `central/app/static/results/job_<id>_<timestamp>/` and offers a downloadable zip.
- You can access raw outputs under that directory via the browser.

## Troubleshooting

- If a remote shows no jobs, ensure its Site ID and Token match an approved user and that the central URL is correct.
- If central Start page reports coercion errors, check your CSV column types and headers; keep them numeric.
- Behind HTTPS, central will honor X-Forwarded-Proto from Nginx so links remain https.
