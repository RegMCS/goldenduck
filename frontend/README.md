# GoldenDuck Frontend

This is a [Next.js](https://nextjs.org/) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

### First time setup

Install dependencies:

```bash
npm install
```

### Start the frontend

```bash
cd frontend
docker compose up --build
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Run locally (without Docker)

```bash
npm run dev
```

Ensure the backend is running and set `BACKEND_SERVICE_URL` if needed (e.g. in `.env.local`):

```bash
BACKEND_SERVICE_URL=http://localhost:8000
```

## Environment variables

| Variable | Description |
|----------|-------------|
| `BACKEND_SERVICE_URL` | Backend API base URL. When running in Docker this is set to `http://host.docker.internal:8000`. For local dev, use `http://localhost:8000` in `.env.local`. |
| `NODE_ENV` | Set to `development` when running `npm run dev`. In development the frontend will work without logging in if the backend is also in dev mode (see below). |

## Dev mode: authentication optional

When the **backend** is run with `NODE_ENV=development`, authentication is not required:

- The frontend will automatically get a “current user” (the first user in the backend DB) when you load the app, so you can use Generate and other features without logging in.
- Ensure at least one user exists (e.g. register once via the UI or the backend’s `POST /api/auth/register`).

For production, run the backend with `NODE_ENV=production`.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.
