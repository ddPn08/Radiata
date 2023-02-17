# Frontend

::: info
First, start the application and install the dependencies.
:::

## Start in debug mode

### Frontend and Server

Linux，Mac

```bash
source venv/bin/activate
```

```bash
# launch-user.sh
export UVICORN_ARGS="--port 8000 --reload"
export COMMANDLINE_ARGS="--skip-install --skip-build-frontend"
```

Windows

```bash
.\venv\Scripts\activate
```

```bash
# launch-user.bat
set UVICORN_ARGS=--port 8000 --reload
set COMMANDLINE_ARGS=--skip-install --skip-build-frontend
```

Re Build Frontend

```bash
cd frontend
pnpm build
```

### Frontend Only

```bash
cd frontend
pnpm build
pnpm vite
```

## If the API is changed

Run with the server running

```bash
cd frontend
pnpm openapi-gen-client
```
