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
export COMMANDLINE_ARGS="--skip-install"
```

Windows

```bash
.\venv\Scripts\activate
```

```bash
# launch-user.bat
set UVICORN_ARGS=--port 8000 --reload
set COMMANDLINE_ARGS=--skip-install
```

Re Build Frontend

```bash
cd frontend
pnpm build
```

## If the API is changed

Run with the server running

```bash
cd frontend
pnpm --prefix ./app openapi-gen-client
```
