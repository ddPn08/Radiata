@echo off

if exist ".git" (
    git fetch --prune
    git reset --hard origin/main
) else (
    git init
    git remote add origin https://github.com/ddPn08/Radiata.git
    git fetch --prune
    git reset --hard origin/main
)

echo "Update complete."

pause
