@echo off
REM PolyVision dataset sync helper
REM Tracks data/complete with DVC and pushes to the uni iDrive.
REM Run from the repo root: scripts\dataset_sync.bat

cd /d "%~dp0\.."

echo.
echo === PolyVision dataset sync ===
echo Remote: J:\Science\Chemistry\RDMS\Juliane Simmchen group\Josh\2. Microplastic AI Project\working_dataset\dvc_cache
echo.

REM Check J: is accessible
if not exist "J:\Science\Chemistry\RDMS\Juliane Simmchen group\Josh\2. Microplastic AI Project\working_dataset" (
    echo ERROR: J: drive not found. Connect to the Strathclyde VPN / campus network and try again.
    exit /b 1
)

REM Stage any changes to data/complete under DVC
echo [1/3] Tracking data\complete with DVC...
dvc add data/complete
if errorlevel 1 ( echo DVC add failed. & exit /b 1 )

REM Commit the updated .dvc pointer to git
echo [2/3] Staging DVC pointer for git...
git add data/complete.dvc .gitignore
git diff --cached --quiet
if errorlevel 1 (
    git commit -m "dvc: update data/complete snapshot"
) else (
    echo    No git changes to commit.
)

REM Push data blobs to the iDrive
echo [3/3] Pushing data to iDrive...
dvc push
if errorlevel 1 ( echo DVC push failed. & exit /b 1 )

echo.
echo Done. Dataset backed up to iDrive.
