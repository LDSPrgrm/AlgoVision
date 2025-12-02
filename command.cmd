:: ---------------------- WINDOWS CMD SECTION ----------------------
@echo off

:: Detect if we're on Windows CMD
if not "%OS%"=="" goto :windows

# ---------------------- LINUX SECTION ----------------------
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
echo "Linux: PYTHONPATH set to $PYTHONPATH"

# Run Python script
python "$SCRIPT_DIR/main.py"
# Don't exit (to avoid killing shells)
return 0 2>/dev/null || exit 0

# ---------------------- END LINUX SECTION ------------------


:windows
set "SCRIPT_DIR=%~dp0"

:: Remove trailing backslash for comparison
set "SCRIPT_DIR_CLEAN=%SCRIPT_DIR:~0,-1%"

:: Check if SCRIPT_DIR_CLEAN is already in PYTHONPATH
echo;%PYTHONPATH%; | findstr /i /c:";%SCRIPT_DIR_CLEAN%;" >nul
if errorlevel 1 (
    if defined PYTHONPATH (
        set "PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%"
    ) else (
        set "PYTHONPATH=%SCRIPT_DIR%"
    )
)

echo Windows: PYTHONPATH set to %PYTHONPATH%

:: Run Python script
python "%SCRIPT_DIR%main.py"

goto :eof
