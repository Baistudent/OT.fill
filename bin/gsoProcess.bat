@echo off
setlocal EnableDelayedExpansion

if "%~1"=="" (
    echo Usage: gsoProcess.bat {rootDir}
    exit /b 1
)

set "rootDir=%~f1"
if not exist "%rootDir%" (
    echo [ERROR] Not found: %rootDir%
    exit /b 1
)

set "resolution=256"
set "scale=0.01"

set /a totalTargets=0
for /d %%D in ("%rootDir%\*") do (
    set "subDir=%%~fD"
    if exist "!subDir!\model.obj" (
        if exist "!subDir!\texture.png" (
            set /a totalTargets+=1
        )
    )
)

if !totalTargets! leq 0 (
    echo [Warning] No subdirectories to process.
    goto :eof
)

set /a currentIndex=0
for /d %%D in ("%rootDir%\*") do (
    set "subDir=%%~fD"
    set "objFile=!subDir!\model.obj"
    set "textureFile=!subDir!\texture.png"
    set "skipDir="

    if not exist "!objFile!" (
        echo [Warning] Skipping !subDir! - missing model.obj
        set "skipDir=1"
    )

    if not exist "!textureFile!" (
        echo [Warning] Skipping !subDir! - missing texture.png
        set "skipDir=1"
    )

    if not defined skipDir (
        set /a currentIndex+=1
        set "folderName=%%~nD"
        for %%F in ("!objFile!") do set "objBase=%%~nF"
        if not defined objBase set "objBase=!folderName!"
        set "logFilePath=!subDir!\!objBase!_!resolution!_!scale!.log"
        call "%~dp0OTFill.bat" "!objFile!" "!textureFile!" %resolution% %scale% >"!logFilePath!" 2>&1
        if errorlevel 1 (
            echo [Error] !subDir! processing failed - log: !logFilePath!
        ) else (
            call :timestamp tsDisplay
            echo [Success] !tsDisplay! [!currentIndex!/!totalTargets!] !folderName! - log: !logFilePath!
        )
    )
)

endlocal
goto :eof

:timestamp
setlocal EnableDelayedExpansion
set "ldt="
for /f "tokens=2 delims==." %%a in ('wmic os get LocalDateTime /value ^| find "="') do if not defined ldt set "ldt=%%a"
if defined ldt (
    set "yyyy=!ldt:~0,4!"
    set "mm=!ldt:~4,2!"
    set "dd=!ldt:~6,2!"
    set "hh=!ldt:~8,2!"
    set "mi=!ldt:~10,2!"
    set "ss=!ldt:~12,2!"
) else (
    for /f "tokens=1-4 delims=:." %%a in ("%time%") do (
        set "hh=%%a"
        set "mi=%%b"
        set "ss=%%c"
    )
    set "hh=0!hh!"
    set "hh=!hh:~-2!"
    set "mi=0!mi!"
    set "mi=!mi:~-2!"
    set "ss=0!ss!"
    set "ss=!ss:~-2!"
    set "yyyy=%date:~0,4%"
    set "mm=%date:~5,2%"
    set "dd=%date:~8,2%"
)
set "friendly=!yyyy!/!mm!/!dd! !hh!:!mi!:!ss!"
set "safe=!yyyy!-!mm!-!dd!_!hh!-!mi!-!ss!"
endlocal & set "%~1=%friendly%" & (if not "%~2"=="" set "%~2=%safe%")
exit /b