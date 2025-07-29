setlocal

if "%~1"=="" (
    exit /b 1
)

if "%~2"=="" (
    exit /b 1
)

set CONFIG=%~1
set IMAGE_PATH=%~2

if /I not "%CONFIG%"=="Debug" if /I not "%CONFIG%"=="Release" (
    exit /b 1
)

set EXE_NAME=onnx_cpu_demo.exe
set EXE_PATH=build\bin\%CONFIG%\%EXE_NAME%

if not exist "%EXE_PATH%" (
    exit /b 1
)

"%EXE_PATH%" "%IMAGE_PATH%"

endlocal