setlocal

set CONFIG=%1
if "%CONFIG%"=="" (
    set CONFIG=Release
)

set EXE=build\bin\%CONFIG%\onnx_gpu_demo.exe

if exist %EXE% (
    echo Running %EXE%...
    %EXE%
) else (
    echo Executable not found! Please build first with build.bat %CONFIG%
)

endlocal