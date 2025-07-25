setlocal

set TOOLCHAIN=D:/vcpkg/scripts/buildsystems/vcpkg.cmake

set CONFIG=%1
if "%CONFIG%"=="" (
    set CONFIG=Release
)

echo [INFO] Build configuration: %CONFIG%

set BUILD_DIR=build

if not exist %BUILD_DIR% (
    mkdir %BUILD_DIR%
)

cd %BUILD_DIR%

cmake .. -DCMAKE_TOOLCHAIN_FILE=%TOOLCHAIN% -DCMAKE_BUILD_TYPE=%CONFIG%

cmake --build . --config %CONFIG%

cd ..
endlocal