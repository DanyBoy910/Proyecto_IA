@echo off
REM Bucle para ejecutar el comando con diferentes valores de "-f"
FOR %%F IN ("O3") DO (
    FOR %%P IN (10:0.05 20:0.1 0:0.01 60:0.0001) DO (
        FOR /F "tokens=1,2 delims=:" %%A IN ("%%P") DO (
            FOR %%H IN (1, 2) DO (
                echo Ejecutando: python main.py train %%F PATIENCE=%%A WEIGHT_DECAY=%%B LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=128_64_32 prueba_%%H >> resultados.txt
                timeout /t 1 /nobreak >nul
                echo Ejecutando: python main.py train %%F PATIENCE=%%A WEIGHT_DECAY=%%B LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=128_64_32 prueba_%%H
                python main.py train %%F PATIENCE=%%A WEIGHT_DECAY=%%B LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=128_64_32
                echo.
            )
        )
    )
    FOR %%P IN (10:0.0001 20:0.0002 0:0.002 60:0.005) DO (
        FOR /F "tokens=1,2 delims=:" %%A IN ("%%P") DO (
            FOR %%H IN (1, 2) DO (
                echo Ejecutando: python main.py train %%F PATIENCE=%%A WEIGHT_DECAY=0.01 LEARNING_RATE=%%B BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=128_64_32 prueba_%%H >> resultados.txt
                timeout /t 1 /nobreak >nul
                echo Ejecutando: python main.py train %%F PATIENCE=%%A WEIGHT_DECAY=0.01 LEARNING_RATE=%%B BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=128_64_32 prueba_%%H
                python main.py train %%F PATIENCE=%%A WEIGHT_DECAY=0.01 LEARNING_RATE=%%B BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=128_64_32
                echo.
            )
        )
    )
    FOR %%P IN (10:16 20:8 0:512 60:64) DO (
        FOR /F "tokens=1,2 delims=:" %%A IN ("%%P") DO (
            FOR %%H IN (1, 2) DO (
                echo Ejecutando: python main.py train %%F PATIENCE=%%A WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=%%B DROPOUT=0.2 HIDDEN_SIZES=128_64_32 prueba_%%H >> resultados.txt
                timeout /t 1 /nobreak >nul
                echo Ejecutando: python main.py train %%F PATIENCE=%%A WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=%%B DROPOUT=0.2 HIDDEN_SIZES=128_64_32 prueba_%%H
                python main.py train %%F PATIENCE=%%A WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=%%B DROPOUT=0.2 HIDDEN_SIZES=128_64_32
                echo.
            )
        )
    )
    FOR %%P IN (10:0.4 20:0.5 0:0.3 60:0.6) DO (
        FOR /F "tokens=1,2 delims=:" %%A IN ("%%P") DO (
            FOR %%H IN (1, 2) DO (
                echo Ejecutando: python main.py train %%F PATIENCE=%%A WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=%%B HIDDEN_SIZES=128_64_32 prueba_%%H >> resultados.txt
                timeout /t 1 /nobreak >nul
                echo Ejecutando: python main.py train %%F PATIENCE=%%A WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=%%B HIDDEN_SIZES=128_64_32 prueba_%%H
                python main.py train %%F PATIENCE=%%A WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=%%B HIDDEN_SIZES=128_64_32
                echo.
            )
        )
    )
    FOR %%P IN (10:512_256_128 20:32_16_8 0:1024_512_256 60:128_64_32) DO (
        FOR /F "tokens=1,2 delims=:" %%A IN ("%%P") DO (
            FOR %%H IN (1, 2) DO (
                echo Ejecutando: python main.py train %%F PATIENCE=%%A WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=%%B prueba_%%H >> resultados.txt
                timeout /t 1 /nobreak >nul
                echo Ejecutando: python main.py train %%F PATIENCE=%%A WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=%%B prueba_%%H
                python main.py train %%F PATIENCE=%%A WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=%%B
                echo.
            )
        )
    )
    FOR %%P IN (0.05:0.0001 0.1:0.0002 0.01:0.002 0.0001:0.005) DO (
        FOR /F "tokens=1,2 delims=:" %%A IN ("%%P") DO (
            FOR %%H IN (1, 2) DO (
                echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=%%A LEARNING_RATE=%%B BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=128_64_32 prueba_%%H >> resultados.txt
                timeout /t 1 /nobreak >nul
                echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=%%A LEARNING_RATE=%%B BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=128_64_32 prueba_%%H
                python main.py train %%F PATIENCE=50 WEIGHT_DECAY=%%A LEARNING_RATE=%%B BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=128_64_32
                echo.
            )
        )
    )
    FOR %%P IN (0.05:16 0.1:8 0.01:512 0.0001:64) DO (
        FOR /F "tokens=1,2 delims=:" %%A IN ("%%P") DO (
            FOR %%H IN (1, 2) DO (
                echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=%%A LEARNING_RATE=0.002 BATCH_SIZE=%%B DROPOUT=0.2 HIDDEN_SIZES=128_64_32 prueba_%%H >> resultados.txt
                timeout /t 1 /nobreak >nul
                echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=%%A LEARNING_RATE=0.002 BATCH_SIZE=%%B DROPOUT=0.2 HIDDEN_SIZES=128_64_32 prueba_%%H
                python main.py train %%F PATIENCE=50 WEIGHT_DECAY=%%A LEARNING_RATE=0.002 BATCH_SIZE=%%B DROPOUT=0.2 HIDDEN_SIZES=128_64_32
                echo.
            )
        )
    )
    FOR %%P IN (0.05:0.4 0.1:0.5 0.01:0.3 0.0001:0.6) DO (
        FOR /F "tokens=1,2 delims=:" %%A IN ("%%P") DO (
            FOR %%H IN (1, 2) DO (
                echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=%%A LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=%%B HIDDEN_SIZES=128_64_32 prueba_%%H >> resultados.txt
                timeout /t 1 /nobreak >nul
                echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=%%A LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=%%B HIDDEN_SIZES=128_64_32 prueba_%%H
                python main.py train %%F PATIENCE=50 WEIGHT_DECAY=%%A LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=%%B HIDDEN_SIZES=128_64_32
                echo.
            )
        )
    )
    FOR %%P IN (0.05:512_256_128 0.1:32_16_8 0.01:1024_512_256 0.0001:128_64_32) DO (
        FOR /F "tokens=1,2 delims=:" %%A IN ("%%P") DO (
            FOR %%H IN (1, 2) DO (
                echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=%%A LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=%%B prueba_%%H >> resultados.txt
                timeout /t 1 /nobreak >nul
                echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=%%A LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=%%B prueba_%%H
                python main.py train %%F PATIENCE=50 WEIGHT_DECAY=%%A LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=%%B
                echo.
            )
        )
    )
    FOR %%P IN (0.0001:16 0.0002:8 0.002:512 0.005:64) DO (
        FOR /F "tokens=1,2 delims=:" %%A IN ("%%P") DO (
            FOR %%H IN (1, 2) DO (
                echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=%%A BATCH_SIZE=%%B DROPOUT=0.2 HIDDEN_SIZES=128_64_32 prueba_%%H >> resultados.txt
                timeout /t 1 /nobreak >nul
                echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=%%A BATCH_SIZE=%%B DROPOUT=0.2 HIDDEN_SIZES=128_64_32 prueba_%%H
                python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=%%A BATCH_SIZE=%%B DROPOUT=0.2 HIDDEN_SIZES=128_64_32
                echo.
            )
        )
    )
    FOR %%P IN (0.0001:0.4 0.0002:0.5 0.002:0.3 0.005:0.6) DO (
        FOR /F "tokens=1,2 delims=:" %%A IN ("%%P") DO (
            FOR %%H IN (1, 2) DO (
                echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=%%A BATCH_SIZE=64 DROPOUT=%%B HIDDEN_SIZES=128_64_32 prueba_%%H >> resultados.txt
                timeout /t 1 /nobreak >nul
                echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=%%A BATCH_SIZE=64 DROPOUT=%%B HIDDEN_SIZES=128_64_32 prueba_%%H
                python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=%%A BATCH_SIZE=64 DROPOUT=%%B HIDDEN_SIZES=128_64_32
                echo.
            )
        )
    )
    FOR %%P IN (0.0001:512_256_128 0.0002:32_16_8 0.002:1024_512_256 0.005:128_64_32) DO (
        FOR /F "tokens=1,2 delims=:" %%A IN ("%%P") DO (
            FOR %%H IN (1, 2) DO (
                echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=%%A BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=%%B prueba_%%H >> resultados.txt
                timeout /t 1 /nobreak >nul
                echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=%%A BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=%%B prueba_%%H
                python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=%%A BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=%%B
                echo.
            )
        )
    )
    FOR %%P IN (16:0.4 8:0.5 512:0.3 64:0.6) DO (
        FOR /F "tokens=1,2 delims=:" %%A IN ("%%P") DO (
            FOR %%H IN (1, 2) DO (
                echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=%%A DROPOUT=%%B HIDDEN_SIZES=128_64_32 prueba_%%H >> resultados.txt
                timeout /t 1 /nobreak >nul
                echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=%%A DROPOUT=%%B HIDDEN_SIZES=128_64_32 prueba_%%H
                python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=%%A DROPOUT=%%B HIDDEN_SIZES=128_64_32
                echo.
            )
        )
    )
    FOR %%P IN (16:512_256_128 8:32_16_8 512:1024_512_256 64:128_64_32) DO (
        FOR /F "tokens=1,2 delims=:" %%A IN ("%%P") DO (
            FOR %%H IN (1, 2) DO (
                echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=%%A DROPOUT=0.2 HIDDEN_SIZES=%%B prueba_%%H >> resultados.txt
                timeout /t 1 /nobreak >nul
                echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=%%A DROPOUT=0.2 HIDDEN_SIZES=%%B prueba_%%H
                python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=%%A DROPOUT=0.2 HIDDEN_SIZES=%%B
                echo.
            )
        )
    )
    FOR %%P IN (0.4:512_256_128 0.5:32_16_8 0.3:1024_512_256 0.6:128_64_32) DO (
        FOR /F "tokens=1,2 delims=:" %%A IN ("%%P") DO (
            FOR %%H IN (1, 2) DO (
                echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=%%A HIDDEN_SIZES=%%B prueba_%%H >> resultados.txt
                timeout /t 1 /nobreak >nul
                echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=%%A HIDDEN_SIZES=%%B prueba_%%H
                python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=%%A HIDDEN_SIZES=%%B
                echo.
            )
        )
    )
)

@REM @echo off
@REM REM Bucle para ejecutar el comando con diferentes valores de "-f"
@REM FOR %%F IN ("P2.5", "O3", "CO") DO (
@REM     FOR %%G IN (0, 10, 20, 30, 40, 50, 60, 70) DO (
@REM         FOR %%H IN (1, 2, 3) DO (
@REM             echo Ejecutando: python main.py train %%F PATIENCE=%%G WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=128_64_32 prueba_%%H >> resultados.txt
@REM             timeout /t 1 /nobreak >nul
@REM             echo Ejecutando: python main.py train %%F PATIENCE=%%G WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=128_64_32 prueba_%%H
@REM             python main.py train %%F PATIENCE=%%G WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=128_64_32
@REM             echo.
@REM         )
@REM     )
@REM     FOR %%G IN (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5) DO (
@REM         FOR %%H IN (1, 2, 3) DO (
@REM             echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=%%G LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=128_64_32 prueba_%%H >> resultados.txt
@REM             timeout /t 1 /nobreak >nul
@REM             echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=%%G LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=128_64_32 prueba_%%H
@REM             python main.py train %%F PATIENCE=50 WEIGHT_DECAY=%%G LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=128_64_32
@REM             echo.
@REM         )
@REM     )
@REM     FOR %%G IN (0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02) DO (
@REM         FOR %%H IN (1, 2, 3) DO (
@REM             echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=%%G BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=128_64_32 prueba_%%H >> resultados.txt
@REM             timeout /t 1 /nobreak >nul
@REM             echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=%%G BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=128_64_32 prueba_%%H
@REM             python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=%%G BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=128_64_32
@REM             echo.
@REM         )
@REM     )
@REM     FOR %%G IN (8, 16, 32, 64, 128, 256, 512, 1024) DO (
@REM         FOR %%H IN (1, 2, 3) DO (
@REM             echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=%%G DROPOUT=0.2 HIDDEN_SIZES=128_64_32 prueba_%%H >> resultados.txt
@REM             timeout /t 1 /nobreak >nul
@REM             echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=%%G DROPOUT=0.2 HIDDEN_SIZES=128_64_32 prueba_%%H
@REM             python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=%%G DROPOUT=0.2 HIDDEN_SIZES=128_64_32
@REM             echo.
@REM         )
@REM     )
@REM     FOR %%G IN (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7) DO (
@REM         FOR %%H IN (1, 2, 3) DO (
@REM             echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=%%G HIDDEN_SIZES=128_64_32 prueba_%%H >> resultados.txt
@REM             timeout /t 1 /nobreak >nul
@REM             echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=%%G HIDDEN_SIZES=128_64_32 prueba_%%H
@REM             python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=%%G HIDDEN_SIZES=128_64_32
@REM             echo.
@REM         )
@REM     )
@REM     FOR %%G IN (8_4_2, 16_8_4, 32_16_8, 64_32_16, 128_64_32, 256_128_64, 512_256_128, 1024_512_256) DO (
@REM         FOR %%H IN (1, 2, 3) DO (
@REM             echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=%%G prueba_%%H >> resultados.txt
@REM             timeout /t 1 /nobreak >nul
@REM             echo Ejecutando: python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=%%G prueba_%%H
@REM             python main.py train %%F PATIENCE=50 WEIGHT_DECAY=0.01 LEARNING_RATE=0.002 BATCH_SIZE=64 DROPOUT=0.2 HIDDEN_SIZES=%%G
@REM             echo.
@REM         )
@REM     )
@REM )