@echo off
REM t-SNE可视化脚本 - Windows版本

echo.
echo ===============================================
echo GDN 模型 t-SNE 可视化工具
echo ===============================================
echo.

REM 设置默认参数
set DATASET=swat
set PERPLEXITY=30
set N_ITER=1000

REM 接受命令行参数
if not "%1"=="" set DATASET=%1
if not "%2"=="" set PERPLEXITY=%2
if not "%3"=="" set N_ITER=%3

echo 参数设置:
echo   数据集: %DATASET%
echo   困惑度(perplexity): %PERPLEXITY%
echo   迭代次数(n_iter): %N_ITER%
echo.

REM 运行可视化脚本
python visualize_tsne.py ^
    --dataset %DATASET% ^
    --perplexity %PERPLEXITY% ^
    --n_iter %N_ITER% ^
    --output_dir "./tsne_results" ^
    --batch_size 256 ^
    --sample_size 5000

echo.
echo 可视化完成！
echo 结果保存在: ./tsne_results/
echo.
pause
