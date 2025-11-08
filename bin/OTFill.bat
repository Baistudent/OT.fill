@echo off
setlocal EnableDelayedExpansion

call :get_timestamp start_all
set "total_elapsed=0"

@REM 设置程序路径, conda环境名
set "texture_to_mesh=..\python.code\texture_to_mesh_01.py"
set "texture_mapping=..\python.code\texture_mapping_02.py"
set "remap_obj_uv=..\python.code\remap_obj_uv_03.py"
set "condaEnv=atlas"

set "OT=..\cpp.code\build\OT\Release\OT.exe"
set "square=..\data\square.obj"

@REM 其他参数(不做输入)
set "threshold=0.1"
set "step_length=0.05"

@REM 获取输入参数
set "objFile=%~f1"
set "objName=%~n1"
set "objPath=%~dp1"
set "textureFile=%~f2"
set "resolution=%~3"
set "scale=%~4"

@REM ========================================================================================
call conda activate %condaEnv%

echo ===================纹理网格化===================
call :get_timestamp stage_mesh_start
set "meshFile=%objPath%%objName%_%resolution%.m"
python "%texture_to_mesh%" "%objFile%" "%textureFile%" "%meshFile%" %resolution%
call :get_timestamp stage_mesh_end
call :calc_elapsed stage_mesh_start stage_mesh_end elapsed_mesh
call :format_elapsed elapsed_mesh elapsed_mesh_str
set /a total_elapsed+=elapsed_mesh

echo ====================最优传输====================
call :get_timestamp stage_ot_start
set "meshOTFile=%objPath%%objName%_%resolution%_%scale%.m"
"%OT%" -scale "%scale%" -threshold "%threshold%" -source "%square%" -target "%meshFile%" -output "%meshOTFile%" -step_length "%step_length%" -key @mWq
call :get_timestamp stage_ot_end
call :calc_elapsed stage_ot_start stage_ot_end elapsed_ot
call :format_elapsed elapsed_ot elapsed_ot_str
set /a total_elapsed+=elapsed_ot

echo ====================纹理映射====================
call :get_timestamp stage_map_start
set "resTextureFile=%objPath%%objName%_%resolution%_%scale%.png"
python "%texture_mapping%" "%meshOTFile%" "%textureFile%" "%resTextureFile%"
call :get_timestamp stage_map_end
call :calc_elapsed stage_map_start stage_map_end elapsed_map
call :format_elapsed elapsed_map elapsed_map_str
set /a total_elapsed+=elapsed_map

echo ====================生成网格====================
call :get_timestamp stage_obj_start
set "resObjFile=%objPath%%objName%_%resolution%_%scale%.obj"
python "%remap_obj_uv%" "%objFile%" "%meshOTFile%" "%resObjFile%" "%resTextureFile%"
call :get_timestamp stage_obj_end
call :calc_elapsed stage_obj_start stage_obj_end elapsed_obj
call :format_elapsed elapsed_obj elapsed_obj_str
set /a total_elapsed+=elapsed_obj

@REM if exist "%meshOTFile%" del /q "%meshOTFile%"
if exist "%meshFile%" del /q "%meshFile%"

call :get_timestamp end_all
call :calc_elapsed start_all end_all elapsed_all
call :format_elapsed total_elapsed total_elapsed_str
call :format_elapsed elapsed_all elapsed_all_str

echo.
echo 耗时统计:
echo  - 纹理网格化: !elapsed_mesh_str!
echo  - 最优传输: !elapsed_ot_str!
echo  - 纹理映射: !elapsed_map_str!
echo  - 生成网格: !elapsed_obj_str!
echo  - 阶段总计: !total_elapsed_str!
echo  - 全流程耗时: !elapsed_all_str!
echo ====================全部完成=====================

endlocal
goto :eof

:get_timestamp
for /f "tokens=1-4 delims=:." %%a in ("%time%") do (
	set "hh=%%a"
	set "mm=%%b"
	set "ss=%%c"
	set "cs=%%d"
)
set "hh=0%hh%"
set "hh=%hh:~-2%"
set "mm=0%mm%"
set "mm=%mm:~-2%"
set "ss=0%ss%"
set "ss=%ss:~-2%"
set "cs=0%cs%"
set "cs=%cs:~-2%"
set /a total=((1%hh%-100)*3600 + (1%mm%-100)*60 + (1%ss%-100))*100 + (1%cs%-100)
set "%~1=%total%"
exit /b

:calc_elapsed
set /a temp=!%2!-!%1!
if !temp! lss 0 set /a temp+=8640000
set "%3=!temp!"
exit /b

:format_elapsed
set "total=!%1!"
set /a seconds=total/100
set /a centi=total%%100
set "centi=0%centi%"
set "centi=%centi:~-2%"
set "%2=%seconds%.%centi%s"
exit /b