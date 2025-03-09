REM Randy Direen, 8/12/2015
REM This batch file is used to create the nuget package.
REM Nuget packages are used in .Net/Mono as a easy way to include DLL assemblies. 
nuget pack VisionBridge.csproj -IncludeReferencedProjects -Prop Configuration=Release