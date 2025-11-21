@echo off

set PYTHON=C:\Users\Bashar\AppData\Local\Programs\Python\Python310\python.exe
set COMMANDLINE_ARGS=--skip-torch-cuda-test --use-cpu all --no-half --no-half-vae --precision full
set VENV_DIR=

call webui.bat
