param(
  [string]$PythonCmd = "py -3"
)

Write-Host "[1/4] 创建虚拟环境 .venv"
Invoke-Expression "$PythonCmd -m venv .venv"

Write-Host "[2/4] 激活虚拟环境"
& .\.venv\Scripts\Activate.ps1

Write-Host "[3/4] 升级 pip"
python -m pip install -U pip

Write-Host "[4/4] 安装依赖"
python -m pip install -r requirements.txt

Write-Host "完成。可运行：python -m src.main --scenario s1_close_threat --steps 20"
