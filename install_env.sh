env_name=$1
if [ ! -d "./luke" ]; then
    echo "Please run this script in the root directory of the project"
    exit 1
fi
cd luke || exit
conda create -n ${env_name} python==3.8.13
source activate ${env_name} # 使用source而不是使用 conda，原因是source可以在shell脚本中使用

pip install poetry==1.1.11

poetry install
# 需要安装 nltk的一个工具
python -m nltk.downloader omw-1.4

cd ..
