# 判断/home/data_91_d是否挂载
if [ ! -d "/home/data_91_d" ]; then
    echo "Please mount /home/data_91_d"
    exit 1
fi

echo "Creating symbolic links to data"
rm -rf data
ln -s /home/data_91_d/chuyg/mix-debias/data data