# https://drive.google.com/file/d/1S_4Z1i5vwU3nFya08UYlrvpUFFkl-r1f/view?usp=share_link
# https://drive.google.com/file/d/1C-K9V03plCPrv5k3lvrwVLxCkm-WPlk5/view?usp=share_link


fileid="1S_4Z1i5vwU3nFya08UYlrvpUFFkl-r1f"
filename="dataset_20_D1.tar.gz"

# fileid=1C-K9V03plCPrv5k3lvrwVLxCkm-WPlk5
# filename=dataset_20_D2.tar.gz

html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}