# cop6616_parallel-computing

*runonce*
echo "PATH=/usr/local/cuda/bin/:$PATH" >> ~/.bash_profile
source ~/.bash_profile

sudo apt install libpng-dev
OR
wget http://prdownloads.sourceforge.net/libpng/libpng-1.6.44.tar.xz
tar xf libpng-1.6.44.tar.xz
cd libpng-1.6.44
./configure --prefix=$HOME/libpng
make
make install

*q2*
nvcc -lm q2.cu -o q2
./q2 1000000 100

*q3*
nvcc -lm -lpng q3.cu -o q3
OR
nvcc -lm q3.cu -o q3 -L$HOME/libpng/lib -I$HOME/libpng/include -lpng
./q3