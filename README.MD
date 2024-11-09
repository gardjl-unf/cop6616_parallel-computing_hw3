# cop6616_parallel-computing

*runonce*
echo "PATH=/usr/local/cuda/bin/:$PATH" >> ~/.bash_profile
source ~/.bash_profile

sudo apt install libpng-dev
OR (NON-SUDO)
wget http://prdownloads.sourceforge.net/libpng/libpng-1.6.44.tar.xz
tar xf libpng-1.6.44.tar.xz
cd libpng-1.6.44
./configure --prefix=$HOME/lib/libpng
make
make install

*q2*
nvcc -lm q2.cu -o q2
./q2 1000000 100

*q3*
nvcc -lm -lpng q3.cu -o q3
OR (NON-SUDO)
nvcc -lm q3.cu -o q3 -L$HOME/lib/libpng/lib -I$HOME/lib/libpng/include -lpng
./q3

*q4*
nvcc -lm q4.cu -o q4
./q4 1000 1000 1000 10