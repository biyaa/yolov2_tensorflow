TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared core/re_org.cc -o re_org.so -fPIC -I $TF_INC -O2

