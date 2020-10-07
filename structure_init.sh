mkdir ./data
mkdir ./data/archive
mkdir ./data/features

curl https://pjreddie.com/media/files/mnist_train.csv --output ./data/archive/mnist_train.csv
curl https://pjreddie.com/media/files/mnist_test.csv --output ./data/archive/mnist_test.csv
