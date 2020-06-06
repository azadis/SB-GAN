#Download and modify the indoor portion of ADE20K dataset
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

cd $DIR
wget https://people.eecs.berkeley.edu/~sazadi/SBGAN/datasets/ADE_indoor.tar.gz
tar -xzvf ADE_indoor.tar.gz
rm ADE_indoor.tar.gz

cd ADE_indoor
mv ADE15c_indoor_img ../
mv ADE15c_indoor_lbl ../
mv ADE15c_indoor_val_img ../
mv ADE15c_indoor_val_lbl ../

mkdir ADE15c_indoor_train_im
mkdir ADE15c_indoor_train_lbl
mkdir ADE15c_indoor_val_im
mkdir ADE15c_indoor_val_lbl

ln -s ${DIR}/ADE15c_indoor_img/*/* ADE15c_indoor_train_im/
ln -s ${DIR}/ADE15c_indoor_lbl/*/* ADE15c_indoor_train_lbl/
ln -s ${DIR}/ADE15c_indoor_val_img/*/* ADE15c_indoor_val_im/
ln -s ${DIR}/ADE15c_indoor_val_lbl/*/* ADE15c_indoor_val_lbl/