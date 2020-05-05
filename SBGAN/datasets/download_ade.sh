#!/bin/bash -f

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
PDIR="$( dirname ${DIR} )"

cd $DIR

wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip
rm ADEChallengeData2016.zip

mkdir ADE_indoor
mkdir ADE_indoor/images
mkdir ADE_indoor/annotations
mkdir ADE_indoor/images/train
mkdir ADE_indoor/images/val
mkdir ADE_indoor/annotations/train
mkdir ADE_indoor/annotations/val

input="ade_indoor_im_info_train.txt"
while IFS= read -r line
do
	pth="$(cut -d' ' -f1 <<<"$line")"
	name="$(cut -d'/' -f5 <<<"$pth")"
	ln -s ${PDIR}/${pth} ADE_indoor/images/train/${name}
done < "$input"


input="ade_indoor_lbl_info_train.txt"
while IFS= read -r line
do
	pth="$(cut -d' ' -f1 <<<"$line")"
	name="$(cut -d'/' -f5 <<<"$pth")"
	ln -s ${PDIR}/${pth} ADE_indoor/annotations/train/${name}
done < "$input"


input="ade_indoor_im_info_val.txt"
while IFS= read -r line
do
	pth="$(cut -d' ' -f1 <<<"$line")"
	name="$(cut -d'/' -f5 <<<"$pth")"
	ln -s ${PDIR}/${pth} ADE_indoor/images/val/${name}
done < "$input"


input="ade_indoor_lbl_info_val.txt"
while IFS= read -r line
do
	pth="$(cut -d' ' -f1 <<<"$line")"
	name="$(cut -d'/' -f5 <<<"$pth")"
	ln -s ${PDIR}/${pth} ADE_indoor/annotations/val/${name}
done < "$input"
