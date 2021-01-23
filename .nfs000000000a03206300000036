#!/bin/bash

# check if data is downloaded use wget
# train data links
TRAINLABEL="https://s3.eu-central-1.wasabisys.com/aicrowd-public-datasets/snakes-challenge/round-4/train_labels.tar.gz"
TRAINIMG="https://s3.eu-central-1.wasabisys.com/aicrowd-public-datasets/snakes-challenge/round-4/train_images.tar.gz"

# test data links
TESTLABEL="https://s3.eu-central-1.wasabisys.com/aicrowd-public-datasets/snakes-challenge/round-4/validate_labels_small.tar.gz"
TESTIMG="https://s3.eu-central-1.wasabisys.com/aicrowd-public-datasets/snakes-challenge/round-4/validate_images.tar.gz"

# download directory
DOWNLOADDIR=../teams/DSC180A_FA20_A00/a01capstonegroup06/

while true; do
    read -p "Do you wish to install this program (approximately 50 GB for downloading the data)?" yn
    case $yn in
        [Yy]* ) 
            read -p "Installing to directory $DOWNLOADDIR, would you like to change directory?" yn
            case $yn in 
                [Yy]* )
                    echo "Enter new directory to download to: "
                    read NEWDOWNLOADDIR
                    echo "Downloading to $NEWDOWNLOADDIR"
                    
                    echo "Downloading train_labels.tar.gz from $TRAINLABEL..."
                    wget -nc -P $NEWDOWNLOADDIR $TRAINLABEL
                    echo "Downloading train_images.tar.gz from $TRAINIMG..."
                    wget -nc -P $NEWDOWNLOADDIR $TRAINIMG
                    echo "Downloading validate_labels.tar.gz from $TESTLABEL..."
                    wget -nc -P $NEWDOWNLOADDIR $TESTLABEL
                    echo "Downloading validate_images.tar.gz from $TESTIMG..."
                    wget -nc -P $NEWDOWNLOADDIR $TESTIMG
                    
                    # extract all
                    for f in $NEWDOWNLOADDIR*.tar.gz
                    do
                    tar zxvf "$f" -C $NEWDOWNLOADDIR
                    done
                    
                    exit
                    ;;
                [Nn]* )
                    # downloading data
                    echo "Downloading train_labels.tar.gz from $TRAINLABEL..."
                    wget -nc -P $DOWNLOADDIR $TRAINLABEL
                    wget -nc -P $DOWNLOADDIR $TRAINIMG
                    wget -nc -P $DOWNLOADDIR $TESTLABEL
                    wget -nc -P $DOWNLOADDIR $TESTIMG

                    # extract all
                    for f in $DOWNLOADDIR*.tar.gz
                    do
                    tar zxvf "$f" -C $DOWNLOADDIR
                    done

                    exit
                    ;;
                * ) 
                echo "Please answer yes or no."
                ;;
            esac
            ;;
        [Nn]* ) 
        exit
        ;;
        * ) 
        echo "Please answer yes or no."
        ;;
    esac
    

done