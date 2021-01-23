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
                    tar zxvf "train_labels.tar.gz" -C $NEWDOWNLOADDIR
                    
                    echo "Downloading train_images.tar.gz from $TRAINIMG..."
                    wget -nc -P $NEWDOWNLOADDIR $TRAINIMG
                    tar zxvf "train_images.tar.gz" -C $NEWDOWNLOADDIR
                    
                    echo "Downloading validate_labels.tar.gz from $TESTLABEL..."
                    wget -nc -P $NEWDOWNLOADDIR $TESTLABEL
                    tar zxvf "validate_labels_small.tar.gz" -C $NEWDOWNLOADDIR
                    
                    echo "Downloading validate_images.tar.gz from $TESTIMG..."
                    wget -nc -P $NEWDOWNLOADDIR $TESTIMG
                    tar zxvf "validate_images.tar.gz" -C $NEWDOWNLOADDIR
                    
                    exit
                    ;;
                [Nn]* )
                    # downloading data
                    echo "Downloading train_labels.tar.gz from $TRAINLABEL..."
                    wget -nc -P $DOWNLOADDIR $TRAINLABEL
                    tar zxvf "train_labels.tar.gz" -C $DOWNLOADDIR
                    
                    echo "Downloading train_images.tar.gz from $TRAINIMG..."
                    wget -nc -P $DOWNLOADDIR $TRAINIMG
                    tar zxvf "train_images.tar.gz" -C $DOWNLOADDIR
                    
                    echo "Downloading validate_labels.tar.gz from $TESTLABEL..."
                    wget -nc -P $DOWNLOADDIR $TESTLABEL
                    tar zxvf "validate_labels_small.tar.gz" -C $DOWNLOADDIR
                    
                    echo "Downloading validate_images.tar.gz from $TESTIMG..."
                    wget -nc -P $DOWNLOADDIR $TESTIMG
                    tar zxvf "validate_images.tar.gz" -C $DOWNLOADDIR

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