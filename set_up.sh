#!/bin/sh

# Set up variables and local directory
UTIL_REPO_NAME="dl_utilities"
SOA_CNNS_REPO_NAME="start_of_art_cnns"

cd `dirname $0`

# Get necessary repo's
echo -n "Getting repositories needed for the project...  "

# Do utilities one
if [ ! -d ../$UTIL_REPO_NAME ]; then
    if [ -d $UTIL_REPO_NAME ]; then
        rm -rf $UTIL_REPO_NAME
    fi

    git clone https://github.com/alijkhalil/$UTIL_REPO_NAME.git > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Download error!" >&2
        exit 1
    fi

    mv $UTIL_REPO_NAME ../
fi

# Do state of art CNNs one
if [ ! -d ../$SOA_CNNS_REPO_NAME ]; then
    if [ -d $SOA_CNNS_REPO_NAME ]; then
        rm -rf $SOA_CNNS_REPO_NAME
    fi

    git clone https://github.com/alijkhalil/$SOA_CNNS_REPO_NAME.git > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Download error!" >&2
        exit 1
    fi

    mv $SOA_CNNS_REPO_NAME ../
fi

# Print success and exit
echo "Done!"
exit 0
