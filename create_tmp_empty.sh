#!/bin/bash

# make a temp file
TMP_FILE=$(mktemp -q XXXXXX)
if [ $? -ne 0 ]; then
    echo "$0: Can't create temp file, bye.."
    exit 1
fi

# write content to tmp file
echo '#!/bin/bash
'$1 > $TMP_FILE

# run the script according to OS
if [[ "$(uname -s)" == "Linux" ]]; then
    sbatch -c 8 $TMP_FILE
elif [[ "$(uname -s)" == "CYGWIN"* ]] || [[ "$(uname -s)" == "MINGW"* ]] || [[ "$(uname -s)" == "MSYS"* ]]; then
    bash $TMP_FILE
else
    echo "The operating system is not recognized."
fi


rm $TMP_FILE