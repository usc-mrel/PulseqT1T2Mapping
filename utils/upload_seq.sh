#!/bin/bash

echo $#

if  [[ $# != 3 ]]; then
    echo Wrong number of arguments, should be SeqFilePath ServerSeqDir SSHUser
    exit 1
fi

SeqFilePath=$1 #seqs/my.seq
ServerSeqDir=$2 #/server/sdata_new/Bilal/pulseq_seqs
SSHUser=$3 # btasdelen

echo Given sequence path:$SeqFilePath .

sftp $SSHUser@lauterbur.usc.edu <<_FINFTP_
    cd $ServerSeqDir
    put $SeqFilePath
    quit
_FINFTP_

exit 0
