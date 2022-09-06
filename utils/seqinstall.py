import subprocess
import pathlib
import os

def seqinstall(seq_file_path: str, remote_path:str = "/server/sdata/Bilal/pulseq_seqs", username="btasdelen"):
    script_path = pathlib.Path(__file__).parent.resolve()
    
    subprocess.run([os.path.join(script_path, "upload_seq.sh"), seq_file_path, remote_path, username])