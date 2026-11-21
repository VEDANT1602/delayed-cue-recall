"""Execute the previously reserved GRU block. Every row uses the preregistered core code."""
import csv,hashlib
from datetime import datetime,timezone
from timecourse import ROOT,run_pair,torch
CORE='a06e13657ed09ccae83c3a0ea8d8cbd9aba2e2626929cfc63ada0d5cf0f6f8ff'
if __name__=='__main__':
    assert hashlib.sha256((ROOT/'timecourse.py').read_bytes()).hexdigest()==CORE
    st=datetime.now(timezone.utc)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    with (ROOT/'curve_confirm_GRU.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['task','cell','seed','mode','delay','update','correct','n','erased_correct'])
        class TaggedWriter:
            def writerow(self,row):
                w.writerow(['confirmation']+row[1:])
        for seed in range(7900,7916):
            run_pair('corridor','GRU',seed,(None,None),TaggedWriter());f.flush()
    en=datetime.now(timezone.utc)
    print('Execution start:',st.isoformat(),'finish:',en.isoformat())
