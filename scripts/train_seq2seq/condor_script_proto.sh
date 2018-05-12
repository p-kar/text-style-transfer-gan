universe = vanilla 
Initialdir = __CODE_ROOT__ 
Executable = /lusr/bin/bash
Arguments = __SAVE_PATH__/run_script.sh
+Group   = "GRAD"
+Project = "__PROJECT__"
+ProjectDescription = "Course Work"
Requirements = TARGET.GPUSlot && CUDAGlobalMemoryMb >= 7000
getenv = True
request_GPUs = 1
+GPUJob = true 

Log =__LOGNAME__/condor.log 
Error = __LOGNAME__/condor.err
Output = __LOGNAME__/condor.out
Notification = complete
Notify_user = __EMAILID__ 
queue 1
