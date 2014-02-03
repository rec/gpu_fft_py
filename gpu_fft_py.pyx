cdef extern from "mailbox.h":
  int mbox_open()
  void mbox_close(int mb)

cdef extern from "gpu_fft.h":
  cdef struct GPU_FFT_COMPLEX:
    float re, im

  cdef struct GPU_FFT:
    GPU_FFT_COMPLEX* input
    GPU_FFT_COMPLEX* output
    int mb, step
    unsigned timeout, noflush, handle, size, vc_msg

  int gpu_fft_prepare(
      int mb,         # mailbox file_desc
      int log2_N,     # log2(FFT_length) = 8...17
      int direction,  # GPU_FFT_FWD: fft(); GPU_FFT_REV: ifft()
      int jobs,       # number of transforms in batch
      GPU_FFT **fft)

  unsigned gpu_fft_execute(GPU_FFT *info)
  void gpu_fft_release(GPU_FFT *info)


cdef class GpuFft:
  cdef GPU_FFT* thisptr
  cdef int mb

  def __cinit__(self, int log_size, is_forward=True, int jobs=10):
    cdef int forward
    if is_forward:
      forward = 1
    else:
      forward = 0

    assert 8 <= log_size <= 17, 'log_size must be between 8 and 17'
    self.mb = mbox_open()
    gpu_fft_prepare(self.mb, log_size, forward, jobs, &self.thisptr)

  def __dealloc__(self):
    gpu_fft_release(self.thisptr)
    mbox_close(self.mb)
