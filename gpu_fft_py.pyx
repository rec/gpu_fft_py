cdef extern from "mailbox.h":
  int mbox_open()
  mbox_close(int mb)


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
